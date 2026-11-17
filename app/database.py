"""SQLite engine (WAL mode) and the queries the app needs.

SQLite is a single local file, needs no server, and WAL mode lets readers and a writer work at the same time.
Every call opens a short-lived Session, so the Store is safe to use from FastAPI's worker threads.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from sqlalchemy import event, func, text
from sqlmodel import Session, SQLModel, col, create_engine, select

from .models import ChatLog, PredictionResult, Review, User

DEFAULT_DB = Path(__file__).resolve().parents[1] / "instance" / "gastricsentinel.db"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_engine(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False, "timeout": 15})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")      # concurrent readers + one writer
        cur.execute("PRAGMA synchronous=NORMAL")    # safe with WAL, much faster commits
        cur.execute("PRAGMA foreign_keys=ON")       # enforce ON DELETE CASCADE
        cur.close()
    return engine


def init_db(engine) -> None:
    SQLModel.metadata.create_all(engine)
    _migrate_legacy(engine)


def _migrate_legacy(engine) -> None:
    """Databases from the Flask version used the tables analyses / chat_messages / reviews. Copy them over once."""
    with engine.begin() as c:
        have = {r[0] for r in c.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
        empty = lambda t: c.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar() == 0  # noqa: E731
        if "analyses" in have and empty("prediction_results"):
            c.execute(text("INSERT INTO prediction_results(id,user_id,created,pred_class,confidence,tumour_prob,mode,model_version,input_ext,result)"
                           " SELECT id,user_id,created,pred_class,confidence,tumour_prob,mode,model_version,input_ext,result FROM analyses"))
            if "chat_messages" in have and empty("chat_logs"):
                c.execute(text("INSERT INTO chat_logs(analysis_id,user_id,role,content,mode,created)"
                               " SELECT analysis_id,user_id,role,content,mode,created FROM chat_messages"))
            if "reviews" in have and empty("analysis_reviews"):
                c.execute(text("INSERT INTO analysis_reviews(analysis_id,reviewer_id,verdict,corrected_class,note,created)"
                               " SELECT analysis_id,reviewer_id,verdict,corrected_class,note,created FROM reviews"))


def _d(row, *drop) -> Optional[dict]:
    return None if row is None else {k: v for k, v in row.model_dump().items() if k not in drop}


class Store:
    def __init__(self, path=None):
        self.path = Path(path or DEFAULT_DB)
        self.engine = make_engine(self.path)

    def init(self):
        init_db(self.engine)

    def session(self) -> Session:
        return Session(self.engine, expire_on_commit=False)

    # ------------------------------------------------------------- users
    def create_user(self, email: str, name: str, pw_hash: str) -> Optional[int]:
        with self.session() as s:
            if s.exec(select(User).where(func.lower(User.email) == email.lower())).first():
                return None
            u = User(email=email, name=name, pw_hash=pw_hash, created=now())
            s.add(u)
            try:
                s.commit()
            except Exception:  # lost a race with another registration of the same email
                s.rollback()
                return None
            return u.id

    def user_by_email(self, email: str) -> Optional[dict]:
        with self.session() as s:
            return _d(s.exec(select(User).where(func.lower(User.email) == email.lower())).first())

    def user(self, uid: int) -> Optional[dict]:
        with self.session() as s:
            return _d(s.get(User, uid))

    # ---------------------------------------------------------- analyses
    def add_analysis(self, rid: str, user_id: Optional[int], result: dict) -> None:
        p = result["prediction"]
        with self.session() as s:
            s.add(PredictionResult(id=rid, user_id=user_id, created=now(), pred_class=p["class"], confidence=float(p["confidence"]),
                                   tumour_prob=result.get("tumour_probability"), mode=result.get("mode"),
                                   model_version=result.get("model_version"), input_ext=result.get("input_ext"), result=result))
            s.commit()

    def get_analysis(self, rid: str) -> Optional[dict]:
        with self.session() as s:
            return _d(s.get(PredictionResult, rid))

    def list_analyses(self, user_id: int, limit: int = 500) -> list[dict]:
        with self.session() as s:
            n = func.count(Review.id)
            q = (select(PredictionResult.id, PredictionResult.created, PredictionResult.pred_class, PredictionResult.confidence,
                        PredictionResult.input_ext, PredictionResult.model_version, PredictionResult.mode, PredictionResult.tumour_prob,
                        n.label("n_reviews"))
                 .outerjoin(Review, Review.analysis_id == PredictionResult.id)
                 .where(PredictionResult.user_id == user_id).group_by(PredictionResult.id)
                 .order_by(col(PredictionResult.created).desc()).limit(limit))
            return [dict(r._mapping) for r in s.exec(q)]

    def delete_analysis(self, rid: str) -> None:
        with self.session() as s:
            row = s.get(PredictionResult, rid)
            if row:
                s.delete(row)
                s.commit()

    # -------------------------------------------------------------- chat
    def add_chat(self, aid: Optional[str], uid: Optional[int], role: str, content: str, mode=None) -> None:
        with self.session() as s:
            s.add(ChatLog(analysis_id=aid, user_id=uid, role=role, content=content, mode=mode, created=now()))
            s.commit()

    def chat_for(self, aid: str, limit: int = 200) -> list[dict]:
        with self.session() as s:
            rows = s.exec(select(ChatLog).where(ChatLog.analysis_id == aid).order_by(ChatLog.id).limit(limit)).all()
            return [{"role": r.role, "content": r.content, "mode": r.mode, "created": r.created} for r in rows]

    # ----------------------------------------------------------- reviews
    def upsert_review(self, aid: str, reviewer_id: int, verdict: str, corrected, note) -> None:
        with self.session() as s:
            r = s.exec(select(Review).where(Review.analysis_id == aid, Review.reviewer_id == reviewer_id)).first()
            if r is None:
                r = Review(analysis_id=aid, reviewer_id=reviewer_id, verdict=verdict, created=now())
            r.verdict, r.corrected_class, r.note, r.created = verdict, corrected, note, now()
            s.add(r)
            s.commit()

    def reviews_for(self, aid: str) -> list[dict]:
        with self.session() as s:
            q = (select(Review, User.name).join(User, User.id == Review.reviewer_id).where(Review.analysis_id == aid).order_by(Review.id))
            return [{"reviewer_id": r.reviewer_id, "reviewer": name, "verdict": r.verdict, "corrected_class": r.corrected_class,
                     "note": r.note, "created": r.created} for r, name in s.exec(q)]

    def review_queue(self, status: str = "pending", limit: int = 100) -> list[dict]:
        with self.session() as s:
            n = func.count(Review.id)
            q = (select(PredictionResult.id, PredictionResult.created, PredictionResult.pred_class, PredictionResult.confidence,
                        User.name.label("owner"), n.label("n"))
                 .join(User, User.id == PredictionResult.user_id).outerjoin(Review, Review.analysis_id == PredictionResult.id)
                 .where(PredictionResult.user_id.is_not(None)).group_by(PredictionResult.id).order_by(col(PredictionResult.created).desc()))
            if status == "pending":
                q = q.having(n == 0)
            elif status == "reviewed":
                q = q.having(n > 0)
            return [dict(r._mapping) | {"n_reviews": r.n} for r in s.exec(q.limit(limit))]
