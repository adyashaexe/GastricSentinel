"""FastAPI app (async, SQLite via SQLModel).

    uvicorn app.app:app --reload          # http://127.0.0.1:8000

The PyTorch Predictor is loaded once at startup. All model work (prediction and Grad-CAM) runs under one
asyncio.Lock, in a worker thread, so the event loop stays responsive and Grad-CAM's hooks never overlap.
Every analysis and every chat message is written to SQLite (app/database.py) and kept; nothing is purged.

Public (no sign-in needed):
    GET  /                       UI                         GET  /api/status       which models are loaded
    POST /api/predict            multipart: image, age, sex, stage, genomic_csv (optional)
    GET  /api/result/<id>        a stored result            GET  /api/model-card   validation facts
    POST /api/chat               assistant                  GET  /report/<id>      PDF report
    GET  /outputs/<name>         stored input / Grad-CAM image
Accounts:
    POST /api/auth/register | login | logout       GET /api/auth/me
    GET  /api/analyses  |  DELETE /api/analyses/<id>  |  GET /api/analyses/<id>/chat
    POST /api/analyses/<id>/review                 GET /api/review/queue?status=pending|reviewed|all
Interactive API docs: /docs

Environment:
    GS_DB                 SQLite path (default <project>/instance/gastricsentinel.db)
    GS_SECRET_KEY         session signing key (default: generated once into <db dir>/secret.key)
    GS_REVIEWER_EMAILS    comma-separated emails that get the reviewer role
    GS_ADMIN_EMAILS       comma-separated emails that get the admin role (also reviewers)
    GS_REQUIRE_LOGIN=1    refuse anonymous predictions
    GS_HTTPS=1            mark the session cookie Secure (set this behind HTTPS)
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import re
import secrets
import shutil
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

from gastricsentinel import chat as chatbot
from gastricsentinel.config import ARTIFACTS, IMAGE_EXTS
from gastricsentinel.inference import InputError, Predictor
from gastricsentinel.model_card import confusion_png, load_model_card, reliability_png
from gastricsentinel.registry import file_version
from gastricsentinel.report_pdf import build_report

from .database import DEFAULT_DB, Store

HERE = Path(__file__).resolve().parent
MAX_UPLOAD = 25 * 1024 * 1024
MAX_SLIDE = 200 * 1024 * 1024
MAX_BATCH = 25
_ID = re.compile(r"^[0-9a-f]{32}$")
_OUT_FILE = re.compile(r"^([0-9a-f]{32})_(gradcam\.png|input\.(png|jpe?g|bmp|tiff?))$")
_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")
_CLASS = re.compile(r"^[A-Za-z0-9_\- ]{1,40}$")
_PW_ITER = 600_000


def parse_genomic_csv(fileobj) -> dict:
    """CSV/TSV with two columns: gene symbol, expression value (header optional)."""
    df = pd.read_csv(fileobj, sep=None, engine="python", header=None, nrows=100_000)
    if df.shape[1] < 2:
        raise InputError("genomic file needs two columns: gene, value")
    df = df.iloc[:, :2]
    if pd.to_numeric(df.iloc[0, 1], errors="coerce") != pd.to_numeric(df.iloc[0, 1], errors="coerce"):
        df = df.iloc[1:]  # header row
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    return {str(g).strip(): float(v) for g, v in zip(df.iloc[:, 0], vals) if v == v}


class RateLimiter:
    """Tiny in-memory sliding-window limiter (per client IP)."""

    def __init__(self, limit: int, window: float = 60.0):
        self.limit, self.window, self.hits = limit, window, defaultdict(deque)

    def allow(self, key: str) -> bool:
        now, q = time.monotonic(), self.hits[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


def _csv_env(name: str) -> set:
    return {e.strip().lower() for e in os.getenv(name, "").split(",") if e.strip()}


def role_of(email: str) -> str:
    """Roles come from the environment, never from the database, so nobody can promote themselves."""
    e = (email or "").lower()
    if e in _csv_env("GS_ADMIN_EMAILS"):
        return "admin"
    if e in _csv_env("GS_REVIEWER_EMAILS"):
        return "reviewer"
    return "user"


def hash_password(pw: str) -> str:
    """PBKDF2-SHA256, stored as pbkdf2:sha256:<iterations>$<salt>$<hex> (the format Werkzeug used, so old hashes still verify)."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), _PW_ITER).hex()
    return f"pbkdf2:sha256:{_PW_ITER}${salt}${dk}"


def check_password(stored: str, pw: str) -> bool:
    try:
        meta, salt, dk = stored.split("$")
        _, algo, iters = meta.split(":")
        calc = hashlib.pbkdf2_hmac(algo, pw.encode(), salt.encode(), int(iters)).hex()
        return hmac.compare_digest(calc, dk)
    except (ValueError, TypeError):
        return False


_DUMMY_HASH = hash_password("not-a-real-password")


def _secret_key(db_dir: Path) -> str:
    if os.getenv("GS_SECRET_KEY"):
        return os.environ["GS_SECRET_KEY"]
    f = db_dir / "secret.key"
    if f.exists():
        return f.read_text().strip()
    k = secrets.token_hex(32)
    db_dir.mkdir(parents=True, exist_ok=True)
    f.write_text(k)
    try:
        f.chmod(0o600)
    except OSError:
        pass
    return k


async def _json(request: Request) -> dict:
    try:
        b = await request.json()
    except Exception:
        return {}
    return b if isinstance(b, dict) else {}


def create_app(artifact_dir=ARTIFACTS, output_dir=None, db_path=None) -> FastAPI:
    out = Path(output_dir or HERE / "static" / "outputs")
    out.mkdir(parents=True, exist_ok=True)
    if db_path is None:
        db_path = os.getenv("GS_DB") or (out / "gastricsentinel.db" if output_dir else DEFAULT_DB)
    store = Store(db_path)
    predict_limit, chat_limit, auth_limit = RateLimiter(20), RateLimiter(40), RateLimiter(10)
    state = {"predictor": None, "error": None, "version": "untrained", "lock": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.init()
        state["lock"] = asyncio.Lock()  # one model, one user of it at a time
        try:
            state["predictor"] = await run_in_threadpool(Predictor, artifact_dir)
            state["version"] = file_version(Path(artifact_dir) / "image" / "image_model.pt")
        except FileNotFoundError as e:
            logging.getLogger("gastricsentinel").warning("models not loaded: %s", e)
            state["error"] = "Models not loaded. Train them first (see README)."
        yield

    app = FastAPI(title="GastricSentinel", lifespan=lifespan)
    app.state.store, app.state.state, app.state.out = store, state, out
    app.add_middleware(SessionMiddleware, secret_key=_secret_key(Path(db_path).parent), same_site="lax",
                       https_only=os.getenv("GS_HTTPS") == "1", max_age=14 * 24 * 3600, session_cookie="gs_session")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    # the templates were written for Flask: url_for('static', filename='css/app.css')
    templates.env.globals["url_for"] = lambda endpoint, **kw: f"/{endpoint}/{kw.get('filename') or kw.get('path', '')}"

    # ------------------------------------------------------------ helpers
    def ip(request: Request) -> str:
        return request.client.host if request.client else "?"

    def current_user(request: Request):
        uid = request.session.get("uid")
        if not uid:
            return None
        u = store.user(uid)
        if u is None:
            request.session.clear()
            return None
        return {"id": u["id"], "email": u["email"], "name": u["name"], "role": role_of(u["email"])}

    def can_view(a: dict, user) -> bool:
        if a["user_id"] is None:  # anonymous: anyone holding the unguessable id
            return True
        return bool(user and (user["id"] == a["user_id"] or user["role"] in ("reviewer", "admin")))

    def need_login(request: Request) -> dict:
        u = current_user(request)
        if u is None:
            raise HTTPException(401, "please sign in")
        return u

    def need_reviewer(request: Request) -> dict:
        u = need_login(request)
        if u["role"] not in ("reviewer", "admin"):
            raise HTTPException(403, "reviewer access required")
        return u

    def load(rid: str, user):
        """Return the analysis or 404. Wrong-owner and missing look identical."""
        a = store.get_analysis(rid) if _ID.match(rid) else None
        if a is None or not can_view(a, user):
            raise HTTPException(404, "not found")
        return a

    def present(a: dict, user) -> dict:
        res = dict(a["result"])
        owned = a["user_id"] is not None
        res["model_version"] = a["model_version"]
        res["reviews"] = store.reviews_for(a["id"]) if owned else []
        res["viewer"] = {"signed_in": bool(user), "owned": owned,
                         "is_owner": bool(user and owned and user["id"] == a["user_id"]),
                         "can_review": bool(user and owned and user["role"] in ("reviewer", "admin"))}
        return res

    def remove_files(rid: str):
        for f in out.glob(f"{rid}*"):
            try:
                f.unlink()
            except OSError:
                pass

    # -------------------------------------------------------- safety nets
    @app.exception_handler(StarletteHTTPException)
    async def http_err(request: Request, exc: StarletteHTTPException):
        return JSONResponse({"error": exc.detail if isinstance(exc.detail, str) else "request failed"}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError):
        return JSONResponse({"error": "invalid request"}, status_code=400)

    @app.middleware("http")
    async def guards(request: Request, call_next):
        # Browsers always send Origin on cross-site POSTs; refuse any that does not match this host.
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            origin = request.headers.get("origin")
            if origin is not None and (origin == "null" or urlparse(origin).netloc != request.headers.get("host")):
                return JSONResponse({"error": "cross-site request blocked"}, status_code=403)
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        if request.url.path.startswith(("/api/", "/outputs/", "/report/")):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    # --------------------------------------------------------------- pages
    @app.get("/", include_in_schema=False)
    def index(request: Request):
        return templates.TemplateResponse(request, "index.html")

    @app.get("/api/status")
    def status():
        p = state["predictor"]
        if p is None:
            return JSONResponse({"ready": False, "error": state["error"]}, status_code=503)
        return {"ready": True, "classes": p.classes, "subtype_models": sorted(p.subtype), "fusion": p.fusion is not None,
                "image_model": p.image_meta, "model_version": state["version"],
                "require_login": os.getenv("GS_REQUIRE_LOGIN") == "1"}

    # ---------------------------------------------------------------- auth
    @app.post("/api/auth/register", status_code=201)
    async def register(request: Request):
        if not auth_limit.allow(ip(request)):
            raise HTTPException(429, "too many attempts, wait a minute")
        b = await _json(request)
        email, name, pw = str(b.get("email", "")).strip().lower(), str(b.get("name", "")).strip(), str(b.get("password", ""))
        if not _EMAIL.match(email):
            raise HTTPException(400, "enter a valid email address")
        if not 1 <= len(name) <= 80:
            raise HTTPException(400, "enter your name")
        if len(pw) < 8 or len(pw) > 200:
            raise HTTPException(400, "password must be at least 8 characters")
        uid = await run_in_threadpool(lambda: store.create_user(email, name, hash_password(pw)))
        if uid is None:
            raise HTTPException(409, "that email already has an account, try signing in")
        request.session.clear()
        request.session["uid"] = uid
        return {"user": current_user(request)}

    @app.post("/api/auth/login")
    async def login(request: Request):
        if not auth_limit.allow(ip(request)):
            raise HTTPException(429, "too many attempts, wait a minute")
        b = await _json(request)
        email, pw = str(b.get("email", "")).strip().lower(), str(b.get("password", ""))
        u = store.user_by_email(email)
        # always run a hash check so timing doesn't reveal whether the email exists
        ok = await run_in_threadpool(check_password, u["pw_hash"] if u else _DUMMY_HASH, pw)
        if not (ok and u):
            raise HTTPException(401, "email or password is incorrect")
        request.session.clear()
        request.session["uid"] = u["id"]
        return {"user": current_user(request)}

    @app.post("/api/auth/logout")
    def logout(request: Request):
        request.session.clear()
        return {"ok": True}

    @app.get("/api/auth/me")
    def me(request: Request):
        return {"user": current_user(request)}

    # ------------------------------------------------------------- predict
    def gate(request: Request, cost: int = 1):
        """Shared checks for every endpoint that runs the model."""
        p = state["predictor"]
        if p is None:
            raise HTTPException(503, state["error"] or "models not loaded")
        user = current_user(request)
        if os.getenv("GS_REQUIRE_LOGIN") == "1" and user is None:
            raise HTTPException(401, "please sign in to run an analysis")
        for _ in range(cost):
            if not predict_limit.allow(ip(request)):
                raise HTTPException(429, "too many requests, slow down")
        return p, user

    async def save_upload(upload: UploadFile, ext: str, limit: int) -> tuple[str, Path]:
        rid = uuid.uuid4().hex
        path = out / f"{rid}_input{ext}"

        def write():
            with open(path, "wb") as fh:
                shutil.copyfileobj(upload.file, fh)
            return path.stat().st_size
        if await run_in_threadpool(write) > limit:
            path.unlink(missing_ok=True)
            raise HTTPException(413, f"file is larger than {limit // (1024 * 1024)} MB")
        return rid, path

    def check_ext(upload: UploadFile | None) -> str:
        if upload is None or not upload.filename:
            raise HTTPException(400, "image file is required")
        ext = Path(upload.filename).suffix.lower()
        if ext not in IMAGE_EXTS:
            raise HTTPException(400, f"unsupported image type '{ext}'")
        return ext

    async def analyse(p, user, upload: UploadFile, clinical=None, genomic=None) -> dict:
        ext = check_ext(upload)
        rid, img_path = await save_upload(upload, ext, MAX_UPLOAD)
        try:
            async with state["lock"]:  # Grad-CAM hooks mutate the shared model
                result = await run_in_threadpool(p.predict, img_path, clinical, genomic)
                cam = await run_in_threadpool(p.gradcam, img_path)
        except InputError as e:
            img_path.unlink(missing_ok=True)
            raise HTTPException(400, str(e))
        except Exception:
            img_path.unlink(missing_ok=True)
            logging.getLogger("gastricsentinel").exception("prediction failed")
            raise HTTPException(500, "prediction failed (see server log)")
        cam.save(out / f"{rid}_gradcam.png")
        result.update(id=rid, gradcam_url=f"/outputs/{rid}_gradcam.png", input_ext=ext, input_url=f"/outputs/{rid}_input{ext}",
                      created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), model_version=state["version"])
        await run_in_threadpool(store.add_analysis, rid, user["id"] if user else None, result)
        return present(store.get_analysis(rid), user)

    @app.post("/api/predict")
    async def predict(request: Request, image: UploadFile | None = File(None), genomic_csv: UploadFile | None = File(None),
                      age: str | None = Form(None), sex: str | None = Form(None), stage: str | None = Form(None)):
        p, user = gate(request)
        check_ext(image)
        try:
            genomic = parse_genomic_csv(genomic_csv.file) if genomic_csv is not None and genomic_csv.filename else None
        except InputError as e:
            raise HTTPException(400, str(e))
        return await analyse(p, user, image, {"age": age, "sex": sex, "stage": stage}, genomic)

    @app.post("/api/batch")
    async def batch(request: Request, images: list[UploadFile] = File(None)):
        """Up to 25 images, analysed one after another. Each becomes its own saved analysis."""
        images = [f for f in (images or []) if f.filename]
        if not images:
            raise HTTPException(400, "choose at least one image")
        if len(images) > MAX_BATCH:
            raise HTTPException(400, f"at most {MAX_BATCH} images per batch")
        p, user = gate(request, cost=1)
        items = []
        for f in images:
            try:
                if not predict_limit.allow(ip(request)):
                    raise HTTPException(429, "too many requests, slow down")
                r = await analyse(p, user, f)
                items.append({"filename": f.filename, "ok": True, "result": r})
            except HTTPException as e:
                items.append({"filename": f.filename, "ok": False, "error": e.detail})
        good = [i["result"] for i in items if i["ok"]]
        counts: dict = {}
        for r in good:
            counts[r["prediction"]["class"]] = counts.get(r["prediction"]["class"], 0) + 1
        summary = {"n": len(items), "ok": len(good), "failed": len(items) - len(good), "class_counts": counts,
                   "low_confidence": sum(1 for r in good if r["prediction"]["low_confidence"]),
                   "ood_flagged": sum(1 for r in good if (r.get("ood") or {}).get("flagged")),
                   "mean_confidence": (sum(r["prediction"]["confidence"] for r in good) / len(good)) if good else None,
                   "saved_to_account": bool(user)}
        return {"items": items, "summary": summary}

    @app.post("/api/slide")
    async def slide(request: Request, image: UploadFile | None = File(None), tile: int = Form(256)):
        """Whole-slide mode: tile a large image, classify each tissue tile, return a map and a summary."""
        p, user = gate(request)
        ext = check_ext(image)
        rid, big = await save_upload(image, ext, MAX_SLIDE)
        try:
            async with state["lock"]:
                result, thumb, overlay = await run_in_threadpool(p.predict_slide, big, tile)
        except InputError as e:
            raise HTTPException(400, str(e))
        except Exception:
            logging.getLogger("gastricsentinel").exception("slide analysis failed")
            raise HTTPException(500, "slide analysis failed (see server log)")
        finally:
            big.unlink(missing_ok=True)  # the full-size upload is not kept; a 1600 px preview is
        thumb.save(out / f"{rid}_input.png")
        overlay.save(out / f"{rid}_gradcam.png")
        result.update(id=rid, gradcam_url=f"/outputs/{rid}_gradcam.png", input_ext=".png", input_url=f"/outputs/{rid}_input.png",
                      created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), model_version=state["version"])
        await run_in_threadpool(store.add_analysis, rid, user["id"] if user else None, result)
        return present(store.get_analysis(rid), user)

    # ------------------------------------------------------ saved analyses
    @app.get("/api/result/{rid}")
    def get_result(rid: str, request: Request):
        u = current_user(request)
        return present(load(rid, u), u)

    @app.get("/api/analyses")
    def my_analyses(request: Request):
        rows = store.list_analyses(need_login(request)["id"])
        for r in rows:
            r["has_report"] = (out / f"{r['id']}.pdf").exists()
        return {"analyses": rows}

    @app.delete("/api/analyses/{rid}")
    def delete_analysis(rid: str, request: Request):
        u = need_login(request)
        a = load(rid, u)
        if a["user_id"] is None or not (u["id"] == a["user_id"] or u["role"] == "admin"):
            raise HTTPException(403, "only the owner can delete this analysis")
        store.delete_analysis(rid)
        remove_files(rid)
        return {"ok": True}

    @app.get("/api/analyses/{rid}/chat")
    def saved_chat(rid: str, request: Request):
        u = need_login(request)
        a = load(rid, u)
        if a["user_id"] is None or not (u["id"] == a["user_id"] or u["role"] == "admin"):
            raise HTTPException(403, "only the owner can read this conversation")
        return {"messages": store.chat_for(rid)}

    # ------------------------------------------------------------- reviews
    @app.post("/api/analyses/{rid}/review")
    async def review(rid: str, request: Request):
        u = need_reviewer(request)
        a = load(rid, u)
        if a["user_id"] is None:
            raise HTTPException(400, "only saved analyses can be reviewed")
        b = await _json(request)
        verdict = b.get("verdict")
        if verdict not in ("agree", "disagree", "uncertain"):
            raise HTTPException(400, "verdict must be agree, disagree or uncertain")
        corrected = (b.get("corrected_class") or "").strip() or None
        if verdict == "disagree":
            known = state["predictor"].classes if state["predictor"] else None
            if not corrected or not _CLASS.match(corrected) or (known and corrected not in known):
                raise HTTPException(400, "choose the class you believe is correct")
        else:
            corrected = None
        note = str(b.get("note") or "").strip()[:1000] or None
        store.upsert_review(rid, u["id"], verdict, corrected, note)
        return {"reviews": store.reviews_for(rid)}

    @app.get("/api/review/queue")
    def queue(request: Request, status: str = "pending"):
        need_reviewer(request)
        return {"items": store.review_queue(status if status in ("pending", "reviewed", "all") else "pending")}

    # ---------------------------------------------------------------- chat
    @app.post("/api/chat")
    async def chat(request: Request):
        if not chat_limit.allow(ip(request)):
            raise HTTPException(429, "too many requests, slow down")
        body = await _json(request)
        rid, result, a, user = body.get("result_id"), None, None, current_user(request)
        if isinstance(rid, str) and _ID.match(rid):
            a = store.get_analysis(rid)
            if a is not None and can_view(a, user):
                result = a["result"]
            else:
                a = None
        history = body.get("history") if isinstance(body.get("history"), list) else []
        message = str(body.get("message", ""))
        reply = await run_in_threadpool(chatbot.reply, message, result, history)
        if message.strip():  # every message is logged, tied to the analysis and the account when there are any
            aid, uid = (rid if a else None), (user["id"] if user else None)
            await run_in_threadpool(store.add_chat, aid, uid, "user", message.strip()[:1000])
            await run_in_threadpool(store.add_chat, aid, uid, "assistant", str(reply.get("reply", ""))[:4000], reply.get("mode"))
        return reply

    # --------------------------------------------------------------- files
    @app.get("/api/model-card")
    def model_card():
        return load_model_card(artifact_dir)

    @app.get("/outputs/{name}")
    def output_file(name: str, request: Request):
        m = _OUT_FILE.match(name)
        if not m:
            raise HTTPException(404, "not found")
        load(m.group(1), current_user(request))  # same access rule as the analysis it belongs to
        f = out / name
        if not f.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(f)

    @app.get("/report/{rid}")
    def report(rid: str, request: Request):
        a = load(rid, current_user(request))
        pdf = build_report(a["result"], out / f"{rid}.pdf", out / f"{rid}_input{a['result']['input_ext']}",
                           out / f"{rid}_gradcam.png", report_id=rid[:8],
                           model_card=load_model_card(artifact_dir), confusion_png=confusion_png(artifact_dir),
                           reliability_png=reliability_png(artifact_dir), reviews=store.reviews_for(rid) if a["user_id"] else None)
        return FileResponse(pdf, media_type="application/pdf", filename=f"gastricsentinel_{rid[:8]}.pdf")

    # -------------------------------------------------------------- export
    def flat(a: dict) -> dict:
        r, p = a["result"], a["result"]["prediction"]
        row = {"id": a["id"], "created": a["created"], "mode": a["mode"], "model_version": a["model_version"],
               "predicted_class": p["class"], "confidence": round(p["confidence"], 4), "low_confidence": p["low_confidence"],
               "tumour_probability": None if a["tumour_prob"] is None else round(a["tumour_prob"], 4),
               "outside_training_distribution": bool((r.get("ood") or {}).get("flagged")),
               "molecular_subtype": (r.get("molecular_subtype") or {}).get("class"), "warnings": " | ".join(r.get("warnings") or [])}
        for c, v in p["probabilities"].items():
            row[f"p_{c}"] = round(v, 4)
        sl = r.get("slide")
        if sl:
            row.update(slide_tissue_tiles=sl["n_tissue_tiles"], slide_tumour_fraction=sl.get("tumour_fraction"))
        if a["user_id"] is not None:
            rv = store.reviews_for(a["id"])
            row["reviews"] = "; ".join(f"{x['reviewer']}: {x['verdict']}" + (f" ({x['corrected_class']})" if x["corrected_class"] else "") for x in rv)
        return row

    @app.get("/api/export")
    def export(request: Request, ids: str = "", format: str = "csv"):
        """Download analyses as CSV or JSON. ids=a,b,c (anonymous ids work); without ids, all of your saved analyses."""
        u = current_user(request)
        if ids:
            want = [i for i in ids.split(",") if _ID.match(i)][:500]
            rows = [a for a in (store.get_analysis(i) for i in want) if a and can_view(a, u)]
        else:
            need_login(request)
            rows = [store.get_analysis(r["id"]) for r in store.list_analyses(u["id"])]
        if not rows:
            raise HTTPException(404, "nothing to export")
        stamp = time.strftime("%Y%m%d-%H%M")
        if format == "json":
            return JSONResponse([present(a, u) for a in rows], headers={"Content-Disposition": f'attachment; filename="gastricsentinel-{stamp}.json"'})
        import csv
        import io as _io
        # spreadsheet formula injection: reviewer-supplied text must not start with = + - @
        safe = lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r") else v  # noqa: E731
        flats = [{k: safe(v) for k, v in flat(a).items()} for a in rows]
        cols = list(dict.fromkeys(k for f_ in flats for k in f_))
        buf = _io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        w.writerows(flats)
        from fastapi.responses import Response
        return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="gastricsentinel-{stamp}.csv"'})

    app.db = store  # exposed for tests and management scripts
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.app:app", host="127.0.0.1", port=8000, reload=False)
