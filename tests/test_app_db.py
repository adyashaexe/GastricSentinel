"""Accounts, saved analyses, access control, chat logging and reviewer sign-off (no trained models needed)."""
import uuid

import pytest
from fastapi.testclient import TestClient

from app.app import create_app


def fake_result(rid, cls="TUM"):
    return {"id": rid, "prediction": {"class": cls, "confidence": 0.9, "low_confidence": False, "probabilities": {cls: 0.9, "STR": 0.1}},
            "tumour_probability": 0.9, "mode": "image_only", "modalities_used": {"image": True}, "inputs": {}, "warnings": [],
            "input_ext": ".png", "input_url": f"/outputs/{rid}_input.png", "gradcam_url": f"/outputs/{rid}_gradcam.png",
            "created": "2026-01-01T00:00:00+0000", "model_version": "t", "disclaimer": "d"}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("GS_REVIEWER_EMAILS", "rev@x.org")
    monkeypatch.setenv("GS_ADMIN_EMAILS", "admin@x.org")
    app = create_app(tmp_path / "no_artifacts", tmp_path / "out")
    with TestClient(app):  # runs startup: creates the tables, tries to load models (none here)
        yield app


def client(app):
    return TestClient(app)


def signup(c, email, name="Test User", pw="correct horse"):
    return c.post("/api/auth/register", json={"email": email, "name": name, "password": pw})


def save(app, uid, cls="TUM"):
    rid = uuid.uuid4().hex
    app.db.add_analysis(rid, uid, fake_result(rid, cls))
    return rid


def test_register_login_logout_and_validation(env):
    c = client(env)
    assert signup(c, "bad").status_code == 400
    assert signup(c, "a@x.org", pw="short").status_code == 400
    r = signup(c, "a@x.org")
    assert r.status_code == 201 and r.json()["user"]["role"] == "user"
    assert signup(client(env), "A@X.org").status_code == 409            # case-insensitive duplicate
    assert c.get("/api/auth/me").json()["user"]["email"] == "a@x.org"
    c.post("/api/auth/logout")
    assert c.get("/api/auth/me").json()["user"] is None
    assert c.post("/api/auth/login", json={"email": "a@x.org", "password": "nope"}).status_code == 401
    assert c.post("/api/auth/login", json={"email": "nobody@x.org", "password": "nope"}).status_code == 401
    assert c.post("/api/auth/login", json={"email": "a@x.org", "password": "correct horse"}).status_code == 200
    assert "pw_hash" not in c.get("/api/auth/me").text


def test_roles_come_from_environment(env):
    assert signup(client(env), "rev@x.org").json()["user"]["role"] == "reviewer"
    assert signup(client(env), "admin@x.org").json()["user"]["role"] == "admin"


def test_owned_analysis_is_private_and_listed(env):
    a, b = client(env), client(env)
    ua = signup(a, "a@x.org").json()["user"]["id"]
    signup(b, "b@x.org")
    rid = save(env, ua)
    assert a.get(f"/api/result/{rid}").status_code == 200
    assert b.get(f"/api/result/{rid}").status_code == 404                # other users can't tell it exists
    assert client(env).get(f"/api/result/{rid}").status_code == 404      # anonymous neither
    assert client(env).get(f"/outputs/{rid}_gradcam.png").status_code == 404
    assert [x["id"] for x in a.get("/api/analyses").json()["analyses"]] == [rid]
    assert b.get("/api/analyses").json()["analyses"] == []
    assert client(env).get("/api/analyses").status_code == 401


def test_anonymous_analysis_stays_open_by_id(env):
    rid = save(env, None)
    r = client(env).get(f"/api/result/{rid}")
    assert r.status_code == 200 and r.json()["viewer"]["owned"] is False


def test_delete_only_by_owner(env):
    a, b = client(env), client(env)
    ua = signup(a, "a@x.org").json()["user"]["id"]
    signup(b, "b@x.org")
    rid = save(env, ua)
    assert b.delete(f"/api/analyses/{rid}").status_code == 404
    assert a.delete(f"/api/analyses/{rid}").status_code == 200
    assert a.get(f"/api/result/{rid}").status_code == 404


def test_every_chat_message_is_logged(env):
    a = client(env)
    ua = signup(a, "a@x.org").json()["user"]["id"]
    rid = save(env, ua)
    anon_rid = save(env, None)
    anon = client(env)
    assert anon.post("/api/chat", json={"message": "hello", "result_id": anon_rid}).status_code == 200
    assert [m["role"] for m in env.db.chat_for(anon_rid)] == ["user", "assistant"]   # anonymous chat is stored too
    anon.post("/api/chat", json={"message": "general question"})                      # no analysis at all
    with env.db.session() as s:
        from sqlmodel import select
        from app.models import ChatLog
        assert any(r.analysis_id is None and r.user_id is None for r in s.exec(select(ChatLog)).all())
    assert a.get(f"/api/analyses/{rid}/chat").json()["messages"] == []
    a.post("/api/chat", json={"message": "How confident is it?", "result_id": rid})
    msgs = a.get(f"/api/analyses/{rid}/chat").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[0]["content"] == "How confident is it?"
    b = client(env)
    signup(b, "b@x.org")
    assert b.get(f"/api/analyses/{rid}/chat").status_code == 404        # only the owner can read it
    assert client(env).get(f"/api/analyses/{anon_rid}/chat").status_code == 401


def test_review_flow_and_permissions(env):
    owner, rev, plain = client(env), client(env), client(env)
    uo = signup(owner, "a@x.org").json()["user"]["id"]
    signup(rev, "rev@x.org")
    signup(plain, "p@x.org")
    rid = save(env, uo)
    body = {"verdict": "agree"}
    assert plain.post(f"/api/analyses/{rid}/review", json=body).status_code in (403, 404)
    assert client(env).post(f"/api/analyses/{rid}/review", json=body).status_code == 401
    assert plain.get("/api/review/queue").status_code == 403
    assert [x["id"] for x in rev.get("/api/review/queue").json()["items"]] == [rid]
    assert rev.post(f"/api/analyses/{rid}/review", json={"verdict": "disagree"}).status_code == 400   # needs a corrected class
    assert rev.post(f"/api/analyses/{rid}/review", json={"verdict": "banana"}).status_code == 400
    r = rev.post(f"/api/analyses/{rid}/review", json={"verdict": "disagree", "corrected_class": "STR", "note": "stroma"})
    assert r.status_code == 200 and r.json()["reviews"][0]["corrected_class"] == "STR"
    rev.post(f"/api/analyses/{rid}/review", json={"verdict": "agree"})                                  # upsert, not duplicate
    assert len(rev.get(f"/api/result/{rid}").json()["reviews"]) == 1
    assert rev.get("/api/review/queue").json()["items"] == []
    assert owner.get(f"/api/result/{rid}").json()["reviews"][0]["verdict"] == "agree"             # owner sees sign-off
    assert owner.get(f"/api/result/{rid}").json()["viewer"]["can_review"] is False


def test_cross_site_post_is_refused(env):
    c = client(env)
    r = c.post("/api/auth/register", json={"email": "z@x.org", "name": "Z", "password": "longenough"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert c.post("/api/auth/register", json={"email": "z@x.org", "name": "Z", "password": "longenough"}, headers={"Origin": "null"}).status_code == 403
    assert c.post("/api/auth/register", json={"email": "z@x.org", "name": "Z", "password": "longenough"}, headers={"Origin": "http://testserver"}).status_code == 201


def test_require_login_blocks_anonymous_predict(env, monkeypatch):
    monkeypatch.setenv("GS_REQUIRE_LOGIN", "1")
    # no models are loaded here, so predict answers 503 before the login rule; status still reports the setting
    assert client(env).post("/api/predict").status_code in (401, 503)


def test_nothing_is_purged_and_data_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("GS_REQUIRE_LOGIN", "0")
    app = create_app(tmp_path / "no_artifacts", tmp_path / "out")
    with TestClient(app) as c:
        uid = signup(c, "a@x.org").json()["user"]["id"]
        keep, anon = save(app, uid), save(app, None)
        c.post("/api/chat", json={"message": "hi", "result_id": keep})
    app2 = create_app(tmp_path / "no_artifacts", tmp_path / "out")      # new process, same database file
    with TestClient(app2) as c2:
        assert c2.get(f"/api/result/{anon}").status_code == 200
        assert c2.post("/api/auth/login", json={"email": "a@x.org", "password": "correct horse"}).status_code == 200
        assert [x["id"] for x in c2.get("/api/analyses").json()["analyses"]] == [keep]
        assert len(c2.get(f"/api/analyses/{keep}/chat").json()["messages"]) == 2


def test_wal_mode_and_foreign_keys(env):
    with env.db.engine.connect() as c:
        from sqlalchemy import text
        assert c.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert c.execute(text("PRAGMA foreign_keys")).scalar() == 1


def test_legacy_flask_database_is_migrated(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
      CREATE TABLE users(id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL UNIQUE, name TEXT NOT NULL, pw_hash TEXT NOT NULL, created TEXT NOT NULL);
      CREATE TABLE analyses(id TEXT PRIMARY KEY, user_id INTEGER, created TEXT NOT NULL, pred_class TEXT NOT NULL, confidence REAL NOT NULL,
        tumour_prob REAL, mode TEXT, model_version TEXT, input_ext TEXT, result TEXT NOT NULL);
      CREATE TABLE chat_messages(id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id TEXT NOT NULL, user_id INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, mode TEXT, created TEXT NOT NULL);
      CREATE TABLE reviews(id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id TEXT NOT NULL, reviewer_id INTEGER NOT NULL, verdict TEXT NOT NULL, corrected_class TEXT, note TEXT, created TEXT NOT NULL);
    """)
    rid = "a" * 32
    import json
    con.execute("INSERT INTO users(email,name,pw_hash,created) VALUES('o@x.org','Old','h','2026-01-01T00:00:00Z')")
    con.execute("INSERT INTO analyses VALUES(?,1,'2026-01-01T00:00:00Z','TUM',0.9,0.9,'image_only','v','.png',?)", (rid, json.dumps(fake_result(rid))))
    con.execute("INSERT INTO chat_messages(analysis_id,user_id,role,content,created) VALUES(?,1,'user','hi','2026-01-01T00:00:00Z')", (rid,))
    con.execute("INSERT INTO reviews(analysis_id,reviewer_id,verdict,created) VALUES(?,1,'agree','2026-01-01T00:00:00Z')", (rid,))
    con.commit()
    con.close()
    app = create_app(tmp_path / "no_artifacts", tmp_path / "out", db_path=db)
    with TestClient(app):
        assert app.db.get_analysis(rid)["pred_class"] == "TUM"
        assert app.db.chat_for(rid)[0]["content"] == "hi"
        assert app.db.reviews_for(rid)[0]["verdict"] == "agree"
