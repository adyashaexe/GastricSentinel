"""Flask web app.

    python -m app.app            # http://127.0.0.1:5000

Endpoints
    GET  /                 UI
    GET  /api/status       which models are loaded
    POST /api/predict      multipart: image (required), age, sex, stage, genomic_csv (optional)
    GET  /report/<id>      PDF report for a previous prediction
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory

from gastricsentinel import chat as chatbot
from gastricsentinel.config import ARTIFACTS, IMAGE_EXTS
from gastricsentinel.inference import InputError, Predictor
from gastricsentinel.model_card import confusion_png, load_model_card
from gastricsentinel.report_pdf import build_report

HERE = Path(__file__).resolve().parent
_ID = re.compile(r"^[0-9a-f]{32}$")
_OUT_FILE = re.compile(r"^[0-9a-f]{32}_(gradcam\.png|input\.(png|jpe?g|bmp|tiff?))$")


def parse_genomic_csv(file_storage) -> dict:
    """CSV/TSV with two columns: gene symbol, expression value (header optional)."""
    df = pd.read_csv(file_storage, sep=None, engine="python", header=None, nrows=100_000)
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


def purge_old(out: Path, hours: float) -> None:
    """Delete stored uploads/results older than `hours` (privacy + disk hygiene)."""
    cutoff = time.time() - hours * 3600
    for f in out.glob("*"):
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def create_app(artifact_dir=ARTIFACTS, output_dir=None) -> Flask:
    app = Flask(__name__, template_folder=str(HERE / "templates"), static_folder=str(HERE / "static"))
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
    out = Path(output_dir or HERE / "static" / "outputs")
    out.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()  # Grad-CAM hooks mutate the shared model
    retention = float(os.getenv("RETENTION_HOURS", "24"))
    predict_limit, chat_limit = RateLimiter(20), RateLimiter(40)
    state = {"predictor": None, "error": None}
    try:
        state["predictor"] = Predictor(artifact_dir)
    except FileNotFoundError as e:
        app.logger.warning("models not loaded: %s", e)  # detail stays in the server log only
        state["error"] = "Models not loaded. Train them first (see README)."

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        p = state["predictor"]
        if p is None:
            return jsonify(ready=False, error=state["error"]), 503
        return jsonify(ready=True, classes=p.classes, subtype_models=sorted(p.subtype),
                       fusion=p.fusion is not None, image_model=p.image_meta)

    @app.post("/api/predict")
    def predict():
        p = state["predictor"]
        if p is None:
            return jsonify(error=state["error"]), 503
        if not predict_limit.allow(request.remote_addr or "?"):
            return jsonify(error="too many requests, slow down"), 429
        purge_old(out, retention)
        f = request.files.get("image")
        if f is None or not f.filename:
            return jsonify(error="image file is required"), 400
        ext = Path(f.filename).suffix.lower()
        if ext not in IMAGE_EXTS:
            return jsonify(error=f"unsupported image type '{ext}'"), 400
        rid = uuid.uuid4().hex
        img_path = out / f"{rid}_input{ext}"
        f.save(img_path)
        try:
            genomic = parse_genomic_csv(request.files["genomic_csv"]) \
                if request.files.get("genomic_csv") and request.files["genomic_csv"].filename else None
            clinical = {k: request.form.get(k) for k in ("age", "sex", "stage")}
            with lock:
                result = p.predict(img_path, clinical, genomic)
                cam = p.gradcam(img_path)
        except InputError as e:
            img_path.unlink(missing_ok=True)
            return jsonify(error=str(e)), 400
        except Exception:
            img_path.unlink(missing_ok=True)
            app.logger.exception("prediction failed")
            return jsonify(error="prediction failed (see server log)"), 500
        cam.save(out / f"{rid}_gradcam.png")
        result["id"] = rid
        result["gradcam_url"] = f"/outputs/{rid}_gradcam.png"
        result["input_ext"] = ext
        result["input_url"] = f"/outputs/{rid}_input{ext}"
        result["created"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (out / f"{rid}.json").write_text(json.dumps(result))
        return jsonify(result)

    @app.post("/api/chat")
    def chat():
        if not chat_limit.allow(request.remote_addr or "?"):
            return jsonify(error="too many requests, slow down"), 429
        body = request.get_json(silent=True) or {}
        rid, result = body.get("result_id"), None
        if isinstance(rid, str) and _ID.match(rid) and (out / f"{rid}.json").exists():
            result = json.loads((out / f"{rid}.json").read_text())
        history = body.get("history") if isinstance(body.get("history"), list) else []
        return jsonify(chatbot.reply(str(body.get("message", "")), result, history))

    @app.get("/outputs/<name>")
    def output_file(name):
        if not _OUT_FILE.match(name):
            abort(404)
        return send_from_directory(out, name)

    @app.get("/api/result/<rid>")
    def get_result(rid):
        if not _ID.match(rid) or not (out / f"{rid}.json").exists():
            abort(404)
        return jsonify(json.loads((out / f"{rid}.json").read_text()))

    @app.get("/api/model-card")
    def model_card():
        return jsonify(load_model_card(artifact_dir))

    @app.get("/report/<rid>")
    def report(rid):
        if not _ID.match(rid) or not (out / f"{rid}.json").exists():
            abort(404)
        result = json.loads((out / f"{rid}.json").read_text())
        pdf = build_report(result, out / f"{rid}.pdf", out / f"{rid}_input{result['input_ext']}",
                           out / f"{rid}_gradcam.png", report_id=rid[:8],
                           model_card=load_model_card(artifact_dir), confusion_png=confusion_png(artifact_dir))
        return send_file(pdf, mimetype="application/pdf", as_attachment=True,
                         download_name=f"gastricsentinel_{rid[:8]}.pdf")

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
