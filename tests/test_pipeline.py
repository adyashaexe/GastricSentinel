"""Run with:  pytest -q tests/        (the end-to-end test takes ~1-2 min on CPU)

Unit tests pin the behaviours that fix the original project's problems:
metric consistency, leakage-free splits, train-only preprocessing, explicit
handling of rare classes and missing modalities.
"""
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gastricsentinel.data_images import build_manifest, make_splits  # noqa: E402
from gastricsentinel.data_tabular import TabularPreprocessor, parse_stage  # noqa: E402
from gastricsentinel.metrics import evaluate  # noqa: E402


# ------------------------------------------------------------------ units
def test_weighted_recall_equals_accuracy_and_f1_is_not_harmonic_of_macros():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    p = np.where(rng.random(400) < 0.7, y, rng.integers(0, 4, 400))
    m = evaluate(y, p, list("abcd"), n_boot=0)
    assert m["weighted"]["recall"] == pytest.approx(m["accuracy"])
    P, R = m["macro"]["precision"], m["macro"]["recall"]
    assert m["macro"]["f1"] <= 2 * P * R / (P + R) + 1e-9  # mean of F1s <= harmonic mean of means
    assert np.array(m["confusion_matrix"]).sum() == 400


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"), "--out", str(d)], check=True)
    return d


def test_group_split_has_no_overlap(synth):
    df = build_manifest(synth / "images", group_regex=r"^(P\d+)_")
    out, info = make_splits(df)
    assert info["leakage_risk"] is False
    g = {s: set(out[out.split == s].group) for s in ("train", "val", "test")}
    assert not (g["train"] & g["test"]) and not (g["train"] & g["val"]) and not (g["val"] & g["test"])
    assert set(out["split"]) == {"train", "val", "test"}


def test_missing_groups_refuses_unless_explicit(synth):
    df = build_manifest(synth / "images")  # no regex -> no groups
    with pytest.raises(ValueError, match="leakage-free"):
        make_splits(df)
    out, info = make_splits(df, allow_patch_level=True)
    assert info["leakage_risk"] is True


def test_regex_that_misses_files_is_an_error(synth):
    with pytest.raises(ValueError, match="matched no group"):
        build_manifest(synth / "images", group_regex=r"^(ZZZ\d+)_")


def test_stage_parsing():
    assert parse_stage("Stage IIIB") == 3 and parse_stage("Stage IV") == 4
    assert parse_stage("Stage IA") == 1 and np.isnan(parse_stage("'--")) and parse_stage(2) == 2


def test_preprocessor_uses_train_statistics_only():
    rng = np.random.default_rng(1)
    n = 40
    X = pd.DataFrame({"age": rng.integers(40, 80, n), "sex": ["male", "female"] * 20, "stage": ["Stage II"] * n})
    for j in range(30):
        X[f"g__G{j}"] = rng.normal(0, 1 + (j == 0) * 5, n)  # G0 has by far the largest variance
    train, test = X.iloc[:30], X.iloc[30:].copy()
    test["g__G0"] = 1e6  # absurd test values must not change anything learned
    a = TabularPreprocessor(True, 5).fit(train)
    b = TabularPreprocessor(True, 5).fit(pd.concat([train, test]))
    assert a.genes_[0] == "G0"
    assert not np.allclose(a.g_mu_, b.g_mu_)  # fitting on everything WOULD have leaked test info
    row, cov = a.make_row(60, "female", "III", {"G0": 1.0})
    assert 0 < cov < 1
    clin, gen = a.transform_parts(row)
    assert clin.shape == (1, 6) and gen.shape == (1, 5)
    assert np.allclose(gen[0, 1:], 0)  # genes not supplied -> neutral z=0, not garbage


# ------------------------------------------------------------ end to end
def _run(*args):
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r


def test_end_to_end(synth, tmp_path):
    art = tmp_path / "artifacts"
    clin = synth / "clinical" / "clinical.tsv"
    expr = synth / "genomics" / "TCGA.STAD.sampleMap_HiSeqV2_PANCAN"
    sub = synth / "clinical" / "subtypes.tsv"

    _run("gastricsentinel.train_image", "--data-dir", str(synth / "images"), "--group-regex", r"^(P\d+)_",
         "--out-dir", str(art / "image"), "--epochs", "2", "--batch-size", "32", "--image-size", "64",
         "--no-pretrained", "--trainable", "all")
    for f in ("image_model.pt", "splits.csv", "leakage_audit.json", "test_metrics.json",
              "test_per_class.csv", "test_confusion_matrix.csv", "test_confusion_matrix.png"):
        assert (art / "image" / f).exists(), f
    assert json.loads((art / "image" / "split_info.json").read_text())["leakage_risk"] is False
    for f in ("ood_stats.npz", "test_reliability.png"):
        assert (art / "image" / f).exists(), f
    mt = json.loads((art / "image" / "test_metrics.json").read_text())
    assert len(mt["reliability"]) == 10 and mt["brier"] >= 0 and mt["context"]["arch"] == "resnet50"
    versions = list((art / "versions").glob("*/meta.json"))
    assert len(versions) == 1

    # a second backbone, then the statistical comparison between the two
    _run("gastricsentinel.train_image", "--data-dir", str(synth / "images"), "--group-regex", r"^(P\d+)_",
         "--out-dir", str(art / "image_b"), "--epochs", "2", "--batch-size", "32", "--image-size", "64",
         "--no-pretrained", "--trainable", "all", "--backbone", "mobilenet_v3_large", "--mixup", "0.2", "--tta",
         "--aug", "light")
    assert json.loads((art / "image_b" / "test_metrics.json").read_text())["context"]["arch"] == "mobilenet_v3_large"
    ids = sorted(v.parent.name for v in (art / "versions").glob("*/meta.json"))
    assert len(ids) == 2
    _run("gastricsentinel.compare_models", ids[0], ids[1], "--artifacts", str(art), "--n-boot", "200")
    cmp_ = json.loads(next((art / "comparisons").glob("*.json")).read_text())
    assert 0 <= cmp_["mcnemar"]["p_value"] <= 1 and cmp_["n_paired"] > 10 and len(cmp_["diff_b_minus_a"]["accuracy_ci95"]) == 2
    from gastricsentinel.registry import list_versions
    assert len(list_versions(art)) == 2

    _run("gastricsentinel.train_tabular", "--clinical-tsv", str(clin), "--expression", str(expr),
         "--subtypes", str(sub), "--out-dir", str(art / "tabular"), "--top-genes", "40", "--trees", "60")
    info = json.loads((art / "tabular" / "tabular_info.json").read_text())
    assert "STAD_RARE" in info["excluded_classes"]  # reported, not silently merged

    _run("gastricsentinel.train_fusion", "--image-dir", str(art / "image"), "--clinical-tsv", str(clin),
         "--expression", str(expr), "--subtypes", str(sub), "--out-dir", str(art / "fusion"),
         "--top-genes", "40", "--epochs", "5")
    ps = json.loads((art / "fusion" / "pairing_summary.json").read_text())
    assert ps["patient_disjoint_across_splits"] and not ps["patient_level_correspondence"]
    assert (art / "fusion" / "ablation_table.csv").exists()

    # ---- serving: fallback across modality combinations
    from gastricsentinel.inference import InputError, Predictor
    pred = Predictor(art)
    img = next((synth / "images" / "TUM").glob("*.png"))
    r0 = pred.predict(img)
    assert r0["mode"] == "image_only" and r0["molecular_subtype"] is None and r0["experimental_fusion"] is None
    assert abs(sum(r0["prediction"]["probabilities"].values()) - 1) < 1e-5

    r1 = pred.predict(img, {"age": 61, "sex": "male", "stage": "III"})
    assert r1["modalities_used"] == {"image": True, "clinical": True, "genomic": False}
    assert r1["molecular_subtype"]["model"] == "clinical_only"
    assert r1["experimental_fusion"]["modalities"] == {"image": True, "clinical": True, "genomic": False}
    assert any("non-matched" in w or "randomly paired" in w for w in r1["warnings"])

    genes = pred.subtype["genomic_clinical"]["pipeline"].named_steps["prep"].genes_
    r2 = pred.predict(img, {"age": 61}, {g: 8.0 for g in genes})
    assert r2["modalities_used"]["genomic"] is True and r2["molecular_subtype"]["model"] == "genomic_clinical"
    assert r2["molecular_subtype"]["top_features"]

    r3 = pred.predict(img, None, {g: 8.0 for g in genes[:3]})  # too little coverage
    assert r3["modalities_used"]["genomic"] is False and any("covers only" in w for w in r3["warnings"])

    for bad in ({"age": 400}, {"sex": "robot"}, {"stage": "V"}, {"age": "abc"}):
        with pytest.raises(InputError):
            pred.predict(img, bad)
    bad_img = tmp_path / "bad.png"
    bad_img.write_bytes(b"not an image")
    with pytest.raises(InputError):
        pred.predict(bad_img)

    cam = pred.gradcam(img)
    assert cam.size[0] > 0

    # ---- web app + PDF
    from fastapi.testclient import TestClient

    from app.app import create_app
    app = create_app(art, tmp_path / "out")
    with TestClient(app) as c:
        _web_checks(c, img, genes)


def _web_checks(c, img, genes):
    assert c.get("/api/status").json()["ready"] is True
    assert c.get("/").status_code == 200
    csv = "gene,value\n" + "\n".join(f"{g},8.0" for g in genes)
    resp = c.post("/api/predict", data={"age": "55", "sex": "female", "stage": "II"},
                  files={"image": ("x.png", img.read_bytes()), "genomic_csv": ("g.csv", csv.encode())})
    assert resp.status_code == 200, resp.text
    j = resp.json()
    assert j["molecular_subtype"]["model"] == "genomic_clinical"
    rep = c.get(f"/report/{j['id']}")
    assert rep.status_code == 200 and rep.content[:4] == b"%PDF"
    assert rep.content.count(b"/Type /Page\n") + rep.content.count(b"/Type /Page ") >= 4, "report should be multi-page"
    assert j["inputs"]["age"] == 55 and j["inputs"]["genomic_provided"] is True
    # new routes
    rid = j["id"]
    assert c.get(f"/api/result/{rid}").json()["id"] == rid
    assert c.get("/api/result/" + "0" * 32).status_code == 404
    mc = c.get("/api/model-card").json()
    assert mc["available"] and mc["image"]["n_test"] > 0 and mc["fusion"]["pairing_mode"]
    assert c.get(f"/outputs/{rid}_gradcam.png").status_code == 200
    assert c.get(f"/outputs/{rid}.json").status_code == 404
    assert c.get("/outputs/..%2Fsecret.png").status_code == 404
    assert c.get("/report/../../etc/passwd").status_code == 404
    assert c.get("/report/" + "0" * 32).status_code == 404
    assert "ood" in j and "flags" in j["ood"] and j["model_info"]["arch"] == "resnet50"
    mc2 = c.get("/api/model-card").json()
    assert mc2["image"]["confusion_matrix"] and mc2["image"]["reliability"] and mc2["history"] and mc2["ood"]["threshold"] > 0
    assert len(mc2["versions"]) == 2 and len(mc2["comparisons"]) == 1
    # batch: two good images and one bad file
    good = img.read_bytes()
    b = c.post("/api/batch", files=[("images", ("a.png", good)), ("images", ("b.png", good)), ("images", ("c.exe", b"x"))])
    assert b.status_code == 200, b.text
    bj = b.json()
    assert bj["summary"]["n"] == 3 and bj["summary"]["ok"] == 2 and bj["summary"]["failed"] == 1
    assert c.post("/api/batch", files=[("images", (f"{i}.png", good)) for i in range(26)]).status_code == 400
    # whole-slide mode on a larger image
    from PIL import Image as PI
    big = PI.new("RGB", (640, 640))
    tile_src = PI.open(img).convert("RGB").resize((320, 320))
    for x in (0, 320):
        for y in (0, 320):
            big.paste(tile_src, (x, y))
    buf = io.BytesIO()
    big.save(buf, "PNG")
    sl = c.post("/api/slide", data={"tile": "320"}, files={"image": ("s.png", buf.getvalue())})
    assert sl.status_code == 200, sl.text
    sj = sl.json()
    assert sj["mode"] == "slide" and sj["slide"]["n_tissue_tiles"] == 4 and len(sj["slide"]["tiles"]) == 4
    assert c.get(sj["gradcam_url"]).status_code == 200 and c.get(f"/report/{sj['id']}").status_code == 200
    tiny = io.BytesIO()
    PI.new("RGB", (100, 100)).save(tiny, "PNG")
    assert c.post("/api/slide", files={"image": ("t.png", tiny.getvalue())}).status_code == 400
    # export
    ex = c.get(f"/api/export?ids={rid},{sj['id']}")
    assert ex.status_code == 200 and ex.text.splitlines()[0].startswith("id,") and len(ex.text.splitlines()) == 3
    assert isinstance(c.get(f"/api/export?ids={rid}&format=json").json(), list)
    assert c.get("/api/export").status_code == 401 and c.get("/api/export?ids=" + "0" * 32).status_code == 404
    assert c.post("/api/predict", data={}).status_code == 400
    bad = c.post("/api/predict", files={"image": ("a.exe", b"x")})
    assert bad.status_code == 400
    bad = c.post("/api/predict", data={"age": "999"}, files={"image": ("x.png", img.read_bytes())})
    assert bad.status_code == 400


# ------------------------------------------------------------ chat / calibration
def test_ece_perfect_and_overconfident():
    from gastricsentinel.metrics import ece
    y = np.array([0, 1, 0, 1])
    assert ece(np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float), y) == pytest.approx(0.0)
    assert ece(np.array([[.99, .01]] * 4), y) > 0.4  # 50% right but 99% confident


def test_chat_offline_refuses_diagnosis_and_explains(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from gastricsentinel import chat
    res = {"prediction": {"class": "TUM", "confidence": 0.91, "low_confidence": False,
                          "probabilities": {"TUM": .91, "STR": .09}},
           "tumour_probability": 0.91, "modalities_used": {"image": True, "clinical": False, "genomic": False},
           "mode": "image_only", "warnings": []}
    assert "can't diagnose" in chat.reply("Do I have cancer?", res)["reply"]
    assert "Grad-CAM" in chat.reply("explain the heatmap", res)["reply"]
    assert "91.0%" in chat.reply("how confident is it", res)["reply"]
    assert chat.reply("hi", None)["mode"] == "offline"
    assert len(chat.reply("x" * 5000, None)["reply"]) > 0


def test_chat_endpoint_and_retention(synth, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import os, time
    from fastapi.testclient import TestClient

    from app.app import RateLimiter, create_app
    out = tmp_path / "out"
    out.mkdir()
    rl = RateLimiter(2)
    assert rl.allow("a") and rl.allow("a") and not rl.allow("a") and rl.allow("b")
    # chat works without trained models (assistant is independent of them)
    with TestClient(create_app(tmp_path / "no-artifacts", out)) as c:
        r = c.post("/api/chat", json={"message": "what are the limitations?", "result_id": "../../x"})
        assert r.status_code == 200 and "pathologist" in r.json()["reply"]
        assert c.post("/api/chat", content="not json", headers={"content-type": "text/plain"}).status_code == 200


def test_image_quality_flags_blank_and_non_he_images():
    from PIL import Image as PI

    from gastricsentinel.ood import image_quality
    blank = image_quality(PI.new("RGB", (224, 224), (250, 250, 250)))
    assert any("tissue" in f.lower() for f in blank["flags"])
    green = image_quality(PI.new("RGB", (224, 224), (30, 200, 40)))
    assert any("H&E" in f for f in green["flags"])
    tiny = image_quality(PI.new("RGB", (32, 32), (200, 100, 160)))
    assert any("small" in f for f in tiny["flags"])
    rng = np.random.default_rng(0)
    he = np.stack([rng.normal(220, 12, (224, 224)), rng.normal(120, 25, (224, 224)), rng.normal(180, 15, (224, 224))], 2).clip(0, 255).astype("uint8")
    assert image_quality(PI.fromarray(he))["flags"] == []
