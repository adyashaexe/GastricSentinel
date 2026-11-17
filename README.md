# GastricSentinel

**AI-assisted multimodal analysis of gastric histopathology: tissue classification, explainability, clinical/genomic context, and a documented report that states its own limits.**

> **Research prototype.** Not clinically validated, not a medical device, and not for diagnosis or treatment decisions. A pathologist must review every output.

GastricSentinel takes an H&E image patch (optionally with age, sex, stage and a gene-expression file) and returns:

- a **tissue class** with a calibrated confidence and an explicit low-confidence flag;
- a **Grad-CAM** map of where the image model looked;
- a **molecular-subtype estimate** from clinical and gene-expression inputs, with **SHAP** feature attributions;
- an **experimental fusion** result, clearly labelled as such;
- a multi-page **PDF report** with inputs, charts, validation figures, limitations and a reviewer sign-off block;
- a built-in **assistant** that explains the result in plain language (and refuses to diagnose).

This repository is a ground-up rebuild of an earlier version. The rebuild fixes the methodological problems that a peer review raised: undefined modality pairing, inconsistent metrics, possible train/test leakage, and unvalidated risk and explainability claims. See [What changed and why](#what-changed-and-why).

---

## Contents

1. [Quick start](#quick-start)
2. [Project layout](#project-layout)
3. [Data you need](#data-you-need)
4. [Training](#training)
5. [Running the web app](#running-the-web-app)
6. [Using the interface](#using-the-interface)
7. [The assistant](#the-assistant)
8. [Outputs and artifacts](#outputs-and-artifacts)
9. [How the methodology works](#how-the-methodology-works)
10. [Configuration](#configuration)
11. [Tests](#tests)
12. [Docker](#docker)
13. [Troubleshooting](#troubleshooting)
14. [What changed and why](#what-changed-and-why)
15. [Limitations](#limitations)
16. [Reporting results honestly](#reporting-results-honestly)

---

## Quick start

Requires **Python 3.10+**. On macOS and recent Linux you must use a virtual environment (system Python refuses `pip install`).

```bash
git clone <your-repo-url> GastricSentinel
cd GastricSentinel

python3 -m venv .venv
source .venv/bin/activate
which python                      # must end in .../GastricSentinel/.venv/bin/python
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use `python -m pip`, not bare `pip`, so you can never install into the wrong interpreter. Re-run `source .venv/bin/activate` in every new terminal.

**Try it without any real data** (synthetic images and tables, a few minutes on CPU):

```bash
python scripts/make_synthetic_data.py --out data_demo
python -m pytest -q tests          # end-to-end test, ~2 minutes
```

**Run it on real data:** see [Data you need](#data-you-need) and [Training](#training), then:

```bash
uvicorn app.app:app --reload       # http://127.0.0.1:8000
```

---

## Project layout

```
GastricSentinel/
├── app/                        FastAPI web app (async)
│   ├── app.py                  routes, rate limiting, startup model loading
│   ├── templates/index.html    single-page UI
│   └── static/
│       ├── css/app.css         themed, responsive styles
│       └── js/
│           ├── app.js          analysis flow, result rendering, evidence panel
│           ├── chat.js         floating assistant
│           ├── fx.js           theme toggle, reveal-on-scroll, effects
│           └── bg.js           animated histology background
├── gastricsentinel/            the Python package
│   ├── config.py               paths, constants, tissue names, disclaimer
│   ├── common.py               seeding, device selection, JSON helpers
│   ├── data_images.py          image dataset, group-aware splitting
│   ├── leakage_audit.py        near-duplicate (perceptual-hash) audit across splits
│   ├── data_tabular.py         clinical/genomic loading + leakage-safe preprocessor
│   ├── pairing.py              patient-disjoint image <-> tabular pairing + disclosure
│   ├── models.py               ResNet-50 classifier, FusionNet
│   ├── metrics.py              one-confusion-matrix metrics, bootstrap CIs, ECE
│   ├── train_image.py          image model training + temperature calibration
│   ├── train_tabular.py        subtype models (clinical-only, genomic+clinical)
│   ├── train_fusion.py         modality-dropout fusion + ablation table
│   ├── explain.py              Grad-CAM, SHAP
│   ├── inference.py            Predictor: validation, fallbacks, predictions
│   ├── model_card.py           reads validation artifacts for the UI and PDF
│   ├── report_pdf.py           multi-page PDF report
│   └── chat.py                 assistant (offline rules, optional LLM)
├── scripts/make_synthetic_data.py
├── tests/test_pipeline.py
├── docs/
│   ├── AUDIT.md                what was wrong in the original and how it was fixed
│   └── CAMERA_READY_CHECKLIST.md
├── Dockerfile
├── requirements.txt
└── README.md
```

Folders created when you run things (all git-ignored): `data/`, `artifacts/`, `app/static/outputs/`.

---

## Data you need

Nothing is downloaded automatically. Put your data under `data/` (any location works; you pass paths on the command line).

### Images (tissue classification)

One folder per class, containing image files:

```
data/images/HMU-GC-HE-30K/
├── ADI/  DEB/  LYM/  MUC/  MUS/  NOR/  STR/  TUM/
```

Class names are **discovered from the folder names** and stored inside the trained checkpoint, so training and serving can never disagree. Supported types: PNG, JPEG, BMP, TIFF.

Sources used during development:

| Dataset | Use | Notes |
|---|---|---|
| **HMU-GC-HE-30K** | main gastric H&E patch dataset | 8 tissue classes. Filenames such as `TUM_1.png` carry **no patient ID**. See the patient-level split note below. |
| **GasHisSDB** | possible external test set | gastric, normal vs abnormal only |
| **TCGA-STAD slides** (GDC) | the only route to truly patient-matched fusion | requires tiling slides into patches yourself |

### Clinical, genomic and subtype labels (TCGA-STAD)

| File | Source | Used for |
|---|---|---|
| `clinical.tsv` (GDC export) | GDC clinical download | age, sex, stage |
| `TCGA.STAD.sampleMap_HiSeqV2_PANCAN` | UCSC Xena TCGA-STAD hub | gene expression (genes × samples) |
| `stad_tcga_pan_can_atlas_2018_clinical_data.tsv` | cBioPortal PanCancer Atlas 2018 | molecular subtype labels (`Patient ID`, `Subtype`) |

Patients are joined on the 12-character TCGA barcode. Only primary-tumour samples (`-01`) are used.

### Patient-level splits: read this

A leakage-safe image split needs a **group ID** (patient or slide) for every image. The trainer **refuses to run** without one unless you pass `--allow-patch-level-split`. If your filenames contain an ID, pass `--group-regex` (its first group is the ID). If they do not, as in HMU-GC-HE-30K, you must use the patch-level flag, and the run is then stamped `leakage_risk` in its outputs, in the web UI and in the PDF.

---

## Training

Run from the project root with the virtual environment active. Three steps; each writes to its own folder under `artifacts/`.

### 1. Image model (six backbones)

```bash
python -m gastricsentinel.train_image \
  --data-dir data/images/HMU-GC-HE-30K \
  --allow-patch-level-split \
  --out-dir artifacts/image \
  --epochs 20 --batch-size 32 --image-size 224 \
  --trainable all --class-weights --num-workers 2
```

If your files carry a patient ID, drop the patch-level flag and add e.g. `--group-regex "^([A-Za-z0-9]+)_"`.

Useful flags:

| Flag | Meaning |
|---|---|
| `--backbone` | `resnet18`, `resnet50` (default), `densenet121`, `efficientnet_b0`, `convnext_tiny`, `mobilenet_v3_large` |
| `--trainable {fc,layer4,all}` | how much of the network is fine-tuned (default `layer4`) |
| `--aug {strong,light}` | stain/geometry augmentation strength |
| `--label-smoothing`, `--mixup`, `--grad-clip` | regularisation options |
| `--warmup-epochs`, `--head-lr-mult` | warm-up + cosine schedule; higher LR for the new head |
| `--tta` | flip test-time augmentation at evaluation |
| `--ood-samples N` | training embeddings used to fit the out-of-distribution detector |
| `--note "text"` | label stored with the saved model version |
| `--class-weights` | weight the loss for class imbalance |
| `--max-per-class N` | cap images per class, for a fast trial run |
| `--epochs`, `--patience` | training length and early-stopping patience |
| `--lr`, `--weight-decay` | optimiser settings |
| `--manifest file.csv` | explicit `path,label,group` table instead of folders |
| `--no-pretrained` | skip ImageNet weights (offline or smoke tests) |

Each run also produces: temperature scaling (kept only if it lowers validation calibration error), a reliability diagram and Brier score, an out-of-distribution detector (`ood_stats.npz`), and a registered model version under `artifacts/versions/<id>/`.

Version and compare commands:

```bash
python -m gastricsentinel.registry list
python -m gastricsentinel.registry activate <version>
python -m gastricsentinel.compare_models <versionA> <versionB>   # exact McNemar + paired bootstrap CIs
```

Models trained before this release still load. Retrain to get the out-of-distribution check, reliability diagram and versions.

A quick trial (about 10–15 minutes on an Apple-silicon Mac):

```bash
python -m gastricsentinel.train_image --data-dir data/images/HMU-GC-HE-30K \
  --allow-patch-level-split --epochs 5 --max-per-class 1000 --batch-size 32 \
  --trainable layer4 --class-weights --num-workers 2
```

Device is chosen automatically: CUDA, then Apple MPS, then CPU. On MPS errors, prefix the command with `PYTORCH_ENABLE_MPS_FALLBACK=1`. A full run on CPU or a laptop GPU takes hours; a free Colab or Kaggle GPU takes well under an hour.

### 2. Clinical and genomic subtype models

```bash
python -m gastricsentinel.train_tabular \
  --clinical-tsv data/clinical/extracted/clinical.tsv \
  --expression data/genomics/TCGA.STAD.sampleMap_HiSeqV2_PANCAN \
  --subtypes data/clinical/stad_tcga_pan_can_atlas_2018_clinical_data.tsv \
  --out-dir artifacts/tabular
```

Trains a **clinical-only** and a **genomic+clinical** random forest and reports 5-fold cross-validated metrics. Subtypes with fewer patients than folds are listed and excluded, never silently merged.

### 3. Fusion (experimental)

```bash
python -m gastricsentinel.train_fusion \
  --image-dir artifacts/image \
  --clinical-tsv data/clinical/extracted/clinical.tsv \
  --expression data/genomics/TCGA.STAD.sampleMap_HiSeqV2_PANCAN \
  --subtypes data/clinical/stad_tcga_pan_can_atlas_2018_clinical_data.tsv \
  --pairing random
```

`--pairing random` attaches each image to an **unrelated** TCGA patient. This is a *control experiment*: the clinical and genomic inputs are noise from the image's point of view, so **no gain over the image-only baseline is expected, and none should be claimed**. Use `--pairing matched` only when you have a manifest linking each patch to its real patient.

The step writes an ablation table comparing image-only, image+clinical, image+genomic and all three, each with a 95% confidence interval.

---

## Running the web app

```bash
uvicorn app.app:app --reload
```

Open **http://127.0.0.1:8000**. The app needs at least `artifacts/image/image_model.pt`. Without it, the page loads but shows "Models not loaded". Subtype and fusion features switch on automatically when their artifacts exist.

Drop `--reload` for anything beyond development. Interactive API docs are at **/docs**.

---

## Using the interface

A sidebar app with a light/dark toggle; on phones the sidebar becomes a drawer. A short guided tour runs on first visit and can be replayed ("Take the tour").

| Section | What it does |
|---|---|
| **Analyze image** | upload a patch (optionally age, sex, stage, gene file). Result page: zoom/pan viewer with Grad-CAM overlay, split and opacity controls, confidence, class probabilities, input check (out-of-distribution warning), subtype, expert review, PDF/CSV/JSON export |
| **Batch** | up to 25 images at once, with a summary table and export |
| **Whole slide** | a large image (up to 200 MB) is tiled, classified, and shown as a tile map with top-scoring tiles. Export proprietary formats (.svs, .ndpi) to PNG/TIFF first |
| **Compare** | two analyses side by side with linked zoom |
| **Library & reports** | every saved analysis, searchable, with reports and export |
| **Review queue** | reviewer accounts only |
| **Model performance** | overview, interactive confusion matrix, reliability diagram, training curves, versions and statistical comparison |
| **About & limits** | scope and limitations |
| **Assistant** | floating button, bottom-right |

Details worth knowing:

- Only the image is required. Anything left out is skipped and **named as skipped** in the results and the PDF.
- A gene-expression file is two columns, `gene,value`. It is used only if it covers at least half of the model's genes.
- Everything is saved in the database and kept; nothing is deleted automatically.
- Zoom: wheel or pinch, drag to pan, double-click to zoom, arrow keys when focused.

### PDF report

`/report/<id>` (the **Download the PDF report** button) produces a multi-page document:

1. Summary of findings with the plain-language reading and any warnings
2. Inputs supplied (and which were omitted)
3. Image analysis: probability chart, input image and Grad-CAM, how to read the map
4. Clinical and genomic analysis: subtype probabilities and SHAP chart
5. Experimental fusion status
6. Model validation: per-class table, confusion matrix, split and leakage audit, near-duplicate statistics, subtype-model and fusion-pairing disclosure
7. Limitations
8. Methods summary
9. A reviewer sign-off block

### HTTP endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | the UI |
| GET | `/api/status` | which models are loaded |
| POST | `/api/predict` | multipart: `image` (required), `age`, `sex`, `stage`, `genomic_csv` |
| GET | `/api/result/<id>` | a stored result (ID is validated; 404 if expired) |
| POST | `/api/batch` | multipart `images` (up to 25) |
| POST | `/api/slide` | multipart `image`, optional `tile` |
| GET | `/api/analyses` | saved analyses for the signed-in user |
| GET | `/api/export?ids=a,b&format=csv\|json` | export analyses |
| POST | `/api/analyses/<id>/review` | expert review (signed in) |
| GET | `/api/model-card` | validation facts read from the artifacts |
| POST | `/api/chat` | assistant; body `{message, result_id, history}` |
| GET | `/report/<id>` | PDF report |
| GET | `/outputs/<name>` | stored input / Grad-CAM image (filename pattern is validated) |

---

## The assistant

A small chat window in the bottom-right corner. It explains the loaded result (class, confidence, Grad-CAM, SHAP, modalities, report, limitations) in plain language. It **refuses to diagnose or advise on treatment**.

- **Offline mode (default):** a rule-based assistant; no network, no keys.
- **Language-model mode:** set an API key in the environment before starting the server:

```bash
export ANTHROPIC_API_KEY=...     # or OPENAI_API_KEY=...
export CHAT_MODEL=...            # optional override
uvicorn app.app:app --reload
```

If the model call fails for any reason, it falls back to offline mode automatically. The page shows which mode answered.

---

## Outputs and artifacts

Everything lands under `artifacts/`:

```
artifacts/
├── image/
│   ├── image_model.pt               model weights, class names, calibration temperature
│   ├── test_metrics.json            accuracy, balanced accuracy, macro P/R/F1, ECE, 95% CIs
│   ├── test_summary.md              the same, readable
│   ├── test_per_class.csv
│   ├── test_confusion_matrix.csv    and .png
│   ├── test_predictions.csv         every test prediction
│   ├── history.csv                  per-epoch curves
│   ├── splits.csv                   which image went to train / val / test
│   ├── split_info.json              split mode and the leakage_risk flag
│   └── leakage_audit.json           near-duplicate counts, test vs train
├── tabular/
│   ├── clinical_only.joblib         subtype models used by the app
│   ├── genomic_clinical.joblib
│   ├── *_metrics.json, *_per_class.csv, *_confusion_matrix.*   cross-validated results
│   └── tabular_info.json            classes, patient count, excluded rare classes
└── fusion/
    ├── fusion_model.pt, fusion_preprocessor.joblib
    ├── pairing_manifest.csv         which image was paired with which patient
    ├── pairing_summary.json         mode and the plain-language disclosure
    └── ablation_table.csv           image-only vs image+clinical vs image+genomic vs all
```

For a paper, the key files are `test_metrics.json`, `test_per_class.csv`, `test_confusion_matrix.png`, `leakage_audit.json` and `ablation_table.csv`.

---

## How the methodology works

**Splitting.** Images are split with `StratifiedGroupKFold` on patient/slide ID so no group appears in more than one split. Without IDs the split is patch-level and flagged.

**Leakage audit.** Perceptual hashes (dHash) find near-identical images across splits and report how many test images have a near-duplicate in training.

**Metrics.** Accuracy, balanced accuracy and macro precision/recall/F1 all come from **one** confusion matrix with the averaging stated, and an assertion checks that weighted recall equals accuracy. 95% intervals come from bootstrap resampling. The original repository reported inconsistent figures from different sources.

**Calibration.** Expected Calibration Error (ECE) is measured; a temperature is fitted on the validation split and kept **only if it improves validation ECE**.

**Tabular preprocessing.** Imputation values, z-score statistics and the top-variance gene list are learned **inside each cross-validation fold** on training patients only, so no information leaks from held-out patients. Missing-value indicators are included.

**Subtype models.** Random forests, evaluated with pooled out-of-fold predictions from stratified 5-fold CV. Classes with too few patients are excluded and reported.

**Fusion.** A small network over image features plus clinical and genomic embeddings, trained with **modality dropout** and a presence mask, so missing inputs are handled by design. Pairing is patient-disjoint and written to a manifest. The disclosure of the pairing mode is attached to every output.

**Explainability.** Grad-CAM from the last residual stage (forward hook plus gradient hook). SHAP TreeExplainer on the random forest. Both are presented as descriptions of model behaviour, not clinical evidence.

**Serving.** The `Predictor` validates inputs, falls back across modalities, flags low confidence (below 60%), and never presents experimental fusion as the primary prediction unless pairing is `matched`.

---

## Configuration

| Setting | Where | Default |
|---|---|---|
| Image size, seed, low-confidence threshold (0.60), minimum genomic coverage (0.50) | `gastricsentinel/config.py` | see file |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, `CHAT_MODEL` | environment | unset (offline assistant) |
| `GS_DB` | environment | `instance/gastricsentinel.db` (or `<outputs>/gastricsentinel.db` when an output dir is passed to `create_app`) |
| `GS_SECRET_KEY` | environment | generated once and saved as `secret.key` next to the database |
| `GS_REVIEWER_EMAILS`, `GS_ADMIN_EMAILS` | environment | empty (comma-separated emails) |
| `GS_REQUIRE_LOGIN` | environment | off (set `1` to require sign-in to analyse) |
| `GS_HTTPS` | environment | off (set `1` behind HTTPS so the session cookie is `Secure`) |
| Upload limit | `app/app.py` | 25 MB |
| Rate limits | `app/app.py` | 20 predictions / 40 chats per minute per IP |

Uploads are stored in `app/static/outputs/`, named by random IDs. Every analysis and chat message is stored in SQLite and kept until its owner (or an admin) deletes the analysis. See [Database and accounts](#database-and-accounts).

## Database and accounts

GastricSentinel uses **SQLite** through SQLModel (`app/database.py` for the engine and queries, `app/models.py` for the tables `users`, `prediction_results`, `chat_logs`, `analysis_reviews`). It is one local file in WAL mode, so reads and writes can overlap, and there is no server to install. A database from the earlier Flask version is migrated automatically on first start. Accounts are optional: without signing in, everything works and results are temporary.

| Table | Holds |
|---|---|
| `users` | email, name, salted PBKDF2-SHA256 password hash |
| `prediction_results` | prediction, confidence, tumour probability, mode, model version, full result JSON, owner (or none) |
| `chat_logs` | every assistant message, linked to the analysis and account when there are any |
| `analysis_reviews` | reviewer verdict (agree / disagree / unsure), suggested class, note; one per reviewer per analysis |

**What each part does**
- **Analysis history.** Signed in, every analysis is saved to your account, listed under "Your saved analyses", and can be reopened or deleted. Each stamps the model version (first 10 hex chars of the SHA-256 of `image_model.pt`).
- **Chat logs.** The assistant conversation is stored with the analysis and restored when you reopen it. Anonymous chats are stored too, without an account attached.
- **Reviewer sign-off.** Reviewers see a queue of saved analyses, open one, and record a verdict. Roles come only from the environment, so nobody can promote themselves:

```bash
export GS_REVIEWER_EMAILS="dr.one@hospital.org,dr.two@hospital.org"
export GS_ADMIN_EMAILS="you@kiit.ac.in"
export GS_SECRET_KEY="$(python -c 'import secrets;print(secrets.token_hex(32))')"
uvicorn app.app:app --reload
```

**Privacy rules.** An analysis owned by an account is visible only to its owner and to reviewers/admins; anyone else gets a 404. Anonymous analyses are reachable only through their unguessable 32-character ID and are stored indefinitely, as are anonymous chat messages (they are not tied to any account). Passwords are never stored in plain text, sign-in is rate-limited, cross-site POSTs are refused, and API responses are not cached.

**Endpoints:** `POST /api/auth/register|login|logout`, `GET /api/auth/me`, `GET /api/analyses`, `DELETE /api/analyses/<id>`, `GET /api/analyses/<id>/chat`, `POST /api/analyses/<id>/review`, `GET /api/review/queue?status=pending|reviewed|all`.

**Backup and reset.** Copy the `.db` file (and `secret.key`) to back up; delete them to start clean. Do not commit them; `instance/` and `*.db` are in `.gitignore`.

---

## Tests

```bash
python -m pytest -q tests
```

The suite builds synthetic data, trains tiny models, and exercises the whole stack: split leakage, preprocessing, metric consistency, pairing disclosure, the predictor, the FastAPI routes (predict, result, model card, output-file validation, chat) and PDF generation (asserting a multi-page report). It takes about two minutes on CPU and needs no real data.

---

## Docker

```bash
docker build -t gastricsentinel .
docker run -p 8000:8000 -v "$PWD/artifacts:/srv/artifacts" gastricsentinel
```

The image runs uvicorn with **one process**: the model is loaded once at startup and Grad-CAM runs under an asyncio lock, because its hooks mutate the shared model. Scale by running more containers.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'numpy'` / `pandas` | the virtual environment isn't active or packages weren't installed into it. Run `source .venv/bin/activate`, check `which python`, then `python -m pip install -r requirements.txt` |
| `error: externally-managed-environment` | you're using the system pip. Create and activate a `.venv`, and use `python -m pip` |
| App says "Models not loaded" | `artifacts/image/image_model.pt` is missing. Make sure training wrote to the same project folder you run the app from |
| Image trainer exits asking for group IDs | pass `--group-regex`, or `--allow-patch-level-split` and accept the `leakage_risk` flag |
| SSL error downloading ResNet-50 weights on macOS | run `/Applications/Python\ 3.*/Install\ Certificates.command` once, then retry |
| MPS error mid-training | prefix with `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| `ERR_UNSAFE_PORT` in a browser | use the default port 8000; some ports are blocked by browsers |
| Genomic file "ignored" | it covers under 50% of the model's genes; check gene symbols match the Xena HiSeqV2 naming |
| Training accuracy far above validation | overfitting; add stronger augmentation or regularisation, and don't train longer |

---

## What changed and why

The original repository could not be reproduced and its claims were not all supportable. The rebuild addresses each reviewer concern directly:

| Concern | Rebuild |
|---|---|
| Image and genomic data were paired without a defined rule | explicit `random` / `matched` pairing, written to a manifest, disclosed in outputs |
| Inconsistent accuracy and F1 figures | all metrics from one confusion matrix, averaging stated, consistency asserted |
| Possible train/test leakage | group-aware splits, patch-level mode refused unless acknowledged and flagged, near-duplicate audit |
| Gene selection and scaling over all patients before splitting | all statistics learned inside each training fold |
| Fusion was not trained for missing modalities | modality dropout and a presence mask |
| Risk score and explainability unvalidated | confidence is calibrated and measured; Grad-CAM and SHAP are labelled as model descriptions |
| Overclaiming | disclaimers everywhere, limits listed in the UI and the PDF |

The full list is in [`docs/AUDIT.md`](docs/AUDIT.md); the pre-submission list is in [`docs/CAMERA_READY_CHECKLIST.md`](docs/CAMERA_READY_CHECKLIST.md).

---

## Limitations

- Research prototype; not validated for clinical use.
- A result describes **one image patch**, not a slide and not a patient.
- Trained on public data. Behaviour on other scanners, stains or hospitals is unknown.
- Image datasets and TCGA-STAD are **different cohorts**. Unless slides are matched to TCGA barcodes, fusion is a control experiment and cannot show multimodal benefit.
- A patch-level split can inflate scores. Report such results as such.
- Rare molecular subtypes (for example POLE, EBV, MSI) have few patients and are estimated unreliably.
- TCGA subtype labels were originally defined using molecular data that includes expression, so predicting them from expression is partly circular.
- Grad-CAM and SHAP describe the model, not biology, and have not been validated by pathologists.

---

## Reporting results honestly

If you publish numbers from this system:

1. State the split mode. If it was patch-level, say so and quote the near-duplicate count from `leakage_audit.json`.
2. Report the confidence intervals, not just point estimates.
3. Describe fusion as a control unless you used `matched` pairing with real patient correspondence.
4. Do not claim that clinical or genomic data improved image classification from a `random`-pairing ablation.
5. Prefer an external test set (a different dataset, scanner or hospital) over any internal split.

---

## Acknowledgements and data licences

Datasets are the property of their creators and are used under their own licences: HMU-GC-HE-30K, GasHisSDB, and TCGA-STAD (via the GDC, UCSC Xena and cBioPortal). Check each licence before redistributing data or trained models.
