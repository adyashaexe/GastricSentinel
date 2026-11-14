# GastricSentinel (rebuilt)

Multimodal gastric cancer analysis **research prototype**: ResNet-50 tissue classifier, TCGA-STAD clinical/genomic models, a modality-aware fusion head, Grad-CAM + SHAP explanations, a small Flask app and PDF reports.

This rebuild replaces a codebase whose pieces disagreed with each other (see `docs/AUDIT.md`). Every number it prints comes from one code path, with the split, pairing and averaging choices written to disk.

> Not clinically validated. Not for diagnosis or treatment decisions.

## What is in here

```
gastricsentinel/
  data_images.py     manifest, GROUP-aware split (refuses to split without patient/slide IDs)
  leakage_audit.py   group-overlap + perceptual-hash near-duplicate check across splits
  data_tabular.py    TCGA-STAD loaders; preprocessor fit on training patients only
  pairing.py         image <-> TCGA record pairing (random | matched), written to pairing_manifest.csv
  models.py          ResNet-50, tabular encoders, FusionNet with modality dropout
  metrics.py         one confusion matrix -> accuracy, macro/weighted P/R/F1, per-class table, CI
  train_image.py     image classifier
  train_tabular.py   patient-level subtype models (clinical-only, genomic+clinical), 5-fold CV
  train_fusion.py    fusion head + modality ablation table
  explain.py         Grad-CAM (images), SHAP TreeExplainer (clinical/genomic features)
  inference.py       Predictor with input validation and modality fallback
  report_pdf.py      PDF report
app/                 Flask UI + API (python -m app.app)
scripts/make_synthetic_data.py   fake data in the real file layouts, for smoke tests only
tests/test_pipeline.py           unit + end-to-end tests
docs/AUDIT.md                    what was inconsistent in the original repo
docs/CAMERA_READY_CHECKLIST.md   reviewer comment -> what to change in the paper
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q tests/        # ~1 min, uses synthetic data, no downloads
```

## Data layout

```
data/images/<CLASS_NAME>/*.png                      # image dataset, one folder per class
data/clinical/clinical.tsv                          # GDC clinical export (TCGA-STAD)
data/genomics/TCGA.STAD.sampleMap_HiSeqV2_PANCAN    # UCSC Xena expression (genes x samples)
data/clinical/stad_tcga_pan_can_atlas_2018_clinical_data.tsv   # cBioPortal (Patient ID, Subtype)
```

Class names come from the folder names and are stored inside the checkpoint; nothing is hard-coded.

## Run

```bash
# 1. Image classifier. Needs patient/slide IDs for a leakage-free split:
python -m gastricsentinel.train_image --data-dir data/images \
    --group-regex "^(PATIENT_ID_PATTERN)_" --out-dir artifacts/image
#    or: --manifest my_manifest.csv   (columns: path,label,group)

# 2. Patient-level subtype models (cross-validated)
python -m gastricsentinel.train_tabular \
    --clinical-tsv data/clinical/clinical.tsv \
    --expression data/genomics/TCGA.STAD.sampleMap_HiSeqV2_PANCAN \
    --subtypes data/clinical/stad_tcga_pan_can_atlas_2018_clinical_data.tsv

# 3. Fusion + ablations
python -m gastricsentinel.train_fusion --image-dir artifacts/image \
    --clinical-tsv data/clinical/clinical.tsv \
    --expression data/genomics/TCGA.STAD.sampleMap_HiSeqV2_PANCAN \
    --subtypes data/clinical/stad_tcga_pan_can_atlas_2018_clinical_data.tsv \
    --pairing random            # or: matched (see below)

# 4. App
python -m app.app               # http://127.0.0.1:5000
```

Outputs land in `artifacts/`: `*_metrics.json`, `*_per_class.csv`, `*_confusion_matrix.{csv,png}`, `*_summary.md`, `splits.csv`, `leakage_audit.json`, `pairing_manifest.csv`, `pairing_summary.json`, `ablation_table.csv`.

## Things you must decide / state honestly

**Patient-level splits.** If your image files carry no patient or slide ID (names like `TUM_8.png`), a leakage-free split cannot be made from the files alone. Look for a metadata file with slide/patient IDs. If none exists, use `--allow-patch-level-split`; every report is then flagged `leakage_risk: true` and the paper must say so. `leakage_audit.json` still measures how many test images have a near-duplicate in train.

**Pairing.** Image datasets (e.g. HMU-GC-HE-30K) and TCGA-STAD are different cohorts with no shared IDs.
- `--pairing matched`: images are tiles from TCGA-STAD slides, `group` = TCGA barcode. Real correspondence.
- `--pairing random`: each image gets a random TCGA patient from the same split, independent of its label (patient-disjoint across splits). This is a **control**: fusion should not beat image-only, and any claim of multimodal benefit needs `matched` data.

The class-conditional pairing (e.g. tumour images paired with tumour-patient records) is deliberately not implemented: it injects label information into the tabular input and inflates accuracy.

**Which baseline.** For the effect of the tabular inputs, compare `fusion image_only` with `fusion image_clinical_genomic` in `ablation_table.csv`. The row `image_model_head` is the separately trained softmax head, kept for reference only.

## Serving behaviour

- Image is required; age/sex/stage and a gene-expression CSV (`gene,value`) are optional.
- Genomic input counts as present only if it covers at least 50% of the model's genes; otherwise it is ignored and a warning is returned.
- The response lists exactly which modalities were used.
- The primary prediction is image-based. Fusion output is shown as "experimental" unless the fusion model was trained with `--pairing matched`.
- "Tumour-class probability" is the model's softmax mass on the `TUM` class. It is not a calibrated clinical risk.
- Uploaded images and results are stored under `app/static/outputs/`; clear that folder if handling anything sensitive. The app binds to 127.0.0.1 and is not hardened for public deployment.
- `.pt`/`.joblib` files are pickles: only load artifacts you trained yourself.

## Front end and assistant

The UI is a single page with no external assets (works offline): animated aurora + interactive particle background, 3D-tilt glass cards, scan-laser while analysing, animated confidence gauge and probability bars, a drag-to-compare Grad-CAM slider, animated SHAP bars and a chat widget. All animation is switched off under the OS "reduce motion" setting.

The assistant explains the current result, Grad-CAM, SHAP, confidence and limitations, and refuses diagnosis/treatment questions.
- Default: built-in offline assistant, no key needed.
- Optional LLM: set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) and optionally `CHAT_MODEL` in the environment. Keys are never written to code or disk. Any provider error silently falls back to the offline assistant. Note that with an LLM key set, the result summary and your question are sent to that provider.
- `/api/predict` and `/api/chat` are rate-limited per IP; stored uploads are deleted after `RETENTION_HOURS` (default 24).

## Calibration

`train_image` fits a temperature on the validation split and reports ECE before/after on the test split (`test_summary.md`). The app's confidence uses the calibrated probabilities.

## Docker

```bash
docker build -t gastricsentinel . && docker run -p 5000:5000 -v $PWD/artifacts:/srv/artifacts gastricsentinel
```

## Not carried over from the old repo

Supabase patient database and patients/dashboard pages (coupled to a hard-coded schema and hard-coded model accuracy). The chatbot was kept and rebuilt (see above). A database can be re-added on top of `Predictor` and `/api/predict`.


## Web interface

Run `python -m app.app` and open http://127.0.0.1:5000.

- Opening sequence (curtain reveal, once per browser session; "Skip" available).
- Eight tabs: Overview, Analyze, Results, Explainability, Clinical & Genomic, Assistant, History, Methods. Results-dependent tabs unlock after an analysis.
- Light and dark theme toggle (top right). Follows the system setting until you choose; the choice is remembered.
- Animations respect `prefers-reduced-motion`.
- Overview and Methods read the real validation record from `artifacts/` (`/api/model-card`); nothing on those tabs is hard-coded.
- The PDF report (`/report/<id>`) is multi-page: findings, inputs, probability chart, Grad-CAM, SHAP, fusion status, per-class validation, confusion matrix, leakage audit, limitations, methods and a reviewer sign-off block.

## Training on your own data layout

```bash
python -m gastricsentinel.train_image --data-dir data/images/HMU-GC-HE-30K --group-regex "<patient-id regex>" --out-dir artifacts/image
python -m gastricsentinel.train_tabular --clinical-tsv data/clinical/extracted/clinical.tsv \
  --expression data/genomics/TCGA.STAD.sampleMap_HiSeqV2_PANCAN \
  --subtypes data/clinical/stad_tcga_pan_can_atlas_2018_clinical_data.tsv --out-dir artifacts/tabular
python -m gastricsentinel.train_fusion --image-dir artifacts/image --clinical-tsv ... --expression ... --pairing random
```
If image filenames carry no patient ID, pass `--allow-patch-level-split`; the report is then flagged `leakage_risk`.
