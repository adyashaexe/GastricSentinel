# Audit of the original repository

Source: `adyashaexe/GastricSentinel`, commit `8ed0c1b`. Findings are from reading the code; nothing was re-run on your real data, so none of the paper's numbers could be traced to a specific script and run.

## A. The three layers describe three different systems

| Layer | What it does |
|---|---|
| `training/train.py` | 2-class (Abnormal/Normal) ResNet-50 on GasHisSDB, saves `model/gastric_resnet50.pkl` |
| `backend/utils.py` | 9 classes `ADI BACK DEB LYM MUC MUS NORM STR TUM` (matches the colorectal NCT-CRC-HE label set, not a gastric one), expects `training/models/gastric_resnet50.pth` |
| Paper | 8-class HMU-GC-HE-30K |

`backend/model_loader.py` builds `Linear(2048, 9)` and loads a `.pth` that the training script never writes (it writes a 2-output `.pkl`). The app cannot load the trained model as shipped.

## B. Fusion model was never trained

- No script trains `FusionModel` (`backend/model_loader.py`); `gastric_fusion.pth` is never produced.
- `load_fusion_model()` silently returns **random weights** when the file is missing.
- `backend/predict.py` then trusts the fusion output whenever its top probability is >= 0.30 (random 9-class softmax can exceed this) and otherwise falls back to the image model. That is not a principled missing-modality mechanism.
- `training/train_fusion_model.py` trains nothing. It evaluates the image model and computes (but never evaluates) genomic predictions, then writes a pickle containing file paths.

## C. Train/serve skew

- `GenomicMLP` expects 20 inputs. The web form supplies 2 scalars (`gene_score`, `genomic_risk`), pads with 18 zeros and z-scores them per sample. Training-side genomic features are 2000 z-scored genes. Unrelated.
- Clinical: the app passes raw age (e.g. 60) and stage 1-4; `build_clinical_features.py` produces z-scores.

## D. Leakage and evaluation

- `train.py`: random stratified split over individual patches, no slide/patient grouping. GasHisSDB patches are cropped from a limited set of source images, so overlapping or near-identical patches across splits are likely.
- `train_fusion_model.py`: the reported "93.2% image accuracy" is computed on the first 1000 abnormal + first 1000 normal files in sorted order, taken from all data with no split applied. Many of those images were training images. It is not a held-out number.
- `build_genomic_features.py`: top-variance gene selection and z-scoring use all patients before any split.
- `train_genomic_clinical.py`: one 80/20 split; with ~7 POLE patients the test set holds about one. Rare classes are silently merged into "Other".
- Missing clinical records are replaced by zeros without a flag (`train_fusion_model.py`).

## E. Backend claims that were not true

- `/stats` returns a hard-coded `model_accuracy: 0.942`.
- `/shap_explain` returns `(p - 1/8) * 1.2` per class. That is not SHAP, and the `1/8` assumes 8 classes while the model has 9.
- `shap_explainer.py` runs DeepExplainer on the image with a zero background. The README says SHAP explains genomic/clinical features.
- Risk score = P(TUM) + P(STR) with cut-offs 0.20/0.45 that were never calibrated or validated (stroma is not tumour).

## F. Hygiene

- `requirements.txt` omits `pandas` and `joblib`, which the training scripts import.
- Supabase project URL is hard-coded in `backend/database.py` (the key is read from the environment, which is correct).
- Flask `debug=True` in `app.py`.

## G. The reported metrics (paper, Review 2)

Accuracy 61.2%, precision 90.3%, recall 79.5%, F1 84.5%.

- Weighted recall equals accuracy by definition, and macro recall equals it on balanced classes. Accuracy 61.2% with recall 79.5% therefore cannot come from one set of predictions with the usual averaging.
- 2 x 0.903 x 0.795 / (0.903 + 0.795) = 0.8456, so the F1 looks like the harmonic mean of the averaged precision and recall, not the mean of per-class F1.
- Most likely the figures come from different runs, subsets or averaging modes. Which, can only be settled by you from the original run logs.

`gastricsentinel/metrics.py` derives everything from one confusion matrix and asserts the accuracy/weighted-recall identity.
