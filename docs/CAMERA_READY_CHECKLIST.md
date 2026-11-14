# Camera-ready checklist (SCOReD 2026, paper #1571360271)

Review 2 was Weak Reject (5); Review 1 was Accept (9). Address Review 2 point by point. Numbers below must come from re-running this project on your real data. Nothing here supplies numbers for you.

| Reviewer point | What this project produces | What to change in the paper |
|---|---|---|
| How were HMU-GC-HE-30K images + clinical + TCGA-STAD genomics paired? | `pairing_manifest.csv`, `pairing_summary.json` with a `disclosure` string | State the real pairing rule in Methods. Different cohorts without shared IDs means no patient-level correspondence; say so. Present fusion as system integration, not as evidence of multimodal gain, unless you have `matched` data. |
| Accuracy 61.2% vs recall 79.5% vs F1 84.5% | `test_metrics.json`, `test_summary.md` with averaging stated; `test_per_class.csv`; confusion matrix CSV/PNG | Replace the metric table with one derived from a single confusion matrix. State macro vs weighted. Add the confusion matrix and per-class table. |
| Patient/slide-level separation, leakage | `split_info.json`, `splits.csv`, `leakage_audit.json` | State the split unit. If the dataset has no patient/slide IDs, say a patient-level split was not possible and report the near-duplicate audit as a limitation. |
| Limited novelty | - | Frame the contribution as an integrated, reproducible pipeline: classification, explainability, reporting, trained missing-modality handling. Keep Review 1's framing. |
| Risk estimation and explainability not validated | Output is called "tumour-class probability"; Grad-CAM/SHAP described as model attention | Do not call the score clinical risk. Add a limitations paragraph: no pathologist validation of heat-maps, no calibration study. |
| (Review 1) image-only / clinical-only models | `ablation_table.csv` (image_only, +clinical, +genomic, all) and `clinical_only_*` subtype metrics | Report the ablation. With random pairing the expected result is no gain; say that. |

## Before you re-run

1. Find out which code and data produced the paper's numbers (logs, notebooks). The old repo's `training/` scripts cannot have produced an 8-class result.
2. Check whether HMU-GC-HE-30K ships patient/slide IDs. That decides whether the split can be leakage-free.
3. Check the camera-ready deadline, IEEE PDF eXpress step, copyright form and registration deadline in the acceptance email.
