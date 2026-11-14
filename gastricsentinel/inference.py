"""Serving-time predictor. Uses exactly the artifacts (classes, preprocessing, models)
produced by training, so train/serve skew is not possible by construction.

Modalities are optional. Which ones were actually used is reported in every response.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image

from .config import ARTIFACTS, DISCLAIMER, LOW_CONFIDENCE, MIN_GENOMIC_COVERAGE, TUMOUR_CLASSES
from .data_images import get_transform
from .data_tabular import parse_sex, parse_stage
from .explain import GradCAM, overlay_cam, tabular_shap
from .models import Backbone, FusionNet, load_image_model


class InputError(ValueError):
    pass


def validate_clinical(clinical: dict | None):
    """Return (age, sex, stage, present). Raises InputError on out-of-range values."""
    c = clinical or {}
    age, sex, stage = c.get("age"), c.get("sex"), c.get("stage")
    if age in ("", None):
        age = None
    else:
        try:
            age = float(age)
        except (TypeError, ValueError):
            raise InputError("age must be a number")
        if not 0 <= age <= 120:
            raise InputError("age must be between 0 and 120")
    if sex in ("", None):
        sex = None
    else:
        sex = parse_sex(sex)
        if sex is None:
            raise InputError("sex must be 'male' or 'female'")
    if stage in ("", None):
        stage = None
    else:
        p = parse_stage(stage if str(stage).lower().startswith("stage") or str(stage).isdigit()
                        else f"Stage {stage}")
        if np.isnan(p):
            raise InputError("stage must be I, II, III or IV")
        stage = int(p)
    return age, sex, stage, any(v is not None for v in (age, sex, stage))


class Predictor:
    def __init__(self, artifact_dir=ARTIFACTS, device="cpu"):
        self.dir, self.device = Path(artifact_dir), torch.device(device)
        img_ckpt = self.dir / "image" / "image_model.pt"
        if not img_ckpt.exists():
            raise FileNotFoundError(f"{img_ckpt} not found - train the image model first "
                                    "(python -m gastricsentinel.train_image ...)")
        self.model, ck = load_image_model(img_ckpt, self.device)
        self.classes, self.image_size = ck["classes"], ck["image_size"]
        self.temperature = float(ck.get("temperature", 1.0))
        self.image_meta = {k: ck.get(k) for k in ("split_mode", "leakage_risk", "best_val_macro_f1", "temperature")}
        self.transform = get_transform(self.image_size, False)
        self.backbone = Backbone(self.model).to(self.device).eval()

        self.fusion = self.fusion_prep = None
        fp = self.dir / "fusion" / "fusion_model.pt"
        if fp.exists():
            fck = torch.load(fp, map_location="cpu", weights_only=True)
            if fck["classes"] != self.classes:
                raise RuntimeError("fusion model classes differ from image model classes")
            self.fusion = FusionNet(len(self.classes), fck["clin_dim"], fck["gen_dim"])
            self.fusion.load_state_dict(fck["state_dict"])
            self.fusion.eval()
            self.fusion_meta = fck
            self.fusion_prep = joblib.load(self.dir / "fusion" / "fusion_preprocessor.joblib")

        self.subtype = {}
        for name in ("genomic_clinical", "clinical_only"):
            p = self.dir / "tabular" / f"{name}.joblib"
            if p.exists():
                self.subtype[name] = joblib.load(p)

    # ------------------------------------------------------------------
    def _tensor(self, image_path):
        try:
            with Image.open(image_path) as im:
                return self.transform(im.convert("RGB")).unsqueeze(0)
        except Exception as e:
            raise InputError(f"could not read image: {e}")

    def predict(self, image_path, clinical: dict | None = None, genomic: dict | None = None) -> dict:
        age, sex, stage, has_clin = validate_clinical(clinical)
        x = self._tensor(image_path).to(self.device)
        warnings = []
        with torch.no_grad():
            feats = self.backbone(x)
            probs = torch.softmax(self.model.fc(feats) / self.temperature, 1)[0].cpu().numpy()
        top = int(probs.argmax())
        result = {
            "prediction": {
                "class": self.classes[top], "confidence": float(probs[top]),
                "low_confidence": bool(probs[top] < LOW_CONFIDENCE),
                "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
            },
            "tumour_probability": None, "experimental_fusion": None, "molecular_subtype": None,
            "mode": "image_only",
            "modalities_used": {"image": True, "clinical": False, "genomic": False},
            "model_info": self.image_meta, "disclaimer": DISCLAIMER,
        }
        tum = [i for i, c in enumerate(self.classes) if c in TUMOUR_CLASSES]
        if tum:
            result["tumour_probability"] = float(probs[tum].sum())
        if result["prediction"]["low_confidence"]:
            warnings.append(f"Low confidence (< {LOW_CONFIDENCE:.0%}); treat as inconclusive.")
        if self.image_meta.get("leakage_risk"):
            warnings.append("Image model was trained with a patch-level split; reported accuracy may be inflated.")

        # --- molecular subtype (patient-level, uses clinical +/- genomic only) ---
        cov = 0.0
        if self.subtype and (has_clin or genomic):
            name = "genomic_clinical" if "genomic_clinical" in self.subtype else "clinical_only"
            bundle = self.subtype[name]
            prep = bundle["pipeline"].named_steps["prep"]
            row, cov = prep.make_row(age, sex, stage, genomic)
            if name == "genomic_clinical" and cov < MIN_GENOMIC_COVERAGE:
                if genomic:
                    warnings.append(f"Genomic file covers only {cov:.0%} of the {len(prep.genes_)} model genes "
                                    f"(< {MIN_GENOMIC_COVERAGE:.0%}); genomic modality ignored.")
                name = "clinical_only" if "clinical_only" in self.subtype else None
            if name and (has_clin or cov >= MIN_GENOMIC_COVERAGE):
                bundle = self.subtype[name]
                pipe = bundle["pipeline"]
                row, _ = pipe.named_steps["prep"].make_row(age, sex, stage, genomic)
                pr = pipe.predict_proba(row)[0]
                shap_rows, ci = tabular_shap(pipe, row)
                result["molecular_subtype"] = {
                    "model": name, "class": bundle["classes"][int(pr.argmax())],
                    "probabilities": {c: float(p) for c, p in zip(bundle["classes"], pr)},
                    "top_features": shap_rows,
                    "note": "Cross-validated on TCGA-STAD; rare subtypes are poorly estimated.",
                }
                result["modalities_used"]["clinical"] = bool(has_clin)
                result["modalities_used"]["genomic"] = name == "genomic_clinical"

        # --- experimental image+tabular fusion ---
        if self.fusion is not None and (has_clin or genomic):
            row, cov_f = self.fusion_prep.make_row(age, sex, stage, genomic)
            gen_ok = cov_f >= MIN_GENOMIC_COVERAGE
            clin_t, gen_t = self.fusion_prep.transform_parts(row)
            mask = torch.tensor([[float(has_clin), float(gen_ok)]])
            with torch.no_grad():
                fp = torch.softmax(self.fusion(feats.cpu(), torch.from_numpy(clin_t),
                                               torch.from_numpy(gen_t), mask), 1)[0].numpy()
            matched = bool(self.fusion_meta.get("patient_level_correspondence"))
            result["experimental_fusion"] = {
                "class": self.classes[int(fp.argmax())], "confidence": float(fp.max()),
                "probabilities": {c: float(p) for c, p in zip(self.classes, fp)},
                "modalities": {"image": True, "clinical": has_clin, "genomic": gen_ok},
                "pairing_mode": self.fusion_meta.get("pairing_mode"),
                "patient_level_correspondence": matched,
            }
            if matched:
                result["prediction"].update(class_fusion=self.classes[int(fp.argmax())])
                result["mode"] = "fusion"
            else:
                warnings.append("Fusion model was trained on randomly paired (non-matched) data; "
                                "shown for reference only, not used for the primary prediction.")
        result["inputs"] = {"age": age, "sex": sex, "stage": stage,
                            "genomic_provided": bool(genomic),
                            "genomic_coverage": float(cov) if genomic else None}
        result["warnings"] = warnings
        return result

    def gradcam(self, image_path, class_idx: int | None = None) -> Image.Image:
        x = self._tensor(image_path).to(self.device)
        cam_engine = GradCAM(self.model)
        try:
            cam, _ = cam_engine(x, class_idx)
        finally:
            cam_engine.close()
        with Image.open(image_path) as im:
            return overlay_cam(im.convert("RGB"), cam)
