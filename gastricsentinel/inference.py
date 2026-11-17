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
from .ood import image_quality, load_stats, mahalanobis


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
        self.ood = load_stats(self.dir / "image" / "ood_stats.npz")

        self.fusion = self.fusion_prep = None
        fp = self.dir / "fusion" / "fusion_model.pt"
        if fp.exists():
            fck = torch.load(fp, map_location="cpu", weights_only=True)
            if fck["classes"] != self.classes:
                raise RuntimeError("fusion model classes differ from image model classes")
            self.fusion = FusionNet(len(self.classes), fck["clin_dim"], fck["gen_dim"], img_dim=fck.get("img_dim", 2048))
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
    def _rgb(self, image_path):
        try:
            with Image.open(image_path) as im:
                return im.convert("RGB")
        except Exception as e:
            raise InputError(f"could not read image: {e}")

    def _tensor(self, image_path):
        return self.transform(self._rgb(image_path)).unsqueeze(0)

    def ood_check(self, feats, quality: dict) -> dict:
        """Combine the cheap image-quality flags with the feature-space distance (when trained)."""
        out = {"quality": quality["metrics"], "flags": list(quality["flags"]), "feature_distance": None, "flagged": bool(quality["flags"])}
        if self.ood is not None:
            d = float(mahalanobis(self.ood, feats.cpu().numpy())[0])
            far = d > self.ood["threshold"]
            out["feature_distance"] = {"score": d, "threshold": self.ood["threshold"], "typical": self.ood["id_p50"],
                                       "ratio": d / self.ood["threshold"], "outside": bool(far)}
            if far:
                out["flags"].append("This image is further from the training images than 99% of held-out images, so the model is "
                                    "likely outside what it learned. Do not trust this prediction.")
                out["flagged"] = True
        return out

    def predict(self, image_path, clinical: dict | None = None, genomic: dict | None = None) -> dict:
        age, sex, stage, has_clin = validate_clinical(clinical)
        rgb = self._rgb(image_path)
        x = self.transform(rgb).unsqueeze(0).to(self.device)
        quality = image_quality(rgb)
        warnings = []
        with torch.no_grad():
            feats = self.backbone(x)
            probs = torch.softmax(self.model.head(feats) / self.temperature, 1)[0].cpu().numpy()
        top = int(probs.argmax())
        result = {
            "prediction": {
                "class": self.classes[top], "confidence": float(probs[top]),
                "low_confidence": bool(probs[top] < LOW_CONFIDENCE),
                "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
            },
            "tumour_probability": None, "experimental_fusion": None, "molecular_subtype": None, "histology_pattern": None,
            "mode": "image_only",
            "modalities_used": {"image": True, "clinical": False, "genomic": False},
            "model_info": {**self.image_meta, "arch": self.model.arch}, "disclaimer": DISCLAIMER,
        }
        tum = [i for i, c in enumerate(self.classes) if c in TUMOUR_CLASSES]
        if tum:
            result["tumour_probability"] = float(probs[tum].sum())
        if result["prediction"]["low_confidence"]:
            warnings.append(f"Low confidence (< {LOW_CONFIDENCE:.0%}); treat as inconclusive.")
        result["ood"] = self.ood_check(feats, quality)
        warnings.extend(result["ood"]["flags"])
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


    # ------------------------------------------------------------------ whole-slide mode
    def predict_slide(self, image_path, tile: int = 256, max_tiles: int = 1500, max_pixels: int = 90_000_000,
                      batch: int = 32, min_tissue: float = 0.2) -> tuple[dict, Image.Image, Image.Image]:
        """Tile a large image, classify every tissue tile, and aggregate.

        Returns (result, thumbnail, overlay). Works on any image PIL can open; proprietary slide formats
        (.svs, .ndpi) must be exported to PNG/TIFF at a working magnification first.
        """
        try:
            im = Image.open(image_path)
            w, h = im.size
            if w * h > max_pixels:
                raise InputError(f"image is {w * h / 1e6:.0f} megapixels; the limit is {max_pixels / 1e6:.0f}. Export it at lower magnification.")
            im = im.convert("RGB")
        except InputError:
            raise
        except Exception as e:
            raise InputError(f"could not read image: {e}")
        tile = int(min(max(tile, 64), 2048))
        while (w // tile) * (h // tile) > max_tiles:
            tile = int(tile * 1.25)
        cols, rows = max(1, w // tile), max(1, h // tile)
        if min(w, h) < tile:
            raise InputError(f"image ({w}x{h}) is smaller than one tile ({tile}px); use the single-image analysis instead")
        tiles, coords = [], []
        quality_by_tile = []
        for r in range(rows):
            for c in range(cols):
                crop = im.crop((c * tile, r * tile, (c + 1) * tile, (r + 1) * tile))
                q = image_quality(crop)["metrics"]
                if q["tissue_fraction"] >= min_tissue:
                    tiles.append(self.transform(crop))
                    coords.append((r, c))
                    quality_by_tile.append(q)
        if not tiles:
            raise InputError("no tissue was found in this image (all tiles are blank or very pale)")
        probs, feats_all = [], []
        with torch.no_grad():
            for i in range(0, len(tiles), batch):
                xb = torch.stack(tiles[i:i + batch]).to(self.device)
                f = self.backbone(xb)
                feats_all.append(f.cpu())
                probs.append(torch.softmax(self.model.head(f) / self.temperature, 1).cpu().numpy())
        P = np.concatenate(probs)
        F = torch.cat(feats_all).numpy()
        n = len(P)
        top = P.argmax(1)
        mean = P.mean(0)
        k = int(mean.argmax())
        tum = [i for i, c in enumerate(self.classes) if c in TUMOUR_CLASSES]
        tprob = P[:, tum].sum(1) if tum else np.zeros(n)
        counts = {c: int((top == i).sum()) for i, c in enumerate(self.classes)}
        order = np.argsort(-tprob)[:8] if tum else np.argsort(-P.max(1))[:8]
        ood_far = 0
        if self.ood is not None:
            d = mahalanobis(self.ood, F)
            ood_far = int((d > self.ood["threshold"]).sum())
        warnings = ["Slide mode summarises many independent patch predictions. It is not a slide-level diagnosis."]
        if ood_far > 0.2 * n:
            warnings.append(f"{ood_far} of {n} tissue tiles ({ood_far / n:.0%}) look unlike the training images; treat this slide result with caution.")
        if n < 10:
            warnings.append("Very few tissue tiles were found; the summary is not meaningful.")
        result = {
            "prediction": {"class": self.classes[k], "confidence": float(mean[k]), "low_confidence": bool(mean[k] < LOW_CONFIDENCE),
                           "probabilities": {c: float(p) for c, p in zip(self.classes, mean)}},
            "tumour_probability": float(tprob.mean()) if tum else None,
            "experimental_fusion": None, "molecular_subtype": None, "mode": "slide",
            "modalities_used": {"image": True, "clinical": False, "genomic": False},
            "model_info": {**self.image_meta, "arch": self.model.arch}, "disclaimer": DISCLAIMER, "warnings": warnings,
            "inputs": {"age": None, "sex": None, "stage": None, "genomic_provided": False, "genomic_coverage": None},
            "ood": {"flagged": ood_far > 0.2 * n, "flags": [], "feature_distance": None, "quality": None},
            "slide": {
                "width": w, "height": h, "tile": tile, "rows": rows, "cols": cols, "n_tissue_tiles": n, "n_tiles": rows * cols,
                "class_counts": counts, "classes": self.classes,
                "tumour_fraction": float((tprob > 0.5).mean()) if tum else None,
                "ood_tiles": ood_far,
                # one row per tissue tile: [row, col, predicted class index, tumour probability, confidence]
                "tiles": [[int(r), int(c), int(top[i]), round(float(tprob[i]), 4), round(float(P[i].max()), 4)] for i, (r, c) in enumerate(coords)],
                "top_tiles": [{"row": coords[i][0], "col": coords[i][1], "class": self.classes[int(top[i])], "tumour_probability": float(tprob[i])} for i in order],
            },
        }
        scale = 1600 / max(w, h) if max(w, h) > 1600 else 1.0
        thumb = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        overlay = thumb.copy().convert("RGBA")
        layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        from PIL import ImageDraw
        dr = ImageDraw.Draw(layer)
        ts = tile * scale
        for (r, c), tp in zip(coords, tprob):
            col = (210, 50, 70, int(40 + 150 * tp)) if tum else (20, 90, 180, 90)
            dr.rectangle([c * ts, r * ts, (c + 1) * ts - 1, (r + 1) * ts - 1], fill=col)
        overlay = Image.alpha_composite(overlay, layer).convert("RGB")
        return result, thumb, overlay
