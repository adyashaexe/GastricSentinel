"""Explainability: Grad-CAM for images, SHAP (TreeExplainer) for clinical/genomic features."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Grad-CAM on the last convolutional stage (model.cam_layer, or ResNet layer4)."""

    def __init__(self, model, layer=None):
        self.model = model
        self.layer = layer if layer is not None else getattr(model, "cam_layer", None) or model.layer4
        self._acts = self._grads = None
        self._h = self.layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self._acts = out.detach()
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "_grads", g.detach()))

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """x: (1,3,H,W) normalised. Returns (cam in [0,1] as HxW numpy, class_idx)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)  # guarantees a graph even with frozen layers
            logits = self.model(x)
            idx = int(logits.argmax(1)) if class_idx is None else int(class_idx)
            logits[0, idx].backward()
        w = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self._acts).sum(1))[0]
        cam = (cam / (cam.max() + 1e-8)).cpu().numpy()
        return cam, idx

    def close(self):
        self._h.remove()


def overlay_cam(image: Image.Image, cam: np.ndarray, alpha: float = 0.40) -> Image.Image:
    import matplotlib
    img = image.convert("RGB")
    heat = Image.fromarray((cam * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    rgba = matplotlib.colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    blend = (1 - alpha) * np.asarray(img, dtype=np.float32) / 255.0 + alpha * rgba
    return Image.fromarray((np.clip(blend, 0, 1) * 255).astype(np.uint8))


def tabular_shap(pipeline, row, top_k: int = 10):
    """SHAP values of the predicted subtype for one patient row (RandomForest pipeline).

    Returns (list of {feature, value, shap}, predicted_class_index) or ([], idx) on failure.
    """
    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    x = prep.transform(row)
    cls = int(np.argmax(clf.predict_proba(x)[0]))
    try:
        import shap
        sv = shap.TreeExplainer(clf).shap_values(x)
        if isinstance(sv, list):
            v = np.asarray(sv[cls])[0]
        else:
            sv = np.asarray(sv)
            v = sv[0, :, cls] if sv.ndim == 3 else sv[0]
        order = np.argsort(-np.abs(v))[:top_k]
        names = prep.feature_names_
        return [{"feature": names[i], "value": float(x[0, i]), "shap": float(v[i])} for i in order], cls
    except Exception as e:  # explanation must never take the prediction down
        print(f"(SHAP unavailable: {e})")
        return [], cls
