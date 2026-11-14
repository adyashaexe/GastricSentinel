"""Detailed multi-page PDF report (reportlab + matplotlib charts).

Sections: 1 Summary - 2 Inputs - 3 Image analysis (class probabilities, Grad-CAM) -
4 Clinical & genomic analysis (subtype, SHAP) - 5 Multimodal fusion - 6 Model validation
(per-class table, confusion matrix, calibration, split/leakage disclosure) - 7 Limitations -
8 Methods - reviewer sign-off.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfgen import canvas as rl_canvas  # noqa: E402
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak,  # noqa: E402
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

from . import __version__
from .config import DISCLAIMER, LOW_CONFIDENCE, MIN_GENOMIC_COVERAGE, TISSUE

NAVY, BLUE, TEAL, ROSE, GREY = "#0b2a4a", "#1457a6", "#0e8f8a", "#b5446e", "#8a94a6"
W = A4[0] - 4 * cm


# ------------------------------------------------------------------ styles
def _styles():
    ss = getSampleStyleSheet()
    base = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=9.2, leading=13.2,
                          textColor=colors.HexColor("#1b2536"))
    return {
        "body": base,
        "just": ParagraphStyle("j", parent=base, alignment=TA_JUSTIFY),
        "small": ParagraphStyle("s", parent=base, fontSize=7.8, leading=10.5, textColor=colors.HexColor("#566176")),
        "h1": ParagraphStyle("h1", parent=base, fontName="Times-Bold", fontSize=15.5, leading=19,
                             textColor=colors.HexColor(NAVY), spaceBefore=12, spaceAfter=5),
        "h2": ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=10.2, leading=13,
                             textColor=colors.HexColor(BLUE), spaceBefore=8, spaceAfter=3),
        "big": ParagraphStyle("big", parent=base, fontName="Times-Bold", fontSize=26, leading=30,
                              textColor=colors.HexColor(NAVY)),
        "cell": ParagraphStyle("c", parent=base, fontSize=8.6, leading=11.4),
        "cellb": ParagraphStyle("cb", parent=base, fontSize=8.6, leading=11.4, fontName="Helvetica-Bold"),
        "warn": ParagraphStyle("w", parent=base, textColor=colors.HexColor("#8a5a00")),
    }


def _tbl(rows, widths, header=True, zebra=True, align_right_from=None):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONTSIZE", (0, 0), (-1, -1), 8.6), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#d5dbe5")),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    if zebra:
        st += [("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#f4f6fa")])]
    if align_right_from is not None:
        st.append(("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"))
    t.setStyle(TableStyle(st))
    return t


def _kv(rows, S, w1=5.2 * cm):
    data = [[Paragraph(k, S["cellb"]), Paragraph(str(v), S["cell"])] for k, v in rows]
    return _tbl(data, [w1, W - w1], header=False)


def _callout(text, S, color="#e8f1fb", edge=BLUE):
    t = Table([[Paragraph(text, S["body"])]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(color)),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(edge)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


# ------------------------------------------------------------------ charts
def _png(fig, w_cm, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image as PILImage
    w, h = PILImage.open(io.BytesIO(buf.getvalue())).size
    buf.seek(0)
    return Image(buf, width=w_cm * cm, height=w_cm * cm * h / w)


def _prob_chart(probs, top_class, title):
    items = sorted(probs.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(items) + 0.9))
    ax.barh([k for k, _ in items], [v * 100 for _, v in items],
            color=[NAVY if k == top_class else "#b9c4d6" for k, _ in items], height=0.62)
    for i, (_, v) in enumerate(items):
        ax.text(v * 100 + 1, i, f"{v:.1%}", va="center", fontsize=8, color="#1b2536")
    ax.set_xlim(0, 112)
    ax.set_xlabel("Probability (%)", fontsize=8)
    ax.set_title(title, fontsize=9.5, loc="left", color=NAVY, fontweight="bold")
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return _png(fig, 13.5)


def _shap_chart(feats):
    feats = sorted(feats, key=lambda f: abs(f["shap"]))
    fig, ax = plt.subplots(figsize=(6.4, 0.36 * len(feats) + 0.9))
    ax.barh([f["feature"] for f in feats], [f["shap"] for f in feats],
            color=[TEAL if f["shap"] >= 0 else ROSE for f in feats], height=0.62)
    ax.axvline(0, color="#566176", lw=0.8)
    ax.set_xlabel("SHAP value (effect on predicted subtype probability)", fontsize=8)
    ax.set_title("Top feature contributions", fontsize=9.5, loc="left", color=NAVY, fontweight="bold")
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return _png(fig, 13.5)


# ---------------------------------------------------------------- page furniture
class _Numbered(rl_canvas.Canvas):
    """Canvas that knows the total page count (for 'Page x of y')."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved)
        for st in self._saved:
            self.__dict__.update(st)
            self._footer(n)
            super().showPage()
        super().save()

    def _footer(self, n):
        self.setStrokeColor(colors.HexColor("#d5dbe5"))
        self.line(2 * cm, 1.55 * cm, A4[0] - 2 * cm, 1.55 * cm)
        self.setFont("Helvetica", 7.2)
        self.setFillColor(colors.HexColor("#566176"))
        self.drawString(2 * cm, 1.15 * cm, "Research prototype - not for diagnosis or treatment decisions. "
                                           "Pathologist review required.")
        self.drawRightString(A4[0] - 2 * cm, 1.15 * cm, f"Page {self._pageNumber} of {n}")


def _header(report_id, stamp):
    def draw(c, doc):
        c.saveState()
        c.setFillColor(colors.HexColor(NAVY))
        c.rect(0, A4[1] - 1.35 * cm, A4[0], 1.35 * cm, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Times-Bold", 11)
        c.drawString(2 * cm, A4[1] - 0.86 * cm, "GastricSentinel")
        c.setFont("Helvetica", 7.8)
        c.drawRightString(A4[0] - 2 * cm, A4[1] - 0.86 * cm, f"Report {report_id}  |  {stamp}")
        c.restoreState()
    return draw


# ------------------------------------------------------------------- build
def build_report(result: dict, out_path, image_path=None, gradcam_path=None, report_id: str = "",
                 model_card: dict | None = None, confusion_png=None) -> Path:
    S = _styles()
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d %H:%M UTC")
    doc = BaseDocTemplate(str(out_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                          topMargin=2.0 * cm, bottomMargin=2.1 * cm, title="GastricSentinel analysis report",
                          author="GastricSentinel")
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(2 * cm, 2.1 * cm, W, A4[1] - 4.1 * cm, id="f",
                                                             leftPadding=0, rightPadding=0, topPadding=0,
                                                             bottomPadding=0)],
                                       onPage=_header(report_id, stamp))])
    pred, used, inp = result["prediction"], result["modalities_used"], result.get("inputs") or {}
    card = model_card or {}
    cls = pred["class"]
    tissue = TISSUE.get(cls, "tissue class from the training data")
    story = []

    # ---- title block
    story += [Paragraph("Histopathology analysis report", S["h1"]),
              Paragraph(f"Report ID <b>{report_id}</b> &nbsp;|&nbsp; Generated {stamp} &nbsp;|&nbsp; "
                        f"GastricSentinel v{__version__}", S["small"]), Spacer(1, 6),
              _callout(f"<b>{DISCLAIMER}</b>", S, "#fff6e0", "#b7791f"), Spacer(1, 8)]

    # ---- 1 summary
    story.append(Paragraph("1. Summary of findings", S["h1"]))
    conf_txt = f"{pred['confidence']:.1%}" + (f" &nbsp;<font color='#8a5a00'>(below the {LOW_CONFIDENCE:.0%} "
                                              "threshold: inconclusive)</font>" if pred["low_confidence"] else "")
    left = [Paragraph("PREDICTED TISSUE CLASS", S["small"]), Paragraph(cls, S["big"]),
            Paragraph(tissue, S["body"])]
    right = _kv([("Calibrated confidence", conf_txt),
                 ("Tumour-class probability", "n/a" if result["tumour_probability"] is None
                  else f"{result['tumour_probability']:.1%}"),
                 ("Primary prediction from", "multimodal fusion" if result["mode"] == "fusion" else "image model"),
                 ("Modalities used", ", ".join(k for k, v in used.items() if v) or "none")], S, 4.0 * cm)
    t = Table([[left, right]], colWidths=[5.2 * cm, W - 5.2 * cm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [t, Spacer(1, 6)]
    reading = (f"The image model assigns this patch to <b>{cls}</b> ({tissue}) with a calibrated confidence of "
               f"{pred['confidence']:.1%}. ")
    if result["tumour_probability"] is not None:
        reading += (f"The probability mass on the tumour class is {result['tumour_probability']:.1%}; this is a model "
                    "output for this single patch, not a clinical risk score. ")
    reading += ("Results describe one image region. They do not describe the whole slide, the patient's disease "
                "status, stage, prognosis or treatment.")
    story.append(Paragraph(reading, S["just"]))
    if result["warnings"]:
        story.append(Spacer(1, 4))
        for w in result["warnings"]:
            story.append(_callout(f"<b>Warning.</b> {w}", S, "#fff6e0", "#b7791f"))
            story.append(Spacer(1, 3))

    # ---- 2 inputs
    story.append(Paragraph("2. Inputs supplied", S["h1"]))
    gc = inp.get("genomic_coverage")
    story.append(_kv([
        ("Image", "supplied (required)"),
        ("Age", "not supplied" if inp.get("age") is None else f"{inp['age']:g} years"),
        ("Sex", inp.get("sex") or "not supplied"),
        ("Stage", "not supplied" if inp.get("stage") is None else f"Stage {['I', 'II', 'III', 'IV'][int(inp['stage']) - 1]}"),
        ("Gene expression file", "not supplied" if not inp.get("genomic_provided") else
         f"supplied; covers {gc:.0%} of the model's genes "
         f"({'used' if (gc or 0) >= MIN_GENOMIC_COVERAGE else 'ignored: below ' + format(MIN_GENOMIC_COVERAGE, '.0%')})"),
    ], S))
    story.append(Paragraph("Optional inputs may be omitted. The system reports which modalities were actually used "
                           "(section 1).", S["small"]))

    # ---- 3 image analysis
    story.append(PageBreak())
    story.append(Paragraph("3. Image analysis", S["h1"]))
    story.append(_prob_chart(pred["probabilities"], cls, "Class probabilities (calibrated)"))
    imgs, caps = [], []
    for path, cap in ((image_path, "Input image"), (gradcam_path, "Grad-CAM overlay")):
        if path and Path(path).exists():
            imgs.append(Image(str(path), 7.6 * cm, 7.6 * cm, kind="proportional"))
            caps.append(Paragraph(f"<i>{cap}</i>", S["small"]))
    if imgs:
        story.append(Spacer(1, 6))
        g = Table([imgs, caps], colWidths=[W / len(imgs)] * len(imgs))
        g.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story.append(KeepTogether([Paragraph("Visual explanation (Grad-CAM)", S["h2"]), g]))
    story.append(Paragraph(
        "<b>How to read Grad-CAM.</b> Warm colours mark image regions that most increased the model's score for "
        "the predicted class; cool colours had little influence. It shows where the network looked. It is not a "
        "validated segmentation of tumour, and a confident-looking heat-map can still be wrong. It has not been "
        "assessed by pathologists in this project.", S["just"]))

    # ---- 4 clinical / genomic
    story.append(Paragraph("4. Clinical and genomic analysis", S["h1"]))
    ms = result.get("molecular_subtype")
    if ms:
        story.append(Paragraph(f"Predicted molecular subtype: <b>{ms['class']}</b> "
                               f"(model: {ms['model'].replace('_', ' ')})", S["body"]))
        story.append(Spacer(1, 3))
        rows = [["Subtype", "Probability"]] + [[k, f"{v:.1%}"] for k, v in
                                                sorted(ms["probabilities"].items(), key=lambda kv: -kv[1])]
        story.append(_tbl(rows, [8 * cm, 4 * cm], align_right_from=1))
        if ms["top_features"]:
            story.append(Spacer(1, 6))
            story.append(_shap_chart(ms["top_features"]))
            story.append(Paragraph("<b>SHAP.</b> Positive values (teal) raised, and negative values (rose) lowered, "
                                   "the probability of the predicted subtype for this patient. Feature names beginning "
                                   "with <i>gene:</i> are expression values; others are clinical inputs. SHAP "
                                   "describes the model, not biology.", S["just"]))
        story.append(Paragraph(ms["note"], S["small"]))
    else:
        story.append(Paragraph("No clinical or genomic inputs were usable, so no subtype analysis was performed.",
                               S["body"]))

    # ---- 5 fusion
    fu = result.get("experimental_fusion")
    story.append(Paragraph("5. Multimodal fusion (experimental)", S["h1"]))
    if fu:
        story.append(_kv([("Fusion prediction", f"{fu['class']} ({fu['confidence']:.1%})"),
                          ("Pairing used in training", fu["pairing_mode"]),
                          ("Patient-level correspondence", "yes" if fu["patient_level_correspondence"] else "no"),
                          ("Modalities at inference", ", ".join(k for k, v in fu["modalities"].items() if v))], S))
        if not fu["patient_level_correspondence"]:
            story.append(Spacer(1, 4))
            story.append(_callout("The fusion model was trained on randomly paired image and clinical/genomic records "
                                  "from different cohorts. It is shown for reference only and was not used for the "
                                  "primary prediction.", S, "#fff6e0", "#b7791f"))
    else:
        story.append(Paragraph("Not run for this analysis (no clinical or genomic input, or no fusion model).", S["body"]))

    # ---- 6 validation
    story.append(PageBreak())
    story.append(Paragraph("6. Model validation and data disclosure", S["h1"]))
    im = card.get("image")
    if im:
        story.append(Paragraph("Held-out test performance of the image model (from training artifacts)", S["h2"]))
        ci = im.get("ci95_accuracy")
        story.append(_kv([
            ("Test images", im["n_test"]),
            ("Accuracy", f"{im['accuracy']:.1%}" + (f"  (95% CI {ci[0]:.1%} - {ci[1]:.1%})" if ci else "")),
            ("Balanced accuracy", f"{im['balanced_accuracy']:.1%}"),
            ("Macro precision / recall / F1", f"{im['macro_precision']:.3f} / {im['macro_recall']:.3f} / {im['macro_f1']:.3f}"),
            ("Expected calibration error", "n/a" if im.get("ece") is None else f"{im['ece']:.3f}"
             + (f"  (temperature {im['temperature']:.2f})" if im.get("temperature") else "")),
        ], S))
        story.append(Paragraph("Averaging: macro = unweighted mean over classes. F1 is the mean of per-class F1.",
                               S["small"]))
        rows = [["Class", "Precision", "Recall", "F1", "Support"]] + [
            [r["class"], f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}", r["support"]]
            for r in im["per_class"]]
        story += [Spacer(1, 4), _tbl(rows, [4 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm], align_right_from=1)]
        if confusion_png and Path(confusion_png).exists():
            story += [Spacer(1, 6), KeepTogether([Paragraph("Confusion matrix (test split)", S["h2"]),
                                                  Image(str(confusion_png), 9.5 * cm, 9.5 * cm, kind="proportional")])]
    else:
        story.append(_callout("No validation artifacts were found next to the deployed model, so performance "
                              "figures cannot be shown. Do not rely on this model without evaluating it.", S,
                              "#fdecef", ROSE))
    sp, la = card.get("split"), card.get("leakage_audit")
    if sp:
        story.append(Paragraph("Data splitting and leakage", S["h2"]))
        rows = [("Split unit", "patient / slide groups" if not sp["leakage_risk"] else "individual patches (random)"),
                ("Leakage risk", "<b>YES: patches from one patient may be in both train and test; accuracy may be "
                                 "inflated</b>" if sp["leakage_risk"] else "no group overlap between splits"),
                ("Images per split", sp.get("counts"))]
        nd = (la or {}).get("test_vs_train")
        if nd:
            rows.append(("Near-duplicate test images", f"{nd['near_duplicates_le_threshold']} of "
                                                      f"{nd['n_images']} ({nd['fraction_near_duplicate']:.1%}) have a near-identical image in training"))
        story.append(_kv(rows, S))
    tb = card.get("tabular")
    if tb:
        story.append(Paragraph("Clinical / genomic subtype model", S["h2"]))
        story.append(_kv([("Protocol", tb["protocol"]), ("Patients", tb["n_patients"]),
                          ("Cross-validated accuracy / macro-F1", f"{tb['accuracy']:.1%} / {tb['macro_f1']:.3f}"),
                          ("Subtypes", ", ".join(tb["classes"])),
                          ("Excluded (too few patients)", tb.get("excluded_classes") or "none")], S))
    fu_c = card.get("fusion")
    if fu_c:
        story.append(Paragraph("Fusion data pairing", S["h2"]))
        story.append(Paragraph(fu_c["disclosure"], S["just"]))

    # ---- 7 limitations
    story.append(Paragraph("7. Limitations", S["h1"]))
    for b in ["Research prototype; not validated for clinical use and not a medical device.",
              "Results describe a single image patch, not a whole slide or a patient.",
              "Trained on public datasets; performance on other scanners, stains or hospitals is unknown.",
              "Tumour-class probability is a model output, not a calibrated clinical risk.",
              "Grad-CAM and SHAP explain model behaviour and have not been validated by pathologists.",
              "Rare molecular subtypes are estimated from very few patients and are unreliable."]:
        story.append(Paragraph(f"&bull; {b}", S["body"]))

    # ---- 8 methods
    story.append(Paragraph("8. Methods summary", S["h1"]))
    story.append(Paragraph(
        "Images are resized to the model's input size and normalised with ImageNet statistics. A ResNet-50 classifier "
        "produces class scores; confidence is calibrated with a temperature fitted on a validation split (kept only "
        "if it improves validation calibration). Grad-CAM uses gradients of the predicted class score with respect to "
        "the final residual stage. Clinical features (age, sex, stage, with missing-value indicators) and the "
        "highest-variance genes are standardised using training-patient statistics only and passed to a random-forest "
        "subtype model explained with SHAP (TreeExplainer). The optional fusion network concatenates image features "
        "with clinical and genomic embeddings and is trained with modality dropout so that missing inputs are handled "
        "by design.", S["just"]))

    # ---- sign-off
    story += [Spacer(1, 14), KeepTogether([Paragraph("Reviewer sign-off", S["h2"]), _tbl([
        ["Reviewed by (name, role)", "Signature", "Date"], ["", "", ""], ["", "", ""]],
        [7.4 * cm, 5.6 * cm, 4 * cm], zebra=False)])]
    doc.build(story, canvasmaker=_Numbered)
    return Path(out_path)
