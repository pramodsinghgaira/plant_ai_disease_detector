import html
import io
import datetime

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from disease_info import lookup, display_name, LOW, MODERATE, HIGH, NONE
from leaf_check import LeafGate, SUPPORTED_CROPS, LEAF_MIN_PROB, UNSUPPORTED_MIN_PROB
from model import (
    CLASSES, get_device, load_classifier, open_image, predict,
    prepare_input_image, to_tensor, check_photo_quality,
)

# ----------------------------------------------------------------------
# AI PLANT DISEASE DETECTOR
# Owner: Pramod Singh Gaira
# EfficientNet-B4 (Khawajaa/plant-disease-detector) + leaf pre-check + Grad-CAM
# Features: Photo Quality Check · PDF Report · Low-Confidence Threshold
#           Prevention Tips · Organic Options · Test-Time Augmentation (TTA)
# ----------------------------------------------------------------------

OWNER = "Pramod Singh Gaira"
MODEL_LABEL = "Khawajaa/plant-disease-detector (EfficientNet-B4)"
GATE_LABEL = "openai/clip-vit-base-patch32"

# Low-confidence threshold — below this confidence the app shows a warning
# and the report flags the diagnosis as uncertain.
LOW_CONF_THRESHOLD = 30  # percent  ← was 50%; now 30% as requested

st.set_page_config(page_title="AI Plant Disease Detector", page_icon="🌿", layout="wide")


# ======================================================================
# MODELS + GRAD-CAM
# ======================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    return load_classifier(get_device())


@st.cache_resource(show_spinner=False)
def load_gate():
    return LeafGate(get_device())


def _jet(x: np.ndarray) -> np.ndarray:
    """Tiny jet-style colormap (blue -> cyan -> yellow -> red); x in [0,1]."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def grad_cam(model, image: Image.Image, class_idx: int, view_size: int = 448):
    """Grad-CAM for `class_idx`. Returns (crop, overlay, heatmap) as PIL images, or None on failure."""
    try:
        import torch
        import torch.nn.functional as F

        device = next(model.parameters()).device
        x = to_tensor(prepare_input_image(image)).to(device)
        store = {}

        def fwd_hook(_, __, out):
            store["act"] = out
            out.register_hook(lambda g: store.__setitem__("grad", g))

        handle = model.grad_cam_target_layer.register_forward_hook(fwd_hook)
        try:
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                logits = model(x)
                logits[0, class_idx].backward()
        finally:
            handle.remove()

        act, grad = store["act"].detach(), store["grad"].detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(view_size, view_size), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        crop = ImageOps.fit(image, (view_size, view_size), method=Image.BICUBIC)
        base = np.asarray(crop, dtype=np.float32) / 255.0
        heat = _jet(cam)
        alpha = (0.25 + 0.55 * cam)[..., None]
        overlay = np.clip(base * (1 - alpha) + heat * alpha, 0, 1)
        return (
            crop,
            Image.fromarray((overlay * 255).astype(np.uint8)),
            Image.fromarray((heat * 255).astype(np.uint8)),
        )
    except Exception as exc:
        st.session_state["gradcam_error"] = str(exc)
        return None


# ======================================================================
# HELPERS
# ======================================================================
RISK_STYLE = {
    NONE:     ("#22c55e", "rgba(34,197,94,0.14)",   "✅"),
    LOW:      ("#84cc16", "rgba(132,204,22,0.14)",  "🟢"),
    MODERATE: ("#f59e0b", "rgba(245,158,11,0.14)",  "🟠"),
    HIGH:     ("#ef4444", "rgba(239,68,68,0.14)",   "🔴"),
}


def html_block(markup: str):
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def show_image(img, caption=None):
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:
        st.image(img, caption=caption)


def esc(s) -> str:
    return html.escape(str(s))


def chips_html() -> str:
    return "".join(f'<span class="chip">{esc(name)}</span>' for name, _ in SUPPORTED_CROPS)


def render_footer():
    html_block(
        f"""
<div class="footer">
  Model: {esc(MODEL_LABEL)} (Hugging Face, free &amp; pretrained) · Leaf pre-check: {esc(GATE_LABEL)}<br>
  App by <b>{esc(OWNER)}</b> · AI predictions are advisory, not a substitute for expert diagnosis
</div>
"""
    )


def render_rejection(icon: str, title: str, message: str):
    html_block(
        f"""
<div class="reject">
  <div class="big">{icon}</div>
  <h2>{esc(title)}</h2>
  <p>{esc(message)}</p>
  <div class="chips-label">Supported plants</div>
  <div class="chips">{chips_html()}</div>
</div>
"""
    )


def render_gate_details(check):
    with st.expander("Pre-check details"):
        st.write(
            f"**Leaf score:** {check.leaf_prob:.0%} (needs ≥ {LEAF_MIN_PROB:.0%}) — "
            f"{'passed' if check.is_leaf else 'failed'}"
            + ("" if check.is_leaf else f" · looks most like: *{check.top_non_leaf}*")
        )
        st.write(
            f"**Closest supported plant:** {check.best_supported} ({check.best_supported_prob:.0%})  \n"
            f"**Closest unsupported plant:** {check.best_unsupported} ({check.best_unsupported_prob:.0%}) — "
            f"rejected only if ≥ {UNSUPPORTED_MIN_PROB:.0%} and ahead of every supported plant."
        )
        st.caption("Zero-shot CLIP scores; they are a screening aid, not a diagnosis.")


# ======================================================================
# PHOTO QUALITY RENDER
# ======================================================================
def render_photo_quality(qc: dict):
    """Render a compact photo quality banner."""
    score = qc["score"]
    passed = qc["passed"]
    issues = qc["issues"]
    warnings = qc["warnings"]

    if score >= 80:
        bar_color = "#22c55e"
        label = "Good"
        icon = "📸"
    elif score >= 55:
        bar_color = "#f59e0b"
        label = "Fair"
        icon = "⚠️"
    else:
        bar_color = "#ef4444"
        label = "Poor"
        icon = "❌"

    html_block(f"""
<div class="qc-bar">
  <div class="qc-left">
    <span class="qc-icon">{icon}</span>
    <span class="qc-label">Photo Quality: <b>{label}</b></span>
    <span class="qc-score">{score}/100</span>
  </div>
  <div class="qc-track"><div class="qc-fill" style="width:{score}%;background:{bar_color};"></div></div>
</div>
""")

    if issues:
        for issue in issues:
            st.error(f"📷 {issue}")
    if warnings:
        for w in warnings:
            st.warning(f"💡 {w}")

    if not passed:
        st.info("The diagnosis will still run but may be less accurate. For best results, retake the photo.")


# ======================================================================
# PDF REPORT GENERATION
# ======================================================================
def generate_pdf_report(
    info: dict,
    conf: float,
    predictions: list,
    qc: dict,
    low_conf: bool,
    tta_std: float | None,
    image: Image.Image,
) -> bytes:
    """Generate a PDF diagnosis report using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, KeepTogether,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        import io as _io

        buf = _io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )

        styles = getSampleStyleSheet()
        # Custom styles
        title_style = ParagraphStyle("ReportTitle", fontSize=20, spaceAfter=6,
                                     textColor=colors.HexColor("#166534"),
                                     fontName="Helvetica-Bold", alignment=TA_CENTER)
        sub_style   = ParagraphStyle("Sub", fontSize=10, spaceAfter=3,
                                     textColor=colors.HexColor("#6b7280"), alignment=TA_CENTER)
        h2_style    = ParagraphStyle("H2", fontSize=13, spaceBefore=14, spaceAfter=6,
                                     textColor=colors.HexColor("#14532d"),
                                     fontName="Helvetica-Bold", borderPad=4)
        body_style  = ParagraphStyle("Body", fontSize=10, spaceAfter=4,
                                     textColor=colors.HexColor("#1f2937"), leading=14)
        bullet_style = ParagraphStyle("Bullet", fontSize=10, spaceAfter=3,
                                      leftIndent=14, textColor=colors.HexColor("#374151"),
                                      leading=14, bulletIndent=4)
        warn_style  = ParagraphStyle("Warn", fontSize=10, spaceAfter=6,
                                     textColor=colors.HexColor("#92400e"),
                                     backColor=colors.HexColor("#fef3c7"),
                                     borderPad=6, leading=14)
        ok_style    = ParagraphStyle("OK", fontSize=10, spaceAfter=6,
                                     textColor=colors.HexColor("#166534"),
                                     backColor=colors.HexColor("#f0fdf4"),
                                     borderPad=6, leading=14)

        story = []

        # --- Header ---
        story.append(Paragraph("🌿 AI Plant Disease Detector", title_style))
        story.append(Paragraph(f"Diagnosis Report · Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", sub_style))
        story.append(Paragraph(f"App by {OWNER} · Model: {MODEL_LABEL}", sub_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#16a34a"), spaceAfter=12))

        # --- Low-confidence warning ---
        if low_conf:
            story.append(Paragraph(
                f"⚠️  LOW CONFIDENCE WARNING: The model's top prediction is only {conf:.1f}%, "
                f"which is below the reliable threshold of {LOW_CONF_THRESHOLD}%. "
                "This diagnosis should be treated as uncertain. Consult the top-5 predictions "
                "and verify with an agronomist.",
                warn_style
            ))

        # --- Photo Quality ---
        story.append(Paragraph("📸 Photo Quality", h2_style))
        qc_score = qc['score']
        qc_label = "Good" if qc_score >= 80 else "Fair" if qc_score >= 55 else "Poor"
        qc_color = colors.HexColor("#166534") if qc_score >= 80 else (
            colors.HexColor("#92400e") if qc_score >= 55 else colors.HexColor("#991b1b")
        )
        qc_style = ParagraphStyle("QC", fontSize=11, spaceAfter=4,
                                   textColor=qc_color, fontName="Helvetica-Bold")
        story.append(Paragraph(f"Quality Score: {qc_score}/100 ({qc_label})", qc_style))
        if qc["issues"]:
            for issue in qc["issues"]:
                story.append(Paragraph(f"• Issue: {issue}", bullet_style))
        if qc["warnings"]:
            for w in qc["warnings"]:
                story.append(Paragraph(f"• Note: {w}", bullet_style))
        if not qc["issues"] and not qc["warnings"]:
            story.append(Paragraph("✅ No photo quality issues detected.", body_style))

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d1fae5"), spaceAfter=8))

        # --- Diagnosis Summary ---
        story.append(Paragraph("🔍 Diagnosis Summary", h2_style))

        risk_color_map = {
            NONE:     colors.HexColor("#166534"),
            LOW:      colors.HexColor("#3f6212"),
            MODERATE: colors.HexColor("#92400e"),
            HIGH:     colors.HexColor("#991b1b"),
        }
        rc = risk_color_map.get(info["risk"], colors.black)

        summary_data = [
            ["Field", "Value"],
            ["Crop",        info["crop"]],
            ["Diagnosis",   info["disease"]],
            ["Pathogen",    info["pathogen"]],
            ["Type",        info["kind"]],
            ["Confidence",  f"{conf:.1f}%" + (" ⚠️ LOW" if low_conf else "")],
            ["Risk Level",  info["risk"]],
        ]
        if tta_std is not None:
            summary_data.append(["TTA Consistency", f"Std dev {tta_std*100:.1f}% (lower = more stable)"])

        tbl = Table(summary_data, colWidths=[5*cm, 12*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#14532d")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 10),
            ("BACKGROUND",  (0, 1), (-1, -1), colors.HexColor("#f0fdf4")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0fdf4"), colors.HexColor("#dcfce7")]),
            ("TEXTCOLOR",   (0, 1), (0, -1), colors.HexColor("#166534")),
            ("FONTNAME",    (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 1), (-1, -1), 10),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#bbf7d0")),
            ("ROWBACKGROUNDS", (0, 6), (-1, 6), [colors.HexColor("#fef3c7")]),
            ("TEXTCOLOR",   (1, 6), (1, 6), rc),
            ("FONTNAME",    (1, 6), (1, 6), "Helvetica-Bold"),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",  (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 10))

        # --- Top-5 Predictions ---
        story.append(Paragraph("📊 Top-5 Predictions (TTA-Averaged)", h2_style))
        pred_data = [["Rank", "Disease", "Confidence", "TTA Std Dev"]]
        for i, p in enumerate(predictions):
            row = [
                str(i + 1),
                display_name(p["label"]),
                f"{p['score']*100:.1f}%",
                f"{p['tta_std']*100:.1f}%" if p.get('tta_std') is not None else "—",
            ]
            pred_data.append(row)

        pred_tbl = Table(pred_data, colWidths=[1.5*cm, 9*cm, 3*cm, 3.5*cm])
        pred_tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#14532d")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f9fafb"), colors.HexColor("#f0fdf4")]),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#d1fae5")),
            ("FONTNAME",    (0, 1), (-1, 1), "Helvetica-Bold"),
            ("BACKGROUND",  (0, 1), (-1, 1), colors.HexColor("#dcfce7")),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))
        story.append(pred_tbl)
        story.append(Spacer(1, 10))

        # --- Treatment ---
        story.append(Paragraph("💊 Treatment", h2_style))
        story.append(Paragraph(info["treatment"], body_style))
        story.append(Spacer(1, 6))

        # --- Prevention Tips ---
        story.append(Paragraph("🛡️ Prevention Tips", h2_style))
        for tip in info.get("prevention", []):
            story.append(Paragraph(f"• {tip}", bullet_style))
        story.append(Spacer(1, 6))

        # --- Organic Options ---
        story.append(Paragraph("🌱 Organic & Biological Options", h2_style))
        for opt in info.get("organic", []):
            story.append(Paragraph(f"• {opt}", bullet_style))

        story.append(Spacer(1, 12))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d1fae5"), spaceAfter=6))
        story.append(Paragraph(
            "⚠️ Disclaimer: AI predictions are advisory only. Always confirm diagnosis with a "
            "qualified agronomist before applying any treatment. Follow all product labels and "
            "local regulations.",
            ParagraphStyle("Disclaimer", fontSize=8, textColor=colors.HexColor("#9ca3af"),
                           alignment=TA_CENTER, leading=12)
        ))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        return None


# ======================================================================
# STYLING
# ======================================================================
html_block(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
"""
)
st.markdown(
    """
<style>
:root{
  --bg-1:#05110a; --bg-2:#0b1e13;
  --accent:#22c55e; --accent-2:#4ade80;
  --card:rgba(255,255,255,0.05); --border:rgba(255,255,255,0.10);
  --soft:rgba(232,245,233,0.62); --text:#eafaee;
}
html, body, [class*="css"]{font-family:'Inter',sans-serif;}
.stApp{
  background:
    radial-gradient(circle at 12% 10%, rgba(34,197,94,0.20), transparent 42%),
    radial-gradient(circle at 90% 18%, rgba(74,222,128,0.13), transparent 40%),
    radial-gradient(circle at 50% 105%, rgba(16,185,129,0.12), transparent 50%),
    linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 60%, #07170e 100%);
  background-attachment:fixed;
}
#MainMenu, footer, header{visibility:hidden;}
.block-container{padding-top:1.6rem; padding-bottom:3rem; max-width:1080px;}

@keyframes fadeUp{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:translateY(0);}}
@keyframes fadeDown{from{opacity:0;transform:translateY(-14px);}to{opacity:1;transform:translateY(0);}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(74,222,128,.45);}50%{box-shadow:0 0 0 8px rgba(74,222,128,0);}}

/* HERO */
.hero{position:relative; overflow:hidden; padding:38px 34px; border-radius:26px;
  background:linear-gradient(135deg, rgba(34,197,94,0.22), rgba(16,120,80,0.07));
  border:1px solid var(--border); box-shadow:0 24px 70px -24px rgba(0,0,0,.7);
  animation:fadeDown .7s ease; display:flex; justify-content:space-between; gap:24px; align-items:center; flex-wrap:wrap;}
.hero::before{content:""; position:absolute; inset:0; pointer-events:none;
  background:radial-gradient(circle at 25% -15%, rgba(74,222,128,.38), transparent 55%);}
.hero-left{position:relative; max-width:600px;}
.badge{display:inline-block; font-size:11.5px; letter-spacing:1.5px; font-weight:600; text-transform:uppercase;
  color:var(--accent-2); background:rgba(34,197,94,.12); border:1px solid rgba(74,222,128,.35);
  padding:5px 14px; border-radius:999px; margin-bottom:14px;}
.hero h1{font-family:'Space Grotesk',sans-serif; font-size:clamp(28px,4.6vw,42px); font-weight:700; margin:0; color:#f2fbf3; letter-spacing:-.5px;}
.hero h1 span{background:linear-gradient(90deg,var(--accent-2),#a7f3d0); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.hero p{color:var(--soft); margin:10px 0 0; font-size:15.5px;}
.owner{position:relative; display:flex; align-items:center; gap:12px; padding:12px 18px; border-radius:16px;
  background:rgba(0,0,0,.25); border:1px solid var(--border);}
.owner .av{width:42px;height:42px;border-radius:50%; display:flex;align-items:center;justify-content:center;
  font-family:'Space Grotesk',sans-serif; font-weight:700; color:#06120b; background:linear-gradient(135deg,var(--accent-2),#16a34a);}
.owner small{display:block; font-size:11px; letter-spacing:1.2px; text-transform:uppercase; color:var(--soft);}
.owner b{font-family:'Space Grotesk',sans-serif; font-size:16px; color:#f2fbf3;}

/* SECTION LABEL */
.section-label{display:flex; align-items:center; gap:10px; margin:30px 0 14px; font-family:'Space Grotesk',sans-serif;
  font-size:16px; font-weight:600; color:#e8f5e9; animation:fadeUp .6s ease;}
.section-label .num{width:25px;height:25px;border-radius:8px; display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,var(--accent),#16a34a); color:#06120b; font-size:12px; font-weight:700;}

/* TABS */
div[data-baseweb="tab-list"]{gap:8px;}
button[data-baseweb="tab"]{background:rgba(255,255,255,.05)!important; border:1px solid var(--border)!important;
  border-radius:12px!important; padding:8px 18px!important; color:var(--text)!important;}
button[data-baseweb="tab"][aria-selected="true"]{background:rgba(34,197,94,.18)!important; border-color:var(--accent-2)!important;}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"]{display:none!important;}

[data-testid="stFileUploaderDropzone"]{background:rgba(255,255,255,.03)!important; border:1.5px dashed rgba(74,222,128,.38)!important; border-radius:16px!important;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--accent-2)!important; background:rgba(34,197,94,.06)!important;}
.stButton>button, [data-testid="stCameraInput"] button, .stDownloadButton>button{
  border-radius:12px!important; border:1px solid rgba(74,222,128,.4)!important;
  background:linear-gradient(135deg,rgba(34,197,94,.22),rgba(34,197,94,.06))!important; color:#eafff0!important; transition:all .2s;}
.stButton>button:hover, .stDownloadButton>button:hover{border-color:var(--accent-2)!important; box-shadow:0 6px 20px -6px rgba(34,197,94,.5); transform:translateY(-1px);}
img{border-radius:16px!important; border:1px solid var(--border);}
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img{border-radius:16px!important;}

/* CARDS */
.card{background:var(--card); border:1px solid var(--border); border-radius:20px; padding:22px; backdrop-filter:blur(14px); animation:fadeUp .55s ease;}
.card h3{font-family:'Space Grotesk',sans-serif; font-size:14px; letter-spacing:1.2px; text-transform:uppercase; color:var(--soft); margin:0 0 14px; font-weight:600;}

/* VERDICT */
.verdict{position:relative; overflow:hidden; border-radius:22px; padding:24px 26px; border:1px solid var(--border);
  background:var(--card); animation:fadeUp .5s ease; display:flex; gap:22px; align-items:center; flex-wrap:wrap;}
.verdict .glow{position:absolute; top:-60px; right:-50px; width:200px; height:200px; border-radius:50%; filter:blur(55px); opacity:.45;}
.ring{--p:0; position:relative; width:112px; height:112px; border-radius:50%; flex:none;
  background:conic-gradient(var(--c) calc(var(--p)*1%), rgba(255,255,255,.08) 0);
  display:flex; align-items:center; justify-content:center;}
.ring::before{content:""; position:absolute; inset:9px; border-radius:50%; background:#0a1a10;}
.ring span{position:relative; font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:22px; color:#f2fbf3;}
.ring small{position:absolute; bottom:25px; font-size:9px; letter-spacing:1px; color:var(--soft); text-transform:uppercase;}
.verdict-main{position:relative; flex:1; min-width:220px;}
.verdict-main .crop{font-size:12px; letter-spacing:1.3px; text-transform:uppercase; color:var(--soft);}
.verdict-main h2{font-family:'Space Grotesk',sans-serif; font-size:clamp(22px,3.4vw,30px); margin:4px 0 12px; color:#f4fff6;}
.pill{display:inline-flex; align-items:center; gap:8px; padding:8px 16px; border-radius:999px; font-weight:700; font-size:14px; border:1px solid;}
.dot{width:9px;height:9px;border-radius:50%; display:inline-block; animation:pulse 2s infinite;}

/* METRIC TILES */
.tiles{display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin-top:14px;}
.tile{background:var(--card); border:1px solid var(--border); border-radius:16px; padding:16px 18px; animation:fadeUp .6s ease;}
.tile .k{font-size:11px; letter-spacing:1.3px; text-transform:uppercase; color:var(--soft); margin-bottom:6px;}
.tile .v{font-family:'Space Grotesk',sans-serif; font-size:18px; font-weight:600; color:#f2fbf3; word-break:break-word;}
.tile .v i{font-weight:500;}

/* TOP-5 */
.bar-row{margin-bottom:14px;}
.bar-row .top{display:flex; justify-content:space-between; gap:10px; font-size:14px; color:var(--text);}
.bar-row .top span:last-child{font-variant-numeric:tabular-nums; color:var(--soft);}
.track{height:9px; border-radius:99px; background:rgba(255,255,255,.07); margin-top:6px; overflow:hidden;}
.fill{height:100%; border-radius:99px; background:linear-gradient(90deg,var(--accent),var(--accent-2));}
.bar-row.first .fill{background:linear-gradient(90deg,var(--c),var(--c2));}
.bar-row.first .top{font-weight:600;}

/* TREATMENT */
.treat{border-left:4px solid var(--c); border-radius:18px; padding:20px 22px; background:var(--card);
  border-top:1px solid var(--border); border-right:1px solid var(--border); border-bottom:1px solid var(--border); animation:fadeUp .6s ease;}
.treat h3{margin:0 0 8px; font-family:'Space Grotesk',sans-serif; font-size:17px; color:#f2fbf3;}
.treat p{margin:0; color:rgba(232,245,233,.85); line-height:1.65; font-size:15px;}

/* PREVENTION & ORGANIC */
.tips-block{border-radius:18px; padding:18px 22px; background:var(--card);
  border:1px solid var(--border); animation:fadeUp .6s ease; margin-top:12px;}
.tips-block h3{margin:0 0 10px; font-family:'Space Grotesk',sans-serif; font-size:15px; color:#f2fbf3;}
.tips-block ul{margin:0; padding-left:18px; color:rgba(232,245,233,.85); line-height:1.7; font-size:14px;}
.tips-block li{margin-bottom:4px;}

/* PHOTO QUALITY BAR */
.qc-bar{display:flex; align-items:center; gap:14px; background:rgba(255,255,255,.04);
  border:1px solid var(--border); border-radius:14px; padding:12px 16px; margin-bottom:12px; flex-wrap:wrap;}
.qc-left{display:flex; align-items:center; gap:10px; flex:1; min-width:200px;}
.qc-icon{font-size:20px;}
.qc-label{font-size:14px; color:var(--text);}
.qc-score{font-family:'Space Grotesk',sans-serif; font-size:14px; font-weight:700; color:#f2fbf3;
  background:rgba(255,255,255,.08); padding:2px 10px; border-radius:999px;}
.qc-track{flex:2; min-width:120px; height:8px; border-radius:99px; background:rgba(255,255,255,.08); overflow:hidden;}
.qc-fill{height:100%; border-radius:99px; transition:width .5s ease;}

/* GRADCAM */
.legend{height:10px; border-radius:99px; margin-top:10px; background:linear-gradient(90deg,#0000ff,#00ffff,#ffff00,#ff0000);}
.legend-labels{display:flex; justify-content:space-between; font-size:11.5px; color:var(--soft); margin-top:5px;}
.note{color:var(--soft); font-size:13.5px; line-height:1.55; margin:6px 0 0;}

.reject{text-align:center; padding:34px 24px; border-radius:22px; border:1px solid rgba(245,158,11,.45);
  background:linear-gradient(160deg, rgba(245,158,11,.10), rgba(255,255,255,.02)); animation:fadeUp .5s ease;}
.reject .big{font-size:42px; margin-bottom:6px;}
.reject h2{font-family:'Space Grotesk',sans-serif; font-size:clamp(20px,3vw,26px); margin:0 0 8px; color:#fff4e0;}
.reject p{color:rgba(232,245,233,.8); font-size:15px; line-height:1.6; margin:0 auto 18px; max-width:520px;}
.chips-label{font-size:11px; letter-spacing:1.3px; text-transform:uppercase; color:var(--soft); margin-bottom:8px;}
.chips{display:flex; flex-wrap:wrap; justify-content:center; gap:8px;}
.chip{padding:5px 12px; border-radius:999px; font-size:12.5px; color:var(--text);
  background:rgba(34,197,94,.12); border:1px solid rgba(74,222,128,.3);}
.empty .chips{margin-top:14px;}
.empty{text-align:center; padding:44px 20px; border-radius:20px; border:1px dashed rgba(74,222,128,.28);
  background:rgba(255,255,255,.02); color:var(--soft); font-size:15px; animation:fadeUp .6s ease;}
.empty .big{font-size:38px; margin-bottom:10px;}
.footer{text-align:center; margin-top:44px; padding-top:20px; border-top:1px solid var(--border);
  color:rgba(232,245,233,.4); font-size:12.5px; line-height:1.8;}
.footer b{color:rgba(232,245,233,.75);}

/* TTA badge */
.tta-badge{display:inline-block; font-size:10px; letter-spacing:1px; font-weight:600; text-transform:uppercase;
  color:#a7f3d0; background:rgba(34,197,94,.10); border:1px solid rgba(74,222,128,.25);
  padding:2px 9px; border-radius:999px; margin-left:8px; vertical-align:middle;}
</style>
""",
    unsafe_allow_html=True,
)

# ======================================================================
# HERO
# ======================================================================
html_block(
    f"""
<div class="hero">
  <div class="hero-left">
    <span class="badge">🌿 Computer Vision · Plant Health</span>
    <h1>AI Plant <span>Disease Detector</span></h1>
    <p>Browse a leaf photo or capture one live — get diagnosis, risk level, pathogen, treatment,
       prevention tips, organic options, and an explainable Grad-CAM heatmap.</p>
  </div>
  <div class="owner"><div class="av">PS</div><div><small>Owner</small><b>{esc(OWNER)}</b></div></div>
</div>
"""
)

# ======================================================================
# SIDEBAR — Settings
# ======================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    use_tta = st.toggle("Test-Time Augmentation (TTA)", value=True,
                        help="Average predictions over 6 augmented views for better accuracy. Adds ~0.5s.")
    conf_threshold = st.slider(
        "Low-confidence threshold (%)", min_value=10, max_value=70, value=LOW_CONF_THRESHOLD,
        help="Diagnoses below this confidence % are flagged as uncertain in the UI and report."
    )
    show_organic = st.toggle("Show organic/biological options", value=True)
    show_prevention = st.toggle("Show prevention tips", value=True)
    st.divider()
    st.caption(f"Model: {MODEL_LABEL}")
    st.caption(f"Gate: {GATE_LABEL}")

# ======================================================================
# INPUT
# ======================================================================
html_block('<div class="section-label"><span class="num">1</span> Provide a leaf image</div>')

tab_upload, tab_cam = st.tabs(["📤 Browse image", "📷 Capture with webcam"])
with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose a leaf image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"], key="uploader"
    )
with tab_cam:
    st.caption("Allow camera access, hold the leaf close in good light, then press *Take photo*.")
    camera_file = st.camera_input("Capture plant leaf", key="camera", label_visibility="collapsed")

image_source, caption = None, "Leaf Image"
if uploaded_file is not None and camera_file is not None:
    choice = st.radio(
        "Both an uploaded file and a webcam photo are available. Analyse which one?",
        ["📤 Uploaded image", "📷 Webcam capture"],
        horizontal=True,
    )
    image_source, caption = (
        (uploaded_file, "Uploaded Leaf") if choice.startswith("📤") else (camera_file, "Captured Leaf")
    )
elif uploaded_file is not None:
    image_source, caption = uploaded_file, "Uploaded Leaf"
elif camera_file is not None:
    image_source, caption = camera_file, "Captured Leaf"

# ======================================================================
# RESULTS
# ======================================================================
if image_source is None:
    html_block(
        f"""
<div class="empty"><div class="big">🌱</div>
Browse a leaf image or capture one with your webcam to get started.<br>
Only leaves of these 14 plants are supported:
<div class="chips">{chips_html()}</div></div>
"""
    )
else:
    image = open_image(image_source)

    # ---- Photo Quality Check -----------------------------------------------
    html_block('<div class="section-label"><span class="num">2</span> Photo Quality Check</div>')
    with st.spinner("Checking photo quality..."):
        qc = check_photo_quality(image)
    render_photo_quality(qc)

    # ---- Pre-check: is it a leaf? -------------------------------------------
    with st.spinner("Loading pre-check model... (first run only)"):
        gate = load_gate()
    with st.spinner("Checking that this is a supported plant leaf..."):
        check = gate.check(image)

    if not check.ok:
        html_block('<div class="section-label"><span class="num">3</span> Image check</div>')
        left, right = st.columns([1, 1.25], gap="large")
        with left:
            show_image(image, caption)
        with right:
            if not check.is_leaf:
                render_rejection(
                    "🍃",
                    "Please upload a leaf image",
                    "This doesn't look like a plant leaf, so no diagnosis was made. "
                    "Upload a clear, close-up photo of a single leaf in good light.",
                )
            else:
                render_rejection(
                    "🌱",
                    "This plant isn't supported",
                    "That looks like a leaf, but not from one of the plants this detector was trained on, "
                    "so no diagnosis was made. Please upload a leaf from one of the plants below.",
                )
        render_gate_details(check)
        render_footer()
        st.stop()

    with st.spinner("Loading AI model... (first run only)"):
        model = load_model()

    spinner_msg = "Analysing with TTA (6 augmented views)..." if use_tta else "Analysing image..."
    with st.spinner(spinner_msg):
        predictions = predict(model, image, top_k=5, use_tta=use_tta)

    top = predictions[0]
    info = lookup(top["label"])
    conf = top["score"] * 100
    tta_std = top.get("tta_std")
    color, soft_bg, icon = RISK_STYLE[info["risk"]]
    healthy = info["kind"] == "Healthy"
    low_conf = conf < conf_threshold

    html_block('<div class="section-label"><span class="num">3</span> Diagnosis</div>')
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        show_image(image, caption)
    with right:
        tta_badge = '<span class="tta-badge">TTA</span>' if use_tta else ""
        tta_tile = (
            f'<div class="tile"><div class="k">TTA Std Dev</div><div class="v">{tta_std*100:.1f}%</div></div>'
            if use_tta and tta_std is not None else ""
        )
        html_block(
            f"""
<div class="verdict">
  <div class="glow" style="background:{color};"></div>
  <div class="ring" style="--p:{conf:.1f};--c:{color};"><span>{conf:.1f}%</span><small>conf.</small></div>
  <div class="verdict-main">
    <div class="crop">{esc(info['crop'])}</div>
    <h2>{esc(info['disease'])}{tta_badge}</h2>
    <span class="pill" style="background:{soft_bg};color:{color};border-color:{color}66;">
      <span class="dot" style="background:{color};"></span>{icon} Risk Level: {esc(info['risk'])}
    </span>
  </div>
</div>
<div class="tiles">
  <div class="tile"><div class="k">Pathogen</div><div class="v"><i>{esc(info['pathogen'])}</i></div></div>
  <div class="tile"><div class="k">Type</div><div class="v">{esc(info['kind'])}</div></div>
  <div class="tile"><div class="k">Confidence</div><div class="v">{conf:.1f}%</div></div>
  <div class="tile"><div class="k">Risk Level</div><div class="v" style="color:{color};">{esc(info['risk'])}</div></div>
  {tta_tile}
</div>
"""
        )

    if low_conf:
        st.warning(
            f"⚠️ Low confidence ({conf:.1f}% < {conf_threshold}%) — the model is unsure. "
            "Retake with a single leaf on a plain background in good light, "
            "and compare against the other top-5 predictions below. "
            "This is flagged in the PDF report."
        )

    # ---- Top 5 + Treatment + Prevention + Organic --------------------------
    html_block('<div class="section-label"><span class="num">4</span> Top-5 · Treatment · Prevention · Organic</div>')
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        rows = ""
        for i, p in enumerate(predictions):
            pct = p["score"] * 100
            first = " first" if i == 0 else ""
            std_note = (f" <small style='color:var(--soft);font-size:11px;'>(σ {p['tta_std']*100:.1f}%)</small>"
                        if use_tta and p.get("tta_std") is not None else "")
            rows += (
                f'<div class="bar-row{first}" style="--c:{color};--c2:{color}bb;">'
                f'<div class="top"><span>{esc(display_name(p["label"]))}{std_note}</span><span>{pct:.1f}%</span></div>'
                f'<div class="track"><div class="fill" style="width:{max(pct, 0.6):.1f}%;"></div></div></div>'
            )
        html_block(f'<div class="card"><h3>Top-5 predictions</h3>{rows}</div>')

    with c2:
        title = "🌽 Treatment" if info["crop"].startswith("Corn") else "💊 Treatment" if not healthy else "🌿 Care advice"
        html_block(
            f"""
<div class="treat" style="--c:{color};">
  <h3>{title}</h3>
  <p>{esc(info['treatment'])}</p>
</div>
"""
        )
        st.caption("General guidance only — confirm with a local agronomist before spraying.")

    # Prevention Tips
    if show_prevention and info.get("prevention"):
        tips_html = "".join(f"<li>{esc(t)}</li>" for t in info["prevention"])
        html_block(f"""
<div class="tips-block">
  <h3>🛡️ Prevention Tips</h3>
  <ul>{tips_html}</ul>
</div>
""")

    # Organic Options
    if show_organic and info.get("organic"):
        org_html = "".join(f"<li>{esc(o)}</li>" for o in info["organic"])
        html_block(f"""
<div class="tips-block">
  <h3>🌱 Organic &amp; Biological Options</h3>
  <ul>{org_html}</ul>
</div>
""")

    # ---- Grad-CAM ----------------------------------------------------------
    html_block('<div class="section-label"><span class="num">5</span> Grad-CAM explainability</div>')
    with st.spinner("Computing Grad-CAM heatmap..."):
        cam = grad_cam(model, image, top["index"])
    if cam is None:
        st.info(f"Grad-CAM could not be generated. ({st.session_state.get('gradcam_error', 'unknown error')})")
    else:
        crop, overlay, heat = cam
        g1, g2 = st.columns(2, gap="large")
        with g1:
            show_image(crop, "Model input (centre-cropped)")
        with g2:
            show_image(overlay, "Grad-CAM overlay")
        html_block(
            """
<div class="legend"></div>
<div class="legend-labels"><span>Low activation</span><span>High activation</span></div>
<p class="note"><b>Grad-CAM Explainability.</b> The heatmap highlights which leaf regions drove the prediction.
Warm colours (red/yellow) = high activation.</p>
"""
        )

    render_gate_details(check)

    # ---- Reports -----------------------------------------------------------
    html_block('<div class="section-label"><span class="num">6</span> Download Report</div>')

    dl1, dl2 = st.columns(2, gap="medium")

    # TXT report (always available)
    with dl1:
        report = io.StringIO()
        report.write("AI PLANT DISEASE DETECTOR — DIAGNOSIS REPORT\n")
        report.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        report.write(f"Owner: {OWNER}\n")
        report.write(f"Model: {MODEL_LABEL}\n")
        report.write("=" * 60 + "\n\n")
        if low_conf:
            report.write(f"⚠️  LOW CONFIDENCE WARNING: {conf:.1f}% < threshold {conf_threshold}%\n"
                         "    This diagnosis should be treated as uncertain.\n\n")
        report.write(f"PHOTO QUALITY\n")
        report.write(f"  Score : {qc['score']}/100\n")
        for iss in qc["issues"]:
            report.write(f"  Issue : {iss}\n")
        for w in qc["warnings"]:
            report.write(f"  Note  : {w}\n")
        report.write("\nDIAGNOSIS\n")
        report.write(f"  Crop        : {info['crop']}\n")
        report.write(f"  Diagnosis   : {info['disease']}\n")
        report.write(f"  Risk Level  : {info['risk']}\n")
        report.write(f"  Pathogen    : {info['pathogen']}\n")
        report.write(f"  Confidence  : {conf:.1f}%\n")
        if use_tta and tta_std is not None:
            report.write(f"  TTA Std Dev : {tta_std*100:.1f}% (lower = more stable)\n")
        report.write("\nTOP-5 PREDICTIONS\n")
        for p in predictions:
            std_note = f"  σ {p['tta_std']*100:.1f}%" if use_tta and p.get("tta_std") is not None else ""
            report.write(f"  {p['score']*100:.1f}%{std_note}  {display_name(p['label'])}\n")
        report.write(f"\nTREATMENT\n  {info['treatment']}\n")
        if info.get("prevention"):
            report.write("\nPREVENTION TIPS\n")
            for tip in info["prevention"]:
                report.write(f"  • {tip}\n")
        if info.get("organic"):
            report.write("\nORGANIC & BIOLOGICAL OPTIONS\n")
            for opt in info["organic"]:
                report.write(f"  • {opt}\n")
        report.write("\n" + "=" * 60 + "\n")
        report.write("Disclaimer: AI predictions are advisory only. Confirm with an agronomist.\n")

        st.download_button(
            "⬇️ Download report (.txt)",
            report.getvalue(),
            file_name=f"plant_diagnosis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        )

    # PDF report
    with dl2:
        with st.spinner("Preparing PDF..."):
            pdf_bytes = generate_pdf_report(
                info=info,
                conf=conf,
                predictions=predictions,
                qc=qc,
                low_conf=low_conf,
                tta_std=tta_std if use_tta else None,
                image=image,
            )
        if pdf_bytes:
            st.download_button(
                "⬇️ Download report (.pdf)",
                data=pdf_bytes,
                file_name=f"plant_diagnosis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )
        else:
            st.info("PDF download requires `reportlab`. Run: `pip install reportlab`")

render_footer()
