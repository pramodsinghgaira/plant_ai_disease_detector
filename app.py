import html
import io

import numpy as np
import streamlit as st
from PIL import Image
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification

from disease_info import lookup, display_name, LOW, MODERATE, HIGH, NONE

# ----------------------------------------------------------------------
# AI PLANT DISEASE DETECTOR
# Owner: Pramod Singh Gaira
# Free pretrained Hugging Face model + Grad-CAM explainability
# ----------------------------------------------------------------------

OWNER = "Pramod Singh Gaira"
MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
BASE_PROCESSOR_NAME = "google/mobilenet_v2_1.0_224"

st.set_page_config(page_title="AI Plant Disease Detector", page_icon="🌿", layout="wide")


# ======================================================================
# MODEL + GRAD-CAM
# ======================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    processor = AutoImageProcessor.from_pretrained(BASE_PROCESSOR_NAME, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return pipeline("image-classification", model=model, image_processor=processor)


def _last_conv_layer(model):
    """Last convolutional block of the backbone (the usual Grad-CAM target)."""
    import torch.nn as nn

    backbone = getattr(model, "mobilenet_v2", None)
    if backbone is not None and hasattr(backbone, "conv_1x1"):
        return backbone.conv_1x1
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def _jet(x: np.ndarray) -> np.ndarray:
    """Tiny jet-style colormap (blue -> cyan -> yellow -> red); x in [0,1]."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def grad_cam(classifier, image: Image.Image, label: str):
    """Return (overlay_image, heatmap_image) for the given class label, or None on failure."""
    try:
        import torch
        import torch.nn.functional as F

        model, processor = classifier.model, classifier.image_processor
        target_layer = _last_conv_layer(model)
        class_idx = model.config.label2id[label]

        # Squash to the model's input size so the heatmap maps onto the whole photo.
        small = image.resize((224, 224), Image.BICUBIC)
        pixel_values = processor(
            images=small, return_tensors="pt", do_resize=False, do_center_crop=False
        )["pixel_values"].to(model.device)

        store = {}

        def fwd_hook(_, __, out):
            store["act"] = out
            out.register_hook(lambda g: store.__setitem__("grad", g))

        handle = target_layer.register_forward_hook(fwd_hook)
        try:
            with torch.enable_grad():
                model.zero_grad()
                logits = model(pixel_values=pixel_values).logits
                logits[0, class_idx].backward()
        finally:
            handle.remove()

        act, grad = store["act"].detach(), store["grad"].detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.size[::-1], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        heat = _jet(cam)
        alpha = (0.25 + 0.55 * cam)[..., None]
        overlay = np.clip(base * (1 - alpha) + heat * alpha, 0, 1)
        return (
            Image.fromarray((overlay * 255).astype(np.uint8)),
            Image.fromarray((heat * 255).astype(np.uint8)),
        )
    except Exception as exc:  # explainability must never break the diagnosis
        st.session_state["gradcam_error"] = str(exc)
        return None


# ======================================================================
# HELPERS
# ======================================================================
RISK_STYLE = {
    NONE: ("#22c55e", "rgba(34,197,94,0.14)", "✅"),
    LOW: ("#84cc16", "rgba(132,204,22,0.14)", "🟢"),
    MODERATE: ("#f59e0b", "rgba(245,158,11,0.14)", "🟠"),
    HIGH: ("#ef4444", "rgba(239,68,68,0.14)", "🔴"),
}


def html_block(markup: str):
    """Render HTML safely (collapses newlines/indent so markdown never treats it as code)."""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def show_image(img, caption=None):
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:  # older Streamlit
        st.image(img, caption=caption, use_container_width=True)


def esc(s) -> str:
    return html.escape(str(s))


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

/* GRADCAM */
.legend{height:10px; border-radius:99px; margin-top:10px; background:linear-gradient(90deg,#0000ff,#00ffff,#ffff00,#ff0000);}
.legend-labels{display:flex; justify-content:space-between; font-size:11.5px; color:var(--soft); margin-top:5px;}
.note{color:var(--soft); font-size:13.5px; line-height:1.55; margin:6px 0 0;}

.empty{text-align:center; padding:44px 20px; border-radius:20px; border:1px dashed rgba(74,222,128,.28);
  background:rgba(255,255,255,.02); color:var(--soft); font-size:15px; animation:fadeUp .6s ease;}
.empty .big{font-size:38px; margin-bottom:10px;}
.footer{text-align:center; margin-top:44px; padding-top:20px; border-top:1px solid var(--border);
  color:rgba(232,245,233,.4); font-size:12.5px; line-height:1.8;}
.footer b{color:rgba(232,245,233,.75);}
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
    <p>Browse a leaf photo or capture one live with your webcam — get diagnosis, risk level, pathogen, treatment and an explainable heatmap.</p>
  </div>
  <div class="owner"><div class="av">PS</div><div><small>Owner</small><b>{esc(OWNER)}</b></div></div>
</div>
"""
)

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
        """
<div class="empty"><div class="big">🌱</div>
Browse a leaf image or capture one with your webcam to get started.<br>
Works for 38 crop-disease classes — apple, corn, grape, potato, tomato and more.</div>
"""
    )
else:
    with st.spinner("Loading AI model... (first run only)"):
        classifier = load_model()

    image = Image.open(image_source).convert("RGB")

    with st.spinner("Analyzing image..."):
        predictions = classifier(image, top_k=5)

    top = predictions[0]
    info = lookup(top["label"])
    conf = top["score"] * 100
    color, soft_bg, icon = RISK_STYLE[info["risk"]]
    healthy = info["kind"] == "Healthy"

    html_block('<div class="section-label"><span class="num">2</span> Diagnosis</div>')
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        show_image(image, caption)
    with right:
        html_block(
            f"""
<div class="verdict">
  <div class="glow" style="background:{color};"></div>
  <div class="ring" style="--p:{conf:.1f};--c:{color};"><span>{conf:.1f}%</span><small>conf.</small></div>
  <div class="verdict-main">
    <div class="crop">{esc(info['crop'])}</div>
    <h2>{esc(info['disease'])}</h2>
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
</div>
"""
        )

    if conf < 50:
        st.warning(
            "Low confidence — the model is unsure. Retake the photo with a single leaf, plain background and good light, "
            "and compare against the other top-5 predictions below."
        )

    # ---- Top 5 + treatment -------------------------------------------------
    html_block('<div class="section-label"><span class="num">3</span> Top-5 predictions &amp; treatment</div>')
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        rows = ""
        for i, p in enumerate(predictions):
            pct = p["score"] * 100
            first = " first" if i == 0 else ""
            rows += (
                f'<div class="bar-row{first}" style="--c:{color};--c2:{color}bb;">'
                f'<div class="top"><span>{esc(display_name(p["label"]))}</span><span>{pct:.1f}%</span></div>'
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
        st.caption("General guidance only — confirm with a local agronomist before spraying, and follow product labels.")

    # ---- Grad-CAM ----------------------------------------------------------
    html_block('<div class="section-label"><span class="num">4</span> Grad-CAM explainability</div>')
    with st.spinner("Computing Grad-CAM heatmap..."):
        cam = grad_cam(classifier, image, top["label"])
    if cam is None:
        st.info(f"Grad-CAM could not be generated for this image. ({st.session_state.get('gradcam_error', 'unknown error')})")
    else:
        overlay, heat = cam
        g1, g2 = st.columns(2, gap="large")
        with g1:
            show_image(image, "Original")
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

    # ---- Report ------------------------------------------------------------
    report = io.StringIO()
    report.write("AI PLANT DISEASE DETECTOR — REPORT\n")
    report.write(f"Owner: {OWNER}\n\n")
    report.write(f"Crop: {info['crop']}\nDiagnosis: {info['disease']}\nRisk Level: {info['risk']}\n")
    report.write(f"Pathogen: {info['pathogen']}\nConfidence: {conf:.1f}%\n\nTop-5 predictions:\n")
    for p in predictions:
        report.write(f"  - {display_name(p['label'])}: {p['score'] * 100:.1f}%\n")
    report.write(f"\nTreatment: {info['treatment']}\n")
    st.write("")
    st.download_button("⬇️ Download report (.txt)", report.getvalue(), file_name="plant_diagnosis_report.txt")

html_block(
    f"""
<div class="footer">
  Model: linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification (Hugging Face, free &amp; pretrained)<br>
  App by <b>{esc(OWNER)}</b> · AI predictions are advisory, not a substitute for expert diagnosis
</div>
"""
)
