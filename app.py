import streamlit as st
from PIL import Image
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification

# ----------------------------------------------------------------------
# AI PLANT DISEASE DETECTOR
# by Pramod Singh Gaira
# Uses a free pretrained model from Hugging Face (no training required)
# ----------------------------------------------------------------------

st.set_page_config(page_title="AI Plant Disease Detector", page_icon="🌿", layout="centered")

MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
# This checkpoint's repo lacks a modern preprocessor_config.json, so we load the
# image processor from the base model it was fine-tuned from instead of letting
# AutoImageProcessor guess it from MODEL_NAME.
BASE_PROCESSOR_NAME = "google/mobilenet_v2_1.0_224"


@st.cache_resource(show_spinner=False)
def load_model():
    processor = AutoImageProcessor.from_pretrained(BASE_PROCESSOR_NAME, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    return pipeline("image-classification", model=model, image_processor=processor)


def get_status(label: str):
    """Map a raw model label to Healthy / Warning / Disease + color."""
    label_lower = label.lower()
    if "healthy" in label_lower:
        return "HEALTHY", "#22c55e", "rgba(34,197,94,0.12)"
    return "DISEASE DETECTED", "#ef4444", "rgba(239,68,68,0.12)"


def clean_label(label: str) -> str:
    return label.replace("___", " - ").replace("_", " ").strip()


# ======================================================================
# GLOBAL STYLING — custom fonts, gradients, glassmorphism, animations
# ======================================================================
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

    <style>
    :root{
        --bg-1:#06120b;
        --bg-2:#0c1f14;
        --accent:#22c55e;
        --accent-2:#4ade80;
        --card-bg:rgba(255,255,255,0.045);
        --card-border:rgba(255,255,255,0.09);
        --text-soft:rgba(232,245,233,0.65);
    }

    html, body, [class*="css"]{
        font-family:'Inter', sans-serif;
    }

    /* animated aurora background */
    .stApp{
        background:
            radial-gradient(circle at 15% 15%, rgba(34,197,94,0.20), transparent 45%),
            radial-gradient(circle at 85% 20%, rgba(74,222,128,0.14), transparent 40%),
            radial-gradient(circle at 50% 100%, rgba(16,185,129,0.12), transparent 50%),
            linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 60%, #081a10 100%);
        background-attachment:fixed;
    }

    /* hide default streamlit chrome */
    #MainMenu, footer, header{visibility:hidden;}
    .block-container{padding-top:2.2rem; padding-bottom:3rem; max-width:780px;}

    /* ---------- HERO ---------- */
    .hero{
        position:relative;
        padding:40px 30px 34px 30px;
        border-radius:24px;
        text-align:center;
        overflow:hidden;
        background:linear-gradient(135deg, rgba(34,197,94,0.20), rgba(16,120,80,0.08));
        border:1px solid var(--card-border);
        box-shadow:0 20px 60px -20px rgba(0,0,0,0.6);
        animation:fadeDown 0.7s ease;
    }
    .hero::before{
        content:"";
        position:absolute; inset:0;
        background:radial-gradient(circle at 30% -10%, rgba(74,222,128,0.35), transparent 55%);
        pointer-events:none;
    }
    .hero-badge{
        display:inline-block;
        font-size:12px;
        letter-spacing:1.5px;
        font-weight:600;
        color:var(--accent-2);
        background:rgba(34,197,94,0.12);
        border:1px solid rgba(74,222,128,0.35);
        padding:5px 14px;
        border-radius:999px;
        margin-bottom:16px;
        text-transform:uppercase;
    }
    .hero h1{
        font-family:'Space Grotesk', sans-serif;
        font-size:clamp(28px, 5vw, 40px);
        font-weight:700;
        margin:0;
        color:#f2fbf3;
        letter-spacing:-0.5px;
    }
    .hero h1 span{
        background:linear-gradient(90deg, var(--accent-2), #a7f3d0);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
    }
    .hero p{
        color:var(--text-soft);
        margin:10px 0 0 0;
        font-size:15.5px;
    }
    .hero .author{
        margin-top:14px;
        font-size:13px;
        color:rgba(232,245,233,0.4);
        letter-spacing:0.3px;
    }

    @keyframes fadeDown{
        from{opacity:0; transform:translateY(-14px);}
        to{opacity:1; transform:translateY(0);}
    }
    @keyframes fadeUp{
        from{opacity:0; transform:translateY(14px);}
        to{opacity:1; transform:translateY(0);}
    }

    /* ---------- SECTION LABELS ---------- */
    .section-label{
        display:flex; align-items:center; gap:10px;
        margin:34px 0 14px 0;
        font-family:'Space Grotesk', sans-serif;
        font-size:15px;
        font-weight:600;
        color:#e8f5e9;
        animation:fadeUp 0.6s ease;
    }
    .section-label .num{
        display:flex; align-items:center; justify-content:center;
        width:24px; height:24px; border-radius:8px;
        background:linear-gradient(135deg, var(--accent), #16a34a);
        color:#06120b; font-size:12px; font-weight:700;
    }

    /* ---------- GLASS CARD WRAPPER ---------- */
    .glass-card{
        background:var(--card-bg);
        border:1px solid var(--card-border);
        border-radius:18px;
        padding:22px;
        backdrop-filter:blur(14px);
        animation:fadeUp 0.6s ease;
    }

    /* ---------- streamlit widget skinning ---------- */
    div[data-testid="stRadio"] > label{display:none;}
    div[data-testid="stRadio"] div[role="radiogroup"]{
        display:flex; gap:10px; flex-wrap:wrap;
    }
    div[data-testid="stRadio"] label{
        background:rgba(255,255,255,0.05);
        border:1px solid var(--card-border);
        padding:9px 16px;
        border-radius:12px;
        transition:all 0.2s ease;
        cursor:pointer;
    }
    div[data-testid="stRadio"] label:hover{
        border-color:var(--accent);
        background:rgba(34,197,94,0.08);
    }

    [data-testid="stFileUploaderDropzone"]{
        background:rgba(255,255,255,0.03) !important;
        border:1.5px dashed rgba(74,222,128,0.35) !important;
        border-radius:16px !important;
        transition:all 0.25s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover{
        border-color:var(--accent-2) !important;
        background:rgba(34,197,94,0.06) !important;
    }

    button[kind="secondary"], .stButton>button, [data-testid="stCameraInput"] button{
        border-radius:12px !important;
        border:1px solid rgba(74,222,128,0.4) !important;
        background:linear-gradient(135deg, rgba(34,197,94,0.18), rgba(34,197,94,0.06)) !important;
        color:#eafff0 !important;
        transition:all 0.2s ease;
    }
    button[kind="secondary"]:hover, .stButton>button:hover{
        border-color:var(--accent-2) !important;
        box-shadow:0 6px 20px -6px rgba(34,197,94,0.5);
        transform:translateY(-1px);
    }

    img{
        border-radius:16px !important;
        border:1px solid var(--card-border);
    }

    div[data-testid="stProgress"] > div > div{
        background:linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
        border-radius:8px;
    }
    div[data-testid="stProgress"]{
        background:rgba(255,255,255,0.05);
        border-radius:8px;
        padding:2px;
    }

    /* ---------- result card ---------- */
    .result-card{
        border-radius:18px;
        padding:20px 22px;
        background:var(--card-bg);
        border:1px solid var(--card-border);
        backdrop-filter:blur(14px);
        animation:fadeUp 0.5s ease;
        position:relative;
        overflow:hidden;
    }
    .result-card .glow{
        position:absolute; top:-40px; right:-40px;
        width:140px; height:140px; border-radius:50%;
        filter:blur(40px); opacity:0.55;
    }
    .result-label{
        font-size:12px; text-transform:uppercase; letter-spacing:1.2px;
        color:var(--text-soft); margin:0 0 4px 0;
    }
    .result-value{
        font-family:'Space Grotesk', sans-serif;
        font-size:20px; font-weight:700; margin:0 0 14px 0;
    }
    .status-pill{
        display:inline-flex; align-items:center; gap:8px;
        padding:8px 16px; border-radius:999px;
        font-weight:700; font-size:14px;
        margin-top:4px;
    }
    .dot{width:9px; height:9px; border-radius:50%; display:inline-block;}

    /* ---------- callout / recommendation ---------- */
    .callout{
        border-radius:16px;
        padding:16px 18px;
        border:1px solid;
        display:flex; gap:12px; align-items:flex-start;
        animation:fadeUp 0.6s ease;
        font-size:14.5px; line-height:1.5;
    }
    .callout .emoji{font-size:20px; line-height:1;}

    .footer-note{
        text-align:center;
        margin-top:40px;
        padding-top:20px;
        border-top:1px solid var(--card-border);
        color:rgba(232,245,233,0.35);
        font-size:12.5px;
    }

    .empty-state{
        text-align:center;
        padding:38px 20px;
        border-radius:18px;
        border:1px dashed rgba(74,222,128,0.25);
        background:rgba(255,255,255,0.02);
        color:var(--text-soft);
        font-size:14.5px;
        animation:fadeUp 0.6s ease;
    }
    .empty-state .big-emoji{font-size:34px; margin-bottom:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- HERO ----------------
st.markdown(
    """
    <div class="hero">
        <span class="hero-badge">🌿 Computer Vision · Plant Health</span>
        <h1>AI Plant <span>Disease Detector</span></h1>
        <p>Upload a leaf photo or use your webcam — get an instant AI diagnosis.</p>
        <div class="author">Built by Pramod Singh Gaira</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- INPUT METHOD ----------------
st.markdown('<div class="section-label"><span class="num">1</span> Choose input method</div>', unsafe_allow_html=True)

input_mode = st.radio(
    "How do you want to provide the leaf image?",
    ["📤 Upload a file", "📷 Use webcam"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown('<div class="section-label"><span class="num">2</span> Provide a leaf image</div>', unsafe_allow_html=True)

image_source = None
caption = "Leaf Image"

if input_mode == "📤 Upload a file":
    uploaded_file = st.file_uploader("Choose a leaf image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_source = uploaded_file
        caption = "Uploaded Leaf"
else:
    camera_file = st.camera_input("Point the webcam at the leaf and capture")
    if camera_file is not None:
        image_source = camera_file
        caption = "Captured Leaf"

if image_source is not None:
    with st.spinner("Loading AI model... (first run only)"):
        classifier = load_model()

if image_source is not None:
    image = Image.open(image_source).convert("RGB")

    st.markdown('<div class="section-label"><span class="num">3</span> Diagnosis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption=caption, use_container_width=True)

    with st.spinner("Analyzing image..."):
        predictions = classifier(image, top_k=3)

    top = predictions[0]
    status_text, status_color, status_glow = get_status(top["label"])

    with col2:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="glow" style="background:{status_color};"></div>
                <p class="result-label">Prediction</p>
                <p class="result-value" style="color:{status_color};">{clean_label(top['label'])}</p>
                <p class="result-label">Confidence</p>
                <p class="result-value" style="color:#eafff0;">{top['score']*100:.2f}%</p>
                <span class="status-pill" style="background:{status_glow}; color:{status_color}; border:1px solid {status_color}55;">
                    <span class="dot" style="background:{status_color};"></span>{status_text}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown('<div class="section-label"><span class="num">4</span> Top 3 predictions</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        for pred in predictions:
            st.progress(min(pred["score"], 1.0), text=f"{clean_label(pred['label'])} — {pred['score']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    if "healthy" in top["label"].lower():
        st.markdown(
            """
            <div class="callout" style="background:rgba(34,197,94,0.08); border-color:rgba(34,197,94,0.35); color:#bbf7d0;">
                <span class="emoji">✅</span>
                <div><b>Recommendation:</b> Plant looks healthy. Continue regular care and monitoring.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="callout" style="background:rgba(239,68,68,0.08); border-color:rgba(239,68,68,0.35); color:#fecaca;">
                <span class="emoji">⚠️</span>
                <div><b>Recommendation:</b> Disease detected — isolate the plant, remove affected leaves, and consult a local expert for treatment.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    st.markdown(
        """
        <div class="empty-state">
            <div class="big-emoji">🌱</div>
            Upload a leaf image or use your webcam above to get started.<br>
            No Arduino hardware needed — this version runs entirely on file upload / webcam + AI.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="footer-note">
        Model: linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification (Hugging Face, free &amp; pretrained)
        &nbsp;•&nbsp; App by Pramod Singh Gaira
    </div>
    """,
    unsafe_allow_html=True,
)
