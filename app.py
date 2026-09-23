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
        return "HEALTHY", "🟢", "#2e7d32"
    return "DISEASE DETECTED", "🔴", "#c62828"


def clean_label(label: str) -> str:
    return label.replace("___", " - ").replace("_", " ").strip()


# ---------------- HEADER ----------------
st.markdown(
    """
    <div style="background-color:#1b5e20;padding:18px;border-radius:10px;text-align:center;">
        <h1 style="color:white;margin:0;">🌿 AI PLANT DISEASE DETECTOR</h1>
        <p style="color:#c8e6c9;margin:4px 0 0 0;">Working Model — File Upload &amp; Webcam Edition</p>
        <p style="color:#e8f5e9;margin:2px 0 0 0;font-size:14px;">by Pramod Singh Gaira</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

st.markdown("### 📷 Choose input method")
input_mode = st.radio(
    "How do you want to provide the leaf image?",
    ["📤 Upload a file", "📷 Use webcam"],
    horizontal=True,
    label_visibility="collapsed",
)

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

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption=caption, use_container_width=True)

    with st.spinner("Analyzing image..."):
        predictions = classifier(image, top_k=3)

    top = predictions[0]
    status_text, status_emoji, status_color = get_status(top["label"])

    with col2:
        st.markdown("#### Result")
        st.markdown(
            f"""
            <div style="border:2px solid {status_color};border-radius:10px;padding:14px;">
                <p style="margin:0;font-size:14px;color:gray;">Prediction</p>
                <p style="margin:0 0 8px 0;font-size:18px;font-weight:bold;color:{status_color};">
                    {clean_label(top['label'])}
                </p>
                <p style="margin:0;font-size:14px;color:gray;">Confidence</p>
                <p style="margin:0 0 8px 0;font-size:16px;font-weight:bold;">{top['score']*100:.2f}%</p>
                <p style="margin:0;font-size:14px;color:gray;">Status</p>
                <p style="margin:0;font-size:18px;font-weight:bold;color:{status_color};">
                    {status_emoji} {status_text}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("#### 🔎 Top 3 Predictions")
    for pred in predictions:
        st.progress(min(pred["score"], 1.0), text=f"{clean_label(pred['label'])} — {pred['score']*100:.2f}%")

    st.write("")
    if "healthy" in top["label"].lower():
        st.success("✅ Recommendation: Plant looks healthy. Continue regular care.")
    else:
        st.error("⚠️ Recommendation: Disease detected — isolate the plant, remove affected leaves, and consult a local expert for treatment.")

else:
    st.info("Upload a leaf image or use your webcam above to get started. No Arduino hardware needed — this version runs entirely on file upload / webcam + AI.")

st.markdown("---")
st.caption("Model: linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification (Hugging Face, free & pretrained) • App by Pramod Singh Gaira")
