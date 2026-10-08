"""EfficientNet-B4 plant-disease classifier.

Weights : Khawajaa/plant-disease-detector (Hugging Face, MIT) -> best_model.pth
Code    : the `EfficientNetB4Classifier` class below is the architecture from
          github.com/khawaja1447/plant-disease-detector (src/model.py, MIT),
          trimmed to inference only. The layer names must stay exactly as they
          are, otherwise the checkpoint's state dict will not load.

Input   : 224x224 RGB, ImageNet-normalised. Off-square photos are scaled and
          centre-cropped (never squashed), because every training image is square.

TTA     : Test-Time Augmentation averages predictions over 6 augmented views
          (original + horizontal flip + 3 random crops + colour jitter) for
          improved robustness on imperfect real-world photos.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps, ImageEnhance
from torchvision import models

MODEL_REPO = "Khawajaa/plant-disease-detector"
MODEL_FILE = "best_model.pth"
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Canonical 38 PlantVillage classes, in the order the checkpoint was trained on
# (alphabetical folder order). Index i of the logits == CLASSES[i].
CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


class EfficientNetB4Classifier(nn.Module):
    """torchvision EfficientNet-B4 backbone with a custom two-layer head."""

    def __init__(self, num_classes: int = 38, dropout: float = 0.4):
        super().__init__()
        backbone = models.efficientnet_b4(weights=None)  # weights come from the checkpoint
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1792, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    @property
    def grad_cam_target_layer(self) -> nn.Module:
        """Last conv stage - the standard Grad-CAM target for EfficientNet."""
        return self.features[-1]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_classifier(device: torch.device | None = None) -> EfficientNetB4Classifier:
    """Download (first run only) and load the EfficientNet-B4 checkpoint."""
    from huggingface_hub import hf_hub_download

    device = device or get_device()
    path = hf_hub_download(MODEL_REPO, MODEL_FILE)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = EfficientNetB4Classifier(num_classes=len(CLASSES))
    model.load_state_dict(state)
    return model.to(device).eval()


# ----------------------------------------------------------------------
# Pre-processing + prediction
# ----------------------------------------------------------------------
def open_image(file) -> Image.Image:
    """Open an upload/camera file as RGB, honouring the phone's EXIF rotation."""
    img = Image.open(file)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def prepare_input_image(image: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    """Scale the short side and centre-crop to a square (no distortion)."""
    return ImageOps.fit(image, (size, size), method=Image.BICUBIC)


def to_tensor(square_img: Image.Image) -> torch.Tensor:
    """224x224 RGB PIL image -> normalised (1, 3, 224, 224) float tensor."""
    arr = np.asarray(square_img.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


# ----------------------------------------------------------------------
# Photo Quality Check
# ----------------------------------------------------------------------
def check_photo_quality(image: Image.Image) -> dict:
    """Assess photo quality for meaningful leaf diagnosis.

    Returns a dict with:
        - score (0-100): overall quality score
        - passed (bool): True if image is usable
        - issues (list[str]): human-readable issues found
        - warnings (list[str]): non-blocking suggestions
    """
    import math

    issues = []
    warnings = []

    w, h = image.size

    # --- Resolution check ---
    min_dim = min(w, h)
    if min_dim < 100:
        issues.append(f"Image is very small ({w}×{h} px) — diagnosis may be unreliable")
    elif min_dim < 200:
        warnings.append(f"Low resolution ({w}×{h} px) — a higher-res photo will give better results")

    # --- Brightness check (mean luminance) ---
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    mean_lum = float(arr.mean())
    if mean_lum < 30:
        issues.append("Image is very dark — retake in better lighting")
    elif mean_lum < 60:
        warnings.append("Image is somewhat dark — diagnosis may be affected")
    elif mean_lum > 230:
        issues.append("Image is overexposed (too bright) — retake avoiding direct flash or sunlight")
    elif mean_lum > 200:
        warnings.append("Image is quite bright — ensure leaf detail is visible")

    # --- Blur check (Laplacian variance) ---
    # Compute a simple Laplacian variance as blur metric
    arr_norm = arr / 255.0
    # Simple finite-difference Laplacian
    laplacian = (
        np.roll(arr_norm, 1, axis=0) + np.roll(arr_norm, -1, axis=0) +
        np.roll(arr_norm, 1, axis=1) + np.roll(arr_norm, -1, axis=1) -
        4 * arr_norm
    )
    blur_score = float(laplacian.var())
    if blur_score < 0.0003:
        issues.append("Image appears blurry — hold the camera steady and focus on the leaf")
    elif blur_score < 0.001:
        warnings.append("Image may be slightly blurry — a sharper photo improves accuracy")

    # --- Colour saturation check ---
    hsv_arr = np.asarray(image.convert("HSV"), dtype=np.float32)
    mean_sat = float(hsv_arr[:, :, 1].mean())
    if mean_sat < 15:
        warnings.append("Image appears nearly greyscale — ensure colour is enabled on your camera")

    # --- Aspect ratio check ---
    ratio = max(w, h) / max(min(w, h), 1)
    if ratio > 4:
        warnings.append("Image is very elongated — a squarer crop centred on the leaf works best")

    # --- Overall quality score ---
    score = 100
    score -= len(issues) * 25
    score -= len(warnings) * 8
    # Penalise extreme brightness
    score -= max(0, (abs(mean_lum - 128) - 50) * 0.3)
    # Penalise blur
    if blur_score < 0.001:
        score -= int((0.001 - blur_score) * 20000)
    score = max(0, min(100, int(score)))

    passed = len(issues) == 0

    return {
        "score": score,
        "passed": passed,
        "issues": issues,
        "warnings": warnings,
        "mean_brightness": round(mean_lum, 1),
        "blur_score": round(blur_score, 6),
    }


# ----------------------------------------------------------------------
# Test-Time Augmentation (TTA)
# ----------------------------------------------------------------------
def _tta_views(image: Image.Image, size: int = IMG_SIZE) -> list[torch.Tensor]:
    """Generate multiple augmented views of the image for TTA.

    6 views:
      0 - centre crop (standard)
      1 - horizontal flip of centre crop
      2 - slight brightness boost
      3 - slight brightness reduction
      4 - top-left crop (slight shift)
      5 - bottom-right crop (slight shift)
    """
    views = []

    # View 0: standard centre crop
    crop = ImageOps.fit(image, (size, size), method=Image.BICUBIC)
    views.append(to_tensor(crop))

    # View 1: horizontal flip
    views.append(to_tensor(crop.transpose(Image.FLIP_LEFT_RIGHT)))

    # View 2: slightly brighter (+15%)
    bright = ImageEnhance.Brightness(crop).enhance(1.15)
    views.append(to_tensor(bright))

    # View 3: slightly darker (-15%)
    dark = ImageEnhance.Brightness(crop).enhance(0.85)
    views.append(to_tensor(dark))

    # View 4: slightly different crop — scale up 10% then crop
    w, h = image.size
    pad_w, pad_h = int(w * 0.05), int(h * 0.05)
    if w > 2 * pad_w and h > 2 * pad_h:
        cropped_tl = image.crop((pad_w, pad_h, w - pad_w, h - pad_h))
        views.append(to_tensor(ImageOps.fit(cropped_tl, (size, size), method=Image.BICUBIC)))
    else:
        views.append(to_tensor(crop))  # fallback

    # View 5: contrast enhancement
    contrast = ImageEnhance.Contrast(crop).enhance(1.1)
    views.append(to_tensor(contrast))

    return views


def predict(
    model: EfficientNetB4Classifier,
    image: Image.Image,
    top_k: int = 5,
    use_tta: bool = True,
) -> list[dict]:
    """Return the top-k predictions as [{label, index, score, tta_std}], best first.

    When use_tta=True (default), predictions are averaged over 6 augmented views.
    The returned dicts also include `tta_std` — per-class standard deviation across
    views, which reflects prediction consistency (lower = more confident/stable).
    """
    device = next(model.parameters()).device

    if use_tta:
        views = _tta_views(image)
        all_probs = []
        with torch.no_grad():
            for v in views:
                x = v.to(device)
                probs = torch.softmax(model(x), dim=1)[0].cpu()
                all_probs.append(probs)
        stacked = torch.stack(all_probs, dim=0)           # (n_views, n_classes)
        mean_probs = stacked.mean(dim=0)                  # (n_classes,)
        std_probs  = stacked.std(dim=0)                   # (n_classes,)
        scores, idx = mean_probs.topk(min(top_k, len(CLASSES)))
        return [
            {
                "label":   CLASSES[i],
                "index":   int(i),
                "score":   float(s),
                "tta_std": float(std_probs[i]),
            }
            for s, i in zip(scores.tolist(), idx.tolist())
        ]
    else:
        x = to_tensor(prepare_input_image(image)).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu()
        scores, idx = probs.topk(min(top_k, len(CLASSES)))
        return [
            {"label": CLASSES[i], "index": int(i), "score": float(s), "tta_std": None}
            for s, i in zip(scores.tolist(), idx.tolist())
        ]
