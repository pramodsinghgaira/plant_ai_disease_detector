"""Pre-check that runs BEFORE the disease model.

The EfficientNet-B4 classifier is closed-set: it always answers with one of its 38
classes, even for a selfie or a mango leaf. So we first ask a general-purpose
vision-language model (CLIP, zero-shot, no training needed) two questions:

  Gate 1 - "Is this a plant leaf at all?"        -> if not: "upload a leaf image"
  Gate 2 - "Is it one of the 14 supported crops?" -> if not: list the supported crops

Only when both gates pass does the app run the disease prediction.

The thresholds below are starting values. Open "Pre-check details" in the app to see
the raw scores for any photo and adjust them if you see wrong accepts / rejects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from PIL import Image

CLIP_NAME = "openai/clip-vit-base-patch32"

# ----------------------------------------------------------------------
# Supported crops: (name shown to the user, noun used in the CLIP prompt)
# ----------------------------------------------------------------------
SUPPORTED_CROPS = [
    ("Apple", "apple tree"),
    ("Blueberry", "blueberry bush"),
    ("Cherry (including sour)", "cherry tree"),
    ("Corn (maize)", "corn plant"),
    ("Grape", "grape vine"),
    ("Orange", "orange tree"),
    ("Peach", "peach tree"),
    ("Pepper, bell", "bell pepper plant"),
    ("Potato", "potato plant"),
    ("Raspberry", "raspberry plant"),
    ("Soybean", "soybean plant"),
    ("Squash", "squash plant"),
    ("Strawberry", "strawberry plant"),
    ("Tomato", "tomato plant"),
]

# Common leaves that are NOT supported - lets CLIP say "this is something else"
# instead of being forced to pick the nearest supported crop.
UNSUPPORTED_CROPS = [
    "mango tree", "banana plant", "rice plant", "wheat plant", "sugarcane plant",
    "cotton plant", "rose bush", "neem tree", "basil plant", "mint plant",
    "hibiscus plant", "tea plant", "coffee plant", "maple tree", "oak tree",
    "fig tree", "palm tree", "money plant (pothos)", "lotus plant", "fern",
    "bamboo", "guava tree", "papaya tree", "sunflower plant",
]

# Gate 1 prompts
LEAF_PROMPTS = [
    "a close-up photo of a plant leaf",
    "a photo of a single green leaf on a plain background",
    "a photo of a diseased plant leaf with spots",
    "a photo of a leaf with brown lesions",
    "a photo of a leaf held in a hand",
    "a photo of a healthy green leaf",
    "a macro photo of a leaf with yellow patches",
]
NON_LEAF_PROMPTS = [
    "a photo of a person", "a selfie of a face", "a photo of a dog", "a photo of a cat",
    "a photo of a bird", "a photo of an animal", "a photo of a car", "a photo of a building",
    "a photo of a plate of food", "a photo of a fruit", "a photo of a vegetable",
    "a photo of a flower", "a photo of a whole tree", "a photo of a landscape",
    "a photo of a farm field", "a photo of the sky", "a screenshot of a phone or computer",
    "a scanned document with text", "a photo of a room interior", "a photo of furniture",
    "a cartoon or illustration", "a photo of a bottle or everyday object",
    "a photo of bare soil", "a photo of a grass lawn", "a plain blank wall",
    "a photo of a tree trunk or bark", "a photo of a plant pot",
]

# ----------------------------------------------------------------------
# Thresholds (tune using the "Pre-check details" panel)
# ----------------------------------------------------------------------
LEAF_MIN_PROB = 0.40          # Gate 1: total probability mass on the "leaf" prompts
UNSUPPORTED_MIN_PROB = 0.45   # Gate 2: reject only if an unsupported crop is both
                              #   >= this and ahead of every supported crop


@dataclass
class GateResult:
    is_leaf: bool
    supported: bool
    leaf_prob: float
    best_supported: str
    best_supported_prob: float
    best_unsupported: str
    best_unsupported_prob: float
    top_non_leaf: str = ""
    scores: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.is_leaf and self.supported


class LeafGate:
    """Zero-shot CLIP gate. Build once (cache it), call `.check(image)` per photo."""

    def __init__(self, device: torch.device | None = None):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(CLIP_NAME).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_NAME)

        crop_prompts = [f"a photo of a {noun} leaf" for _, noun in SUPPORTED_CROPS]
        unsup_prompts = [f"a photo of a {noun} leaf" for noun in UNSUPPORTED_CROPS]
        self.n_leaf, self.n_non = len(LEAF_PROMPTS), len(NON_LEAF_PROMPTS)
        self.n_sup, self.n_unsup = len(crop_prompts), len(unsup_prompts)

        self.prompts = LEAF_PROMPTS + NON_LEAF_PROMPTS + crop_prompts + unsup_prompts
        tok = self.processor.tokenizer(self.prompts, padding=True, return_tensors="pt")
        self.input_ids = tok["input_ids"].to(self.device)
        self.attention_mask = tok["attention_mask"].to(self.device)

    @torch.no_grad()
    def check(self, image: Image.Image) -> GateResult:
        pixel_values = self.processor.image_processor(
            images=image.convert("RGB"), return_tensors="pt"
        )["pixel_values"].to(self.device)

        out = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            pixel_values=pixel_values,
        )
        logits = out.logits_per_image[0].float().cpu()  # already scaled by CLIP's logit_scale

        a = self.n_leaf + self.n_non
        gate1 = torch.softmax(logits[:a], dim=0)
        gate2 = torch.softmax(logits[a:], dim=0)

        leaf_prob = float(gate1[: self.n_leaf].sum())
        non_leaf_probs = gate1[self.n_leaf:]
        top_non_leaf = NON_LEAF_PROMPTS[int(non_leaf_probs.argmax())]

        sup_probs, unsup_probs = gate2[: self.n_sup], gate2[self.n_sup:]
        i_sup, i_unsup = int(sup_probs.argmax()), int(unsup_probs.argmax())
        best_sup_p, best_unsup_p = float(sup_probs[i_sup]), float(unsup_probs[i_unsup])

        is_leaf = leaf_prob >= LEAF_MIN_PROB
        unsupported = best_unsup_p >= UNSUPPORTED_MIN_PROB and best_unsup_p > best_sup_p

        return GateResult(
            is_leaf=is_leaf,
            supported=not unsupported,
            leaf_prob=leaf_prob,
            best_supported=SUPPORTED_CROPS[i_sup][0],
            best_supported_prob=best_sup_p,
            best_unsupported=UNSUPPORTED_CROPS[i_unsup],
            best_unsupported_prob=best_unsup_p,
            top_non_leaf=top_non_leaf.replace("a photo of ", "").replace("a ", "", 1),
            scores={
                "leaf_prob": leaf_prob,
                "top_non_leaf_prob": float(non_leaf_probs.max()),
                "supported_by_crop": {
                    name: float(p) for (name, _), p in zip(SUPPORTED_CROPS, sup_probs.tolist())
                },
            },
        )
