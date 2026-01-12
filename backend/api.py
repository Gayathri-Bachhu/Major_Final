from __future__ import annotations

import base64
import io
from typing import List

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


class PredictRequest(BaseModel):
    data_url: str
    topk: int = 5


class Prediction(BaseModel):
    label: str
    prob: float


class PredictResponse(BaseModel):
    preds: List[Prediction]


app = FastAPI(title="Adversarial Demo ImageNet Inference API")

# Allow local Vite dev server (http://localhost:5173) and others while developing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load pretrained ImageNet model once at startup.
_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
_MODEL = resnet18(weights=_WEIGHTS)
_MODEL.eval()
_PREPROCESS = _WEIGHTS.transforms()
_CATEGORIES = _WEIGHTS.meta["categories"]


def _data_url_to_pil(data_url: str) -> Image.Image:
    """Decode a `data:image/...;base64,...` string into a RGB PIL image."""

    try:
        header, b64 = data_url.split(",", 1)
    except ValueError as exc:  # pragma: no cover - simple input validation
        raise ValueError("Invalid data URL (missing comma separator)") from exc

    img_bytes = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img


@app.post("/predict-imagenet", response_model=PredictResponse)
def predict_imagenet(req: PredictRequest) -> PredictResponse:
    """Run ImageNet classification on an uploaded image.

    Expects a browser-style data URL (e.g. `data:image/png;base64,...`).
    Returns Top-K predictions from a pretrained ResNet-18 (ImageNet-1k).
    """

    img = _data_url_to_pil(req.data_url)
    x = _PREPROCESS(img).unsqueeze(0)

    with torch.no_grad():
        logits = _MODEL(x)
        probs = torch.softmax(logits, dim=1)[0]

    topk = int(max(1, min(req.topk, probs.numel())))
    top_probs, top_idxs = torch.topk(probs, k=topk)

    preds: list[Prediction] = []
    for p, idx in zip(top_probs.tolist(), top_idxs.tolist()):
        label = _CATEGORIES[int(idx)] if 0 <= int(idx) < len(_CATEGORIES) else str(idx)
        preds.append(Prediction(label=label, prob=float(p)))

    return PredictResponse(preds=preds)