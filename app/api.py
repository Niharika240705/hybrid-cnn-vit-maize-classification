import io
import os
import sys
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import HybridCNNViTModel
from src.device import select_device


CONFIG_PATH = Path(os.getenv("MAIZE_CONFIG", PROJECT_ROOT / "src" / "config.yaml"))
CHECKPOINT_PATH = Path(
    os.getenv(
        "MAIZE_CHECKPOINT",
        PROJECT_ROOT / "checkpoints" / "phase2" / "variety_pretrained_split_stratified" / "model_best.pth",
    )
)

app = FastAPI(title="Maize Seed Classifier API", version="1.0.0")
_model = None
_device = None
_preprocess = None


def _load_model():
    global _model, _device, _preprocess
    if _model is not None:
        return _model
    if not CHECKPOINT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint not found: {CHECKPOINT_PATH}. Train Phase 2 first.",
        )

    with CONFIG_PATH.open("r") as config_file:
        config = yaml.safe_load(config_file)

    _device = select_device(config.get("device"))
    _model = HybridCNNViTModel(
        cnn_backbone=config["cnn_backbone"],
        vit_backbone=config["vit_backbone"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_classes=2,
        pretrained=False,
    )
    _model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=_device))
    _model.to(_device).eval()
    _preprocess = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint_available": CHECKPOINT_PATH.exists()}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")

    model = _load_model()
    try:
        image_bytes = await image.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Invalid image: {error}") from error

    tensor = _preprocess(input_image).unsqueeze(0).to(_device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    class_index = int(torch.argmax(probabilities).item())
    labels = ["Bad", "Good"]

    return {"class": labels[class_index], "confidence": float(probabilities[class_index].item())}