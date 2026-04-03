import io
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from torchvision import transforms

from src.config import ModelServingSettings
from src.models.osnet import osnet_x1_0

settings = ModelServingSettings()

# Global model reference
model = None
transform = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, transform
    print(f"[ModelServing] Loading {settings.model_name} model...")
    model = osnet_x1_0(
        pretrained=True,
        gem_p=settings.gem_p,
        weights_dir=settings.weights_dir,
    )
    model.eval()
    print(f"[ModelServing] Model loaded on {next(model.parameters()).device}")

    transform = transforms.Compose([
        transforms.Resize((settings.image_height, settings.image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Warmup
    dummy = torch.randn(1, 3, settings.image_height, settings.image_width).to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        model(dummy)
    print("[ModelServing] Warmup complete")

    yield

    print("[ModelServing] Shutting down")


app = FastAPI(title="ReID Model Serving (PoC)", lifespan=lifespan)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.model_name}


@app.post("/embedding")
async def extract_embedding(
    image: UploadFile = File(...),
    model_name: str = Form(default="osnet"),
):
    image_bytes = await image.read()
    tensor = preprocess_image(image_bytes).to(next(model.parameters()).device)

    with torch.no_grad():
        embedding = model(tensor)

    embedding_np = embedding.cpu().numpy().flatten()
    # L2 normalize
    norm = np.linalg.norm(embedding_np)
    if norm > 1e-8:
        embedding_np = embedding_np / norm

    return {
        "embedding": embedding_np.tolist(),
        "shape": list(embedding_np.shape),
        "model": model_name,
    }


@app.post("/embedding/batch")
async def extract_embedding_batch(
    images: list[UploadFile] = File(...),
    model_name: str = Form(default="osnet"),
):
    tensors = []
    for img_file in images:
        image_bytes = await img_file.read()
        tensors.append(preprocess_image(image_bytes))

    batch = torch.cat(tensors, dim=0).to(next(model.parameters()).device)

    with torch.no_grad():
        embeddings = model(batch)

    results = []
    for emb in embeddings:
        emb_np = emb.cpu().numpy().flatten()
        norm = np.linalg.norm(emb_np)
        if norm > 1e-8:
            emb_np = emb_np / norm
        results.append({
            "embedding": emb_np.tolist(),
            "shape": list(emb_np.shape),
            "model": model_name,
        })

    return results
