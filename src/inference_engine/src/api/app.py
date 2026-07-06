from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List

import structlog
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.core.config import settings
from src.services.batch_queue import EmbeddingBatchQueue
from src.services.model_registry import ModelRegistry

log = structlog.get_logger()

registry = ModelRegistry()
batch_queue = EmbeddingBatchQueue(
    registry,
    max_batch_size=settings.max_batch_size,
    timeout_ms=settings.batch_timeout_ms,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load()
    batch_queue.start()
    yield
    await batch_queue.stop()


app = FastAPI(title=settings.service_name, lifespan=lifespan)


# health

@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": settings.service_name}


@app.get("/readyz")
def readyz():
    ready = registry.is_ready
    return {
        "status": "ready" if ready else "not_ready",
        "service": settings.service_name,
        "device": str(registry.device),
    }


# embedding endpoints

@app.post("/embedding")
async def embedding(
    image: UploadFile = File(...),
    model: str = Form("osnet"),
):
    """Single-image embedding — transparently batched with concurrent requests."""
    data = await image.read()
    try:
        features = await batch_queue.enqueue(data, model=model)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"embedding": features, "shape": [len(features)]}


@app.post("/embedding/batch")
async def embedding_batch(
    images: List[UploadFile] = File(...),
    model: str = Form("osnet"),
):
    """Explicit multi-image batch embedding extraction."""
    blobs = [await img.read() for img in images]
    try:
        features_list = registry.extract_embedding_batch(blobs, model=model)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"embeddings": features_list, "count": len(features_list)}


# similarity

@app.post("/similarity")
async def similarity(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    model: str = Form("osnet"),
):
    data1, data2 = await image1.read(), await image2.read()
    try:
        sim = registry.compute_similarity(data1, data2, model=model)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"similarity": sim, "model_used": model}


# gender classification

@app.post("/gender/classify")
async def gender_classify(image: UploadFile = File(...)):
    data = await image.read()
    try:
        result = registry.classify_gender(data)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return result


# multri attributes classification

@app.post("/attributes/classify")
async def attributes_classify(image: UploadFile = File(...)):
    
    data = await image.read()
    try:
        result = registry.classify_attributes(data)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return result
