"""Tests for inference engine endpoints with mocked models."""
from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import app, batch_queue, registry


def _make_jpeg() -> bytes:
    """Create a minimal valid JPEG image."""
    img = Image.fromarray(np.zeros((64, 32, 3), dtype=np.uint8))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _mock_registry():
    """Patch registry and batch queue so endpoint tests stay lightweight."""
    # Set osnet to a truthy value so the is_ready property returns True
    registry.osnet = True
    with (
        patch.object(registry, "load", return_value=None),
        patch.object(registry, "extract_embedding", return_value=[0.1] * 512),
        patch.object(registry, "extract_embedding_batch", return_value=[[0.1] * 512, [0.2] * 512]),
        patch.object(registry, "preprocess_embedding", return_value=np.zeros((1, 3, 256, 128), dtype=np.float32)),
        patch.object(registry, "extract_embedding_from_tensors", return_value=[[0.1] * 512]),
        patch.object(registry, "compute_similarity", return_value=0.85),
        patch.object(
            registry,
            "classify_gender",
            return_value={
                "gender": "male",
                "confidence": 0.92,
                "probabilities": {"male": 0.92, "female": 0.08},
            },
        ),
        patch.object(batch_queue, "start", return_value=None),
        patch.object(batch_queue, "stop", new=AsyncMock(return_value=None)),
        patch.object(batch_queue, "enqueue", new=AsyncMock(return_value=[0.1] * 512)),
    ):
        yield
    registry.osnet = None


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_embedding_single(client):
    jpeg = _make_jpeg()
    r = client.post("/embedding", files={"image": ("img.jpg", jpeg, "image/jpeg")}, data={"model": "osnet"})
    assert r.status_code == 200
    body = r.json()
    assert "embedding" in body
    assert len(body["embedding"]) == 512
    assert body["shape"] == [512]


def test_embedding_batch(client):
    jpeg = _make_jpeg()
    files = [
        ("images", ("a.jpg", jpeg, "image/jpeg")),
        ("images", ("b.jpg", jpeg, "image/jpeg")),
    ]
    r = client.post("/embedding/batch", files=files, data={"model": "osnet"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["embeddings"]) == 2


def test_embedding_batch_fails_when_embedding_model_is_unavailable(client):
    registry.osnet = None
    registry.triton = None
    with patch.object(registry, "extract_embedding_batch", side_effect=RuntimeError("No ReID embedding model loaded")):
        jpeg = _make_jpeg()
        files = [("images", ("a.jpg", jpeg, "image/jpeg"))]
        r = client.post("/embedding/batch", files=files, data={"model": "osnet"})

    assert r.status_code == 503
    assert "No ReID embedding model loaded" in r.json()["detail"]


def test_similarity(client):
    jpeg = _make_jpeg()
    r = client.post(
        "/similarity",
        files={
            "image1": ("a.jpg", jpeg, "image/jpeg"),
            "image2": ("b.jpg", jpeg, "image/jpeg"),
        },
        data={"model": "osnet"},
    )
    assert r.status_code == 200
    assert r.json()["similarity"] == pytest.approx(0.85)


def test_gender_classify(client):
    jpeg = _make_jpeg()
    r = client.post("/gender/classify", files={"image": ("img.jpg", jpeg, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["gender"] == "male"
    assert body["confidence"] == pytest.approx(0.92)
    assert "probabilities" in body
