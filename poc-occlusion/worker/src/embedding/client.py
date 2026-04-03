from pathlib import Path
from typing import Any, Dict, Tuple, Union

import httpx


class ModelServiceClient:
    """Async client for the model serving API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    def _ensure_client(self):
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.timeout)

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None

    async def extract_features(
        self,
        image_data: bytes,
        model: str = "osnet",
        original_idx: int | None = None,
    ) -> Tuple[int | None, Dict[str, Any]]:
        self._ensure_client()
        files = {"image": ("image.jpg", image_data, "image/jpeg")}
        data = {"model": model}
        response = await self.client.post(
            f"{self.base_url}/embedding", files=files, data=data
        )
        if response.status_code == 200:
            return original_idx, response.json()
        else:
            raise Exception(
                f"Feature extraction failed: {response.status_code}: {response.text}"
            )

    async def extract_features_batch(
        self,
        images: list[bytes],
        model: str = "osnet",
    ) -> list[Dict[str, Any]]:
        self._ensure_client()
        files = [("images", ("image.jpg", img, "image/jpeg")) for img in images]
        data = {"model": model}
        response = await self.client.post(
            f"{self.base_url}/embedding/batch", files=files, data=data
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Batch feature extraction failed: {response.status_code}: {response.text}"
            )
