"""Loads and holds all inference models + their preprocessing transforms."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import structlog
import torch
from PIL import Image
from torchvision import transforms

from src.core.config import settings

log = structlog.get_logger()

# ImageNet normalisation
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    cwd = Path.cwd() / path
    if cwd.exists():
        return cwd

    parents = Path(__file__).resolve().parents
    if len(parents) > 4:
        repo = parents[4] / path
        if repo.exists():
            return repo
    return cwd


class ModelRegistry:
    """Singleton-ish registry: call ``load()`` once at startup."""

    def __init__(self) -> None:
        self.device: torch.device = torch.device("cpu")
        self.osnet = None
        self.lmbn = None
        self.gender_model = None
        self.multi_attr_model = None  # 8-task PA-100K classifier; takes priority over gender_model

        self.embedding_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 128)),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        self.classification_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    # ── Bootstrap ─────────────────────────────────────────────────────

    def load(self) -> None:
        self.device = self._pick_device()
        log.info("model_registry.device", device=str(self.device))

        # OSNet (always loaded)
        osnet_path = _resolve_path(settings.osnet_weights)
        if osnet_path.exists():
            from src.models.osnet import osnet_x1_0
            self.osnet = osnet_x1_0(weight_path=str(osnet_path), device=self.device)
            self.osnet.eval()
            log.info("model_registry.osnet_loaded")
        else:
            log.warning("model_registry.osnet_weights_missing", path=str(osnet_path))

        # LMBN (optional)
        if settings.lmbn_weights:
            lmbn_path = _resolve_path(settings.lmbn_weights)
            if lmbn_path.exists():
                from src.models.lightmbn_n import LMBN_n
                self.lmbn = LMBN_n(
                    num_classes=1000, feats=512, activation_map=False,
                    osnet_weight_path=str(osnet_path) if osnet_path.exists() else None,
                    device=self.device,
                )
                self.lmbn.eval()
                log.info("model_registry.lmbn_loaded")

        # Multi-attribute classifier (8 PA-100K tasks). Loaded preferentially over the
        # legacy single-task gender classifier — when present, /gender/classify is served
        # by extracting the gender head from this model.
        multi_attr_path = _resolve_path(settings.multi_attr_weights)
        if multi_attr_path.exists():
            try:
                from src.models.multi_attr_classifier import MultiAttrEfficientNetB0
                self.multi_attr_model = MultiAttrEfficientNetB0(
                    weight_path=str(multi_attr_path), device=self.device,
                )
                log.info("model_registry.multi_attr_loaded", path=str(multi_attr_path))
            except Exception as exc:
                self.multi_attr_model = None
                log.warning(
                    "model_registry.multi_attr_load_failed",
                    path=str(multi_attr_path),
                    error=str(exc),
                )
        else:
            log.info("model_registry.multi_attr_weights_missing", path=str(multi_attr_path))

        # Legacy single-task gender classifier (loaded only if multi-attr is absent).
        if self.multi_attr_model is None:
            eff_path = _resolve_path(settings.efficientnet_weights)
            if eff_path.exists():
                try:
                    from src.models.gender_classifier import GenderClassificationModel
                    self.gender_model = GenderClassificationModel(str(eff_path), self.device)
                    log.info("model_registry.gender_loaded")
                except Exception as exc:
                    self.gender_model = None
                    log.warning(
                        "model_registry.gender_load_failed",
                        path=str(eff_path),
                        error=str(exc),
                    )
            else:
                log.warning("model_registry.gender_weights_missing", path=str(eff_path))

        self._warmup()

    def _warmup(self) -> None:
        with torch.no_grad():
            dummy_emb = torch.randn(1, 3, 256, 128, device=self.device)
            if self.osnet:
                self.osnet(dummy_emb)
            if self.lmbn:
                self.lmbn(dummy_emb)
            dummy_cls = torch.randn(1, 3, 224, 224, device=self.device)
            if self.multi_attr_model:
                self.multi_attr_model(dummy_cls)
            elif self.gender_model:
                self.gender_model(dummy_cls)
        log.info("model_registry.warmup_done")

    # ── Inference helpers ─────────────────────────────────────────────

    def preprocess_embedding(self, image_bytes: bytes) -> torch.Tensor:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.embedding_transform(img).unsqueeze(0).to(self.device)

    def preprocess_classification(self, image_bytes: bytes) -> torch.Tensor:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.classification_transform(img).unsqueeze(0).to(self.device)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            return vec / norm
        return vec

    def _fallback_embedding_from_tensor(self, tensor: torch.Tensor) -> list[float]:
        flat = tensor.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if flat.size == 0:
            flat = np.zeros(1, dtype=np.float32)

        target_dim = settings.embedding_dim
        if flat.size >= target_dim:
            vec = flat[:target_dim]
        else:
            repeats = int(np.ceil(target_dim / flat.size))
            vec = np.tile(flat, repeats)[:target_dim]

        vec = self._l2_normalize(vec.astype(np.float32))
        return vec.tolist()

    @torch.no_grad()
    def extract_embedding_from_tensors(
        self, tensors: torch.Tensor, model: str = "osnet",
    ) -> list[list[float]]:
        """Batch inference on pre-built tensors. Used by EmbeddingBatchQueue."""
        if model == "lmbn" and self.lmbn is not None:
            features = self.lmbn(tensors).mean(dim=2)
            return features.cpu().numpy().tolist()
        if self.osnet is not None:
            features = self.osnet(tensors)
            return features.cpu().numpy().tolist()

        return [self._fallback_embedding_from_tensor(tensor) for tensor in tensors]

    @torch.no_grad()
    def extract_embedding(self, image_bytes: bytes, model: str = "osnet") -> list[float]:
        tensor = self.preprocess_embedding(image_bytes)
        if model == "lmbn" and self.lmbn is not None:
            features = self.lmbn(tensor)  # [1, 512, 7]
            features = features.mean(dim=2)  # [1, 512]
            return features.cpu().numpy().flatten().tolist()
        if self.osnet is not None:
            features = self.osnet(tensor)  # [1, 512]
            return features.cpu().numpy().flatten().tolist()

        return self._fallback_embedding_from_tensor(tensor[0])

    @torch.no_grad()
    def extract_embedding_batch(self, images: list[bytes], model: str = "osnet") -> list[list[float]]:
        if not images:
            return []
        tensors = torch.cat([self.preprocess_embedding(img) for img in images], dim=0)
        if model == "lmbn" and self.lmbn is not None:
            features = self.lmbn(tensors).mean(dim=2)
        elif self.osnet is not None:
            features = self.osnet(tensors)
        else:
            raise RuntimeError("No embedding model loaded")
        return features.cpu().numpy().tolist()

    @torch.no_grad()
    def classify_attributes(self, image_bytes: bytes) -> dict:
        """Run the 8-task multi-attribute classifier on a single crop."""
        if self.multi_attr_model is None:
            raise RuntimeError("Multi-attribute model not loaded")
        tensor = self.preprocess_classification(image_bytes)
        return self.multi_attr_model.predict(tensor)

    @torch.no_grad()
    def classify_gender(self, image_bytes: bytes) -> dict:
        """Backward-compatible gender endpoint.

        Prefers the multi-attribute model (better accuracy, regularized training)
        and extracts its gender head. Falls back to the legacy single-task model.
        Response shape is unchanged — callers like ``GenderVoter`` keep working.
        """
        if self.multi_attr_model is not None:
            tensor = self.preprocess_classification(image_bytes)
            attrs = self.multi_attr_model.predict(tensor)
            gen = attrs["gender"]
            return {
                "gender": gen["label"],
                "confidence": gen["confidence"],
                "probabilities": gen["probabilities"],
            }
        if self.gender_model is None:
            return {
                "gender": "unknown",
                "confidence": 0.0,
                "probabilities": {},
            }
        tensor = self.preprocess_classification(image_bytes)
        logits = self.gender_model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax())
        from src.models.gender_classifier import GenderClassificationModel
        labels = GenderClassificationModel.LABELS
        return {
            "gender": labels[idx],
            "confidence": float(probs[idx]),
            "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))},
        }

    @torch.no_grad()
    def compute_similarity(self, img1_bytes: bytes, img2_bytes: bytes, model: str = "osnet") -> float:
        t1 = self.preprocess_embedding(img1_bytes)
        t2 = self.preprocess_embedding(img2_bytes)
        if model == "lmbn" and self.lmbn is not None:
            f1 = self.lmbn(t1).mean(dim=2)
            f2 = self.lmbn(t2).mean(dim=2)
        elif self.osnet is not None:
            f1 = self.osnet(t1)
            f2 = self.osnet(t2)
        else:
            raise RuntimeError("No embedding model loaded")
        return float(torch.nn.functional.cosine_similarity(f1, f2).item())

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _pick_device() -> torch.device:
        if settings.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(settings.device)

    @property
    def is_ready(self) -> bool:
        return self.osnet is not None
