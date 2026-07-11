"""Loads and holds all inference models + their preprocessing transforms."""
# app.py is http gate, this is infer: image bytes to tensor, run model and tensor output to json
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
import structlog
import torch
from PIL import Image
from torchvision import transforms

from src.core.config import settings

log = structlog.get_logger()

# ImageNet normalisation
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class _LetterboxResize:

    def __init__(self, target_h: int = 256, target_w: int = 128, fill: float = 0.5) -> None:
        self.target_h = target_h
        self.target_w = target_w
        self.fill = fill

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3:
            return tensor
        _, h, w = tensor.shape
        if h <= 0 or w <= 0:
            return tensor
        scale = min(self.target_h / h, self.target_w / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        # Resize preserving aspect via bilinear interpolation
        resized = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        # Center pad to target with the fill value
        pad_top = (self.target_h - new_h) // 2
        pad_bottom = self.target_h - new_h - pad_top
        pad_left = (self.target_w - new_w) // 2
        pad_right = self.target_w - new_w - pad_left
        # torch.nn.functional.pad uses (pad_left, pad_right, pad_top, pad_bottom)
        padded = torch.nn.functional.pad(
            resized,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=self.fill,
        )
        return padded


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
        self.osnet_onnx = None
        self.osnet_onnx_input_name: str | None = None
        self.osnet_ain = None
        self.lmbn = None
        self.gender_model = None
        self.multi_attr_model = None  # 8-task PA-100K classifier; takes priority over gender_model
        self.standalone_gender_model = None  # PETA-trained gender classifier; overrides multi_attr gender head
        self.effb0_gender_model = None  # EfficientNet-B0 (lukemelas) gender classifier; top priority when present

        self.embedding_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 128)),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        # PAR classifier crops are typically taller than wide (~2:1 H:W).
        # Naive Resize((224, 224)) squeezes that to 1:1 and warps body
        # geometry — degrades attributes that depend on local layout
        # (sleeve, lower, hat, glasses). Default to aspect-preserving
        # letterbox; settings.par_letterbox=False restores the old path.
        _classification_resize = (
            _LetterboxResize(target_h=224, target_w=224, fill=0.5)
            if getattr(settings, "par_letterbox", True)
            else transforms.Resize((224, 224))
        )
        self.classification_transform = transforms.Compose([
            transforms.ToTensor(),
            _classification_resize,
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    # ── Bootstrap ─────────────────────────────────────────────────────

    def load(self) -> None:
        self.device = self._pick_device()
        log.info("model_registry.device", device=str(self.device))

        onnx_path = _resolve_path(settings.osnet_onnx_path) if settings.osnet_onnx_path else None
        if onnx_path is not None and onnx_path.exists():
            try:
                self.osnet_onnx = ort.InferenceSession(
                    str(onnx_path),
                    providers=["CPUExecutionProvider"],
                )
                self.osnet_onnx_input_name = self.osnet_onnx.get_inputs()[0].name
                log.info("model_registry.osnet_onnx_loaded", path=str(onnx_path))
            except Exception as exc:
                self.osnet_onnx = None
                self.osnet_onnx_input_name = None
                log.warning(
                    "model_registry.osnet_onnx_load_failed",
                    path=str(onnx_path),
                    error=str(exc),
                )

        # OSNet (always loaded)
        osnet_path = _resolve_path(settings.osnet_weights) if settings.osnet_weights else None
        if osnet_path is not None and osnet_path.exists():
            try:
                from src.models.osnet import osnet_x1_0
                self.osnet = osnet_x1_0(weight_path=str(osnet_path), device=self.device)
                self.osnet.eval()
                log.info("model_registry.osnet_loaded")
            except Exception as exc:
                self.osnet = None
                log.warning(
                    "model_registry.osnet_load_failed",
                    path=str(osnet_path),
                    error=str(exc),
                )
        else:
            log.warning("model_registry.osnet_weights_missing", path=str(osnet_path) if osnet_path else "")

        # OSNet-AIN (domain-generalization variant; loaded if weights present)
        if getattr(settings, "osnet_ain_weights", ""):
            ain_path = _resolve_path(settings.osnet_ain_weights)
            if ain_path is not None and ain_path.exists():
                try:
                    from src.models.osnet_ain import osnet_ain_x1_0
                    self.osnet_ain = osnet_ain_x1_0(weight_path=str(ain_path), device=self.device)
                    self.osnet_ain.eval()
                    log.info("model_registry.osnet_ain_loaded", path=str(ain_path))
                except Exception as exc:
                    self.osnet_ain = None
                    log.warning(
                        "model_registry.osnet_ain_load_failed",
                        path=str(ain_path),
                        error=str(exc),
                    )
            else:
                log.warning("model_registry.osnet_ain_weights_missing", path=str(ain_path) if ain_path else "")

        # LMBN (optional). Not used for production embeddings (the worker
        # requests "osnet_ain"); load is best-effort so a missing/broken LMBN
        # or backbone weight degrades to lmbn=None instead of taking down the
        # whole engine at startup.
        if settings.lmbn_weights:
            lmbn_path = _resolve_path(settings.lmbn_weights)
            if lmbn_path.exists():
                try:
                    from src.models.lightmbn_n import LMBN_n
                    self.lmbn = LMBN_n(
                        num_classes=767, feats=512, activation_map=False,
                        osnet_weight_path=str(osnet_path) if osnet_path and osnet_path.exists() else None,
                        device=self.device,
                    )
                    ckpt = torch.load(str(lmbn_path), map_location=self.device, weights_only=False)
                    state_dict = ckpt
                    if isinstance(ckpt, dict):
                        for key in ("state_dict", "net", "model"):
                            if key in ckpt and isinstance(ckpt[key], dict):
                                state_dict = ckpt[key]
                                break
                    def _strip(k: str) -> str:
                        for prefix in ("module.", "model."):
                            if k.startswith(prefix):
                                k = k[len(prefix):]
                        return k
                    state_dict = {_strip(k): v for k, v in state_dict.items()}
                    # Drop shape-mismatched keys — typically classifier heads from a
                    # fine-tune saved with a different num_classes than the production
                    # model (767). load_state_dict(strict=False) ignores missing/unexpected
                    # keys but still errors on shape mismatch. The classifier is unused
                    # at inference; only the BNNeck embedding is read.
                    model_sd = self.lmbn.state_dict()
                    filtered: dict[str, torch.Tensor] = {}
                    dropped: list[str] = []
                    for k, v in state_dict.items():
                        if k in model_sd and model_sd[k].shape != v.shape:
                            dropped.append(k)
                            continue
                        filtered[k] = v
                    missing, unexpected = self.lmbn.load_state_dict(filtered, strict=False)
                    self.lmbn.to(self.device)
                    self.lmbn.eval()
                    log.info(
                        "model_registry.lmbn_loaded",
                        path=str(lmbn_path),
                        missing=len(missing),
                        unexpected=len(unexpected),
                        dropped_shape_mismatch=len(dropped),
                    )
                except Exception as exc:
                    self.lmbn = None
                    log.warning(
                        "model_registry.lmbn_load_failed",
                        path=str(lmbn_path),
                        error=str(exc),
                    )

        # Multi-attribute classifier (8 PA-100K tasks). Loaded preferentially over the
        # legacy single-task gender classifier — when present, /gender/classify is served
        # by extracting the gender head from this model.
        multi_attr_path = _resolve_path(settings.multi_attr_weights) if settings.multi_attr_weights else None
        if multi_attr_path is not None and multi_attr_path.exists():
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
            log.info("model_registry.multi_attr_weights_missing", path=str(multi_attr_path) if multi_attr_path else "")

        # Standalone gender classifier (PETA-trained, 88% acc). Overrides the gender
        # head of the multi-attr model when present.
        standalone_gender_path = (
            _resolve_path(settings.standalone_gender_weights)
            if settings.standalone_gender_weights
            else None
        )
        if standalone_gender_path is not None and standalone_gender_path.exists():
            try:
                from src.models.standalone_gender import StandaloneGenderModel
                self.standalone_gender_model = StandaloneGenderModel(
                    str(standalone_gender_path), self.device
                )
                log.info("model_registry.standalone_gender_loaded", path=str(standalone_gender_path))
            except Exception as exc:
                self.standalone_gender_model = None
                log.warning(
                    "model_registry.standalone_gender_load_failed",
                    path=str(standalone_gender_path),
                    error=str(exc),
                )
        else:
            log.info(
                "model_registry.standalone_gender_weights_missing",
                path=str(standalone_gender_path) if standalone_gender_path else "",
            )

        # Standalone gender classifier (PETA-trained, 88% acc). Overrides the gender
        # head of the multi-attr model when present.
        standalone_gender_path = _resolve_path(settings.standalone_gender_weights)
        if standalone_gender_path.exists():
            try:
                from src.models.standalone_gender import StandaloneGenderModel
                self.standalone_gender_model = StandaloneGenderModel(
                    str(standalone_gender_path), self.device
                )
                log.info("model_registry.standalone_gender_loaded", path=str(standalone_gender_path))
            except Exception as exc:
                self.standalone_gender_model = None
                log.warning(
                    "model_registry.standalone_gender_load_failed",
                    path=str(standalone_gender_path),
                    error=str(exc),
                )
        else:
            log.info("model_registry.standalone_gender_weights_missing", path=str(standalone_gender_path))

        # EfficientNet-B0 gender classifier (efficientnet_pytorch layout). Highest-priority
        # gender source when present — overrides both the standalone and multi-attr gender heads.
        effb0_gender_path = (
            _resolve_path(settings.effb0_gender_weights)
            if settings.effb0_gender_weights
            else None
        )
        if effb0_gender_path is not None and effb0_gender_path.exists():
            try:
                from src.models.effb0_gender import EffB0GenderModel
                self.effb0_gender_model = EffB0GenderModel(
                    str(effb0_gender_path), self.device
                )
                log.info("model_registry.effb0_gender_loaded", path=str(effb0_gender_path))
            except Exception as exc:
                self.effb0_gender_model = None
                log.warning(
                    "model_registry.effb0_gender_load_failed",
                    path=str(effb0_gender_path),
                    error=str(exc),
                )
        else:
            log.info(
                "model_registry.effb0_gender_weights_missing",
                path=str(effb0_gender_path) if effb0_gender_path else "",
            )

        # Legacy single-task gender classifier (loaded only if multi-attr is absent).
        if self.multi_attr_model is None:
            eff_path = _resolve_path(settings.efficientnet_weights) if settings.efficientnet_weights else None
            if eff_path is not None and eff_path.exists():
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
                log.warning("model_registry.gender_weights_missing", path=str(eff_path) if eff_path else "")

        self._warmup()

    def _warmup(self) -> None:
        with torch.no_grad():
            dummy_emb = torch.randn(1, 3, 256, 128, device=self.device)
            if self.osnet:
                self.osnet(dummy_emb)
            if self.osnet_ain:
                self.osnet_ain(dummy_emb)
            if self.osnet_onnx is not None and self.osnet_onnx_input_name is not None:
                self._extract_onnx_embedding(dummy_emb)
            if self.lmbn:
                self.lmbn(dummy_emb)
            dummy_cls = torch.randn(1, 3, 224, 224, device=self.device)
            if self.multi_attr_model:
                self.multi_attr_model(dummy_cls)
            elif self.gender_model:
                self.gender_model(dummy_cls)
            if self.standalone_gender_model:
                self.standalone_gender_model.model(dummy_cls)
            if self.effb0_gender_model:
                self.effb0_gender_model.model(dummy_cls)
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

    @staticmethod
    def _raise_embedding_unavailable() -> None:
        raise RuntimeError(
            "No ReID embedding model loaded. Configure OSNet PyTorch/ONNX weights; "
            "refusing to return fallback pixel embeddings."
        )

    def _extract_onnx_embedding(self, tensors: torch.Tensor) -> np.ndarray:
        if self.osnet_onnx is None or self.osnet_onnx_input_name is None:
            self._raise_embedding_unavailable()
        arr = tensors.detach().cpu().numpy().astype(np.float32)
        emb = self.osnet_onnx.run(None, {self.osnet_onnx_input_name: arr})[0].astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.maximum(norms, 1e-8)

    @torch.no_grad()
    def extract_embedding_from_tensors(
        self, tensors: torch.Tensor, model: str = "osnet",
    ) -> list[list[float]]:
        """Batch inference on pre-built tensors. Used by EmbeddingBatchQueue."""
        if model == "osnet_ain" and self.osnet_ain is not None:
            features = self.osnet_ain(tensors)
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)
            return features.cpu().numpy().tolist()
        if model == "lmbn" and self.lmbn is not None:
            features = self.lmbn(tensors).mean(dim=2)
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)
            return features.cpu().numpy().tolist()
        if self.osnet is not None:
            features = self.osnet(tensors)
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)
            return features.cpu().numpy().tolist()
        if self.osnet_onnx is not None:
            return self._extract_onnx_embedding(tensors).tolist()

        self._raise_embedding_unavailable()

    @torch.no_grad()
    def extract_embedding(self, image_bytes: bytes, model: str = "osnet") -> list[float]:
        tensor = self.preprocess_embedding(image_bytes)
        if model == "osnet_ain" and self.osnet_ain is not None:
            features = self.osnet_ain(tensor)  # [1, 512]
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)
            return features.cpu().numpy().flatten().tolist()
        if model == "lmbn" and self.lmbn is not None:
            features = self.lmbn(tensor)  # [1, 512, 7]
            features = features.mean(dim=2)  # [1, 512]
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)
            return features.cpu().numpy().flatten().tolist()
        if self.osnet is not None:
            features = self.osnet(tensor)  # [1, 512]
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)
            return features.cpu().numpy().flatten().tolist()
        if self.osnet_onnx is not None:
            return self._extract_onnx_embedding(tensor).flatten().tolist()

        self._raise_embedding_unavailable()

    @torch.no_grad()
    def extract_embedding_batch(self, images: list[bytes], model: str = "osnet") -> list[list[float]]:
        if not images:
            return []
        tensors = torch.cat([self.preprocess_embedding(img) for img in images], dim=0)
        if model == "osnet_ain" and self.osnet_ain is not None:
            features = self.osnet_ain(tensors)
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)
        elif model == "lmbn" and self.lmbn is not None:
            features = self.lmbn(tensors).mean(dim=2)
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)
        elif self.osnet is not None:
            features = self.osnet(tensors)
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)
        elif self.osnet_onnx is not None:
            return self._extract_onnx_embedding(tensors).tolist()
        else:
            raise RuntimeError("No embedding model loaded")
        return features.cpu().numpy().tolist()

    @torch.no_grad()
    def classify_attributes(self, image_bytes: bytes) -> dict:
        """Run the 8-task multi-attribute classifier on a single crop."""
        if self.multi_attr_model is None:
            raise RuntimeError("Multi-attribute model not loaded")
        tensor = self.preprocess_classification(image_bytes)
        result = self.multi_attr_model.predict(tensor)
        # Gender head override, highest-priority first. The EffB0 model owns its own
        # preprocessing (plain resize, not letterbox), so it takes raw bytes.
        if self.effb0_gender_model is not None:
            result["gender"] = self.effb0_gender_model.predict(image_bytes)
        elif self.standalone_gender_model is not None:
            result["gender"] = self.standalone_gender_model.predict(tensor)
        return result

    @torch.no_grad()
    def classify_gender(self, image_bytes: bytes) -> dict:
        """Backward-compatible gender endpoint.

        Prefers the EfficientNet-B0 gender model, then the standalone PETA-trained model.
        Falls back to the multi-attribute model gender head, then the legacy model.
        Response shape is unchanged — callers like ``GenderVoter`` keep working.
        """
        if self.effb0_gender_model is not None:
            gen = self.effb0_gender_model.predict(image_bytes)
            return {
                "gender": gen["label"],
                "confidence": gen["confidence"],
                "probabilities": gen["probabilities"],
            }
        if self.standalone_gender_model is not None:
            tensor = self.preprocess_classification(image_bytes)
            gen = self.standalone_gender_model.predict(tensor)
            return {
                "gender": gen["label"],
                "confidence": gen["confidence"],
                "probabilities": gen["probabilities"],
            }
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
            # cuda -> mps -> cpu. MPS lets the engine use the Apple Silicon GPU
            # when running natively on macOS (Docker on macOS has no Metal
            # passthrough, so MPS only applies to host-native runs). The cuda
            # branch stays first so GPU/Linux hosts keep their existing behavior.
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(settings.device)

    @property
    def is_ready(self) -> bool:
        return (
            self.osnet is not None
            or self.osnet_ain is not None
            or self.osnet_onnx is not None
        )
