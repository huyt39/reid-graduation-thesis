"""Triton inference backend — drop-in replacement for in-process PyTorch.

Only the raw tensor I/O hits Triton. Preprocessing (resize/normalize/NCHW)
and post-processing (L2 norm for embeddings, softmax+argmax for attribute
heads) stay on the FastAPI side so Triton's batcher sees uniform tensors.
"""
from __future__ import annotations

import numpy as np
import structlog
import tritonclient.http as httpclient

log = structlog.get_logger()


# Must mirror src.models.multi_attr_classifier.LABEL_NAMES so the JSON
# shape returned to the worker is unchanged.
ATTR_TASKS = [
    "gender", "age_child", "backpack", "sidebag",
    "hat", "glasses", "sleeve", "lower",
]
ATTR_LABELS = {
    "gender":    ["male", "female"],
    "age_child": ["adult", "child"],
    "backpack":  ["no_backpack", "backpack"],
    "sidebag":   ["no_sidebag", "sidebag"],
    "hat":       ["no_hat", "hat"],
    "glasses":   ["no_glasses", "glasses"],
    "sleeve":    ["short_sleeve", "long_sleeve", "long_sleeve_other"],
    "lower":     ["trousers", "shorts"],
}


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class TritonBackend:
    def __init__(self, url: str = "triton:8000") -> None:
        self.url = url
        self.client = httpclient.InferenceServerClient(url=url, verbose=False)
        self._osnet_ready = False
        self._multi_attr_ready = False
        self._check_models()

    def _check_models(self) -> None:
        try:
            self._osnet_ready = self.client.is_model_ready("osnet")
        except Exception as exc:
            log.warning("triton.osnet_ready_check_failed", error=str(exc))
        try:
            self._multi_attr_ready = self.client.is_model_ready("multi_attr")
        except Exception as exc:
            log.warning("triton.multi_attr_ready_check_failed", error=str(exc))
        log.info(
            "triton.models_checked",
            osnet=self._osnet_ready,
            multi_attr=self._multi_attr_ready,
        )

    # ── Embedding ──────────────────────────────────────────────────────

    def embed(self, tensor: np.ndarray) -> np.ndarray:
        """Run OSNet on a NCHW float32 batch and return L2-normalized embeddings.

        tensor: shape (N, 3, 256, 128) float32
        returns: shape (N, 512) float32, L2-normalized rows
        """
        if not self._osnet_ready:
            raise RuntimeError("Triton osnet model not ready")

        inputs = [httpclient.InferInput("input", tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(tensor.astype(np.float32))
        outputs = [httpclient.InferRequestedOutput("embedding")]

        try:
            # tritonclient[http] uses a gevent-backed connection pool that can keep
            # stale sockets after container restarts or rapid repeated calls. A fresh
            # lightweight client per infer avoids returning intermittent raw 500s from
            # the API layer while keeping preprocessing/batching unchanged.
            client = httpclient.InferenceServerClient(url=self.url, verbose=False)
            result = client.infer("osnet", inputs, outputs=outputs)
        except Exception as exc:
            log.error("triton.osnet_infer_failed", error=str(exc))
            raise RuntimeError(f"Triton osnet inference failed: {exc}") from exc

        emb = result.as_numpy("embedding")
        if emb is None:
            raise RuntimeError("Triton returned no embedding output")
        # L2 normalize per row
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        return (emb / norms).astype(np.float32)

    # ── Multi-attribute classification ────────────────────────────────

    def classify_attributes(self, tensor: np.ndarray) -> dict[str, dict]:
        """Run the 8-task classifier on a single 1x3x224x224 tensor.

        Returns dict shaped like MultiAttrEfficientNetB0.predict():
            {task: {"label": str, "confidence": float, "probabilities": {label: prob}}}
        """
        if not self._multi_attr_ready:
            raise RuntimeError("Triton multi_attr model not ready")

        if tensor.ndim == 3:
            tensor = np.expand_dims(tensor, 0)

        inputs = [httpclient.InferInput("input", tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(tensor.astype(np.float32))
        outputs = [httpclient.InferRequestedOutput(t) for t in ATTR_TASKS]

        try:
            client = httpclient.InferenceServerClient(url=self.url, verbose=False)
            result = client.infer("multi_attr", inputs, outputs=outputs)
        except Exception as exc:
            log.error("triton.multi_attr_infer_failed", error=str(exc))
            raise RuntimeError(f"Triton multi_attr inference failed: {exc}") from exc

        out: dict[str, dict] = {}
        for task in ATTR_TASKS:
            logits = result.as_numpy(task)
            if logits is None:
                continue
            probs = _softmax(logits[0])
            labels = ATTR_LABELS[task][: probs.shape[0]]
            # if model has more classes than known labels, pad with positional names
            while len(labels) < probs.shape[0]:
                labels.append(f"class_{len(labels)}")
            idx = int(probs.argmax())
            out[task] = {
                "label": labels[idx],
                "confidence": float(probs[idx]),
                "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))},
            }
        return out

    @property
    def osnet_ready(self) -> bool:
        return self._osnet_ready

    @property
    def multi_attr_ready(self) -> bool:
        return self._multi_attr_ready
