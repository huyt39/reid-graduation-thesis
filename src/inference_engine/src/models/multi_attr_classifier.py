"""Multi-attribute pedestrian classifier — 8 PA-100K tasks on a single EfficientNet-B0 backbone.

Loads checkpoints produced by ``Yolo-for-Edge-Devices/huy_backup/MultiAttr_EfficientNetB0.py``
and exposes a single forward pass returning per-task logits, plus a ``predict()`` helper
that returns labels + confidences in the JSON shape the API serves.

Task layout MUST match the training script's TASK_NAMES / TASK_NUM_CLASSES order — head
indexing is positional (state_dict keys are ``heads.<task>.weight`` / ``heads.<task>.bias``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


# Order and class counts — must match training script.
TASK_NAMES = ["gender", "age_child", "backpack", "sidebag",
              "hat", "glasses", "sleeve", "lower"]

TASK_NUM_CLASSES = {
    "gender":    2,
    "age_child": 2,
    "backpack":  2,
    "sidebag":   2,
    "hat":       2,
    "glasses":   2,
    "sleeve":    2,
    "lower":     3,
}

LABEL_NAMES = {
    "gender":    ["male", "female"],
    "age_child": ["adult", "child"],
    "backpack":  ["no_backpack", "backpack"],
    "sidebag":   ["no_sidebag", "sidebag"],
    "hat":       ["no_hat", "hat"],
    "glasses":   ["no_glasses", "glasses"],
    "sleeve":    ["short_sleeve", "long_sleeve"],
    "lower":     ["trousers", "shorts", "skirt_dress"],
}


class MultiAttrEfficientNetB0(nn.Module):
    """EfficientNet-B0 backbone + 8 classification heads, inference-only."""

    def __init__(
        self,
        weight_path: str | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build architecture only — actual weights come from the checkpoint.
        backbone = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.dropout = backbone.classifier[0]  # Dropout(0.2)
        feat_dim = backbone.classifier[1].in_features  # 1280

        self.heads = nn.ModuleDict({
            t: nn.Linear(feat_dim, TASK_NUM_CLASSES[t]) for t in TASK_NAMES
        })

        if weight_path is not None:
            state = torch.load(weight_path, map_location=self.device, weights_only=True)
            self.load_state_dict(state)

        self.to(self.device)
        self.eval()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(self.device)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return {t: head(x) for t, head in self.heads.items()}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, dict]:
        """Single-image prediction → dict of {task: {label, confidence, probabilities}}."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        logits = self.forward(x)
        out: dict[str, dict] = {}
        for t in TASK_NAMES:
            probs = torch.softmax(logits[t], dim=1)[0]
            idx = int(probs.argmax())
            labels = LABEL_NAMES[t]
            out[t] = {
                "label": labels[idx],
                "confidence": float(probs[idx]),
                "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))},
            }
        return out
