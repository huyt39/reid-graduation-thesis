"""Standalone gender classifier trained on PETA from ImageNet-pretrained EfficientNet-B0.

Label mapping: female=0, male=1 (matches training in train_gender_standalone.py).
88% val_acc on PETA — replaces the biased gender head of the PA-100K multi-attr model.
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class StandaloneGenderModel:
    LABELS = ["female", "male"]  # index 0 = female, index 1 = male

    def __init__(self, weight_path: str, device: torch.device) -> None:
        base = efficientnet_b0(weights=None)
        base.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 2))
        state = torch.load(weight_path, map_location=device)
        base.load_state_dict(state)
        self.model = base.to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict(self, tensor: torch.Tensor) -> dict:
        probs = torch.softmax(self.model(tensor.to(self.device)), dim=1)[0]
        idx = int(probs.argmax())
        return {
            "label": self.LABELS[idx],
            "confidence": round(float(probs[idx]), 4),
            "probabilities": {self.LABELS[i]: round(float(probs[i]), 4) for i in range(2)},
        }
