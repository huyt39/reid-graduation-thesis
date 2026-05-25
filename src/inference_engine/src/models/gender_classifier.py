"""EfficientNet-B0 gender classifier."""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class GenderClassificationModel:
    LABELS = ["male", "female"]

    def __init__(self, weight_path: str, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = efficientnet_b0(weights=None)
        self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 2))
        self.model.load_state_dict(
            torch.load(weight_path, map_location=self.device)
        )
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image.to(self.device))
