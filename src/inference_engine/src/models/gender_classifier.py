"""EfficientNet-B0 gender classifier.

Ported from reid-microservices/src/model_serving/src/models/efficientnet.py.
"""
import torch
from efficientnet_pytorch import EfficientNet


class GenderClassificationModel:
    LABELS = ["male", "female"]

    def __init__(self, weight_path: str, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EfficientNet.from_pretrained("efficientnet-b0")
        self.model._fc = torch.nn.Linear(self.model._fc.in_features, 2)
        self.model.load_state_dict(
            torch.load(weight_path, map_location=self.device)
        )
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image.to(self.device))
