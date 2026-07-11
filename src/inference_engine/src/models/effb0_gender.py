"""EfficientNet-B0 gender classifier (``efficientnet_pytorch`` / lukemelas naming).

Loads checkpoints saved from an ``EfficientNet.from_name('efficientnet-b0', num_classes=2)``
head — state_dict keys are ``_conv_stem`` / ``_blocks.*`` / ``_fc``, which is a different
layout from the torchvision-based ``StandaloneGenderModel`` (``features.*`` / ``classifier.*``),
so the two are NOT interchangeable loaders.

Preprocessing and label order were verified empirically against the previously deployed
gender model on device2/device3 crops (9/9 agreement):
  * input  = plain 224x224 resize + ImageNet normalization (NO aspect-preserving letterbox)
  * labels = index 0 -> male, index 1 -> female

Because this checkpoint needs plain-resize (not the letterbox used by the multi-attr
classifier), the model owns its full preprocessing from raw bytes rather than reusing the
registry's shared classification transform — keeping it correct regardless of the
``par_letterbox`` setting.
"""
from io import BytesIO

import torch
from PIL import Image
from torchvision import transforms

from efficientnet_pytorch import EfficientNet

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class EffB0GenderModel:
    LABELS = ["male", "female"]  # index 0 = male, index 1 = female (verified)

    def __init__(self, weight_path: str, device: torch.device) -> None:
        model = EfficientNet.from_name("efficientnet-b0", num_classes=2)
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
        self.model = model.to(device).eval()
        self.device = device
        # Plain 224x224 resize + ImageNet norm — this checkpoint was NOT trained with
        # the aspect-preserving letterbox used for the multi-attribute classifier.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> dict:
        """Single-crop gender prediction → {label, confidence, probabilities}."""
        tensor = self.preprocess(image_bytes)
        probs = torch.softmax(self.model(tensor), dim=1)[0]
        idx = int(probs.argmax())
        return {
            "label": self.LABELS[idx],
            "confidence": round(float(probs[idx]), 4),
            "probabilities": {self.LABELS[i]: round(float(probs[i]), 4) for i in range(2)},
        }
