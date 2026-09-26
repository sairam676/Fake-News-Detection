"""ResNet-18 Baseline Feature Extractor for GenImage RGB classification benchmark."""

from typing import Optional, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from .extractors import preprocess_image_tensor


class ResNet18FeatureExtractor:
    """Extracts 512-dimensional feature representations from ResNet-18."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        try:
            import torchvision.models as models
            weights = models.ResNet18_Weights.DEFAULT
            base_model = models.resnet18(weights=weights)
        except Exception:
            # Fallback via PyTorch Hub if torchvision module import fails
            base_model = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V1")

        # Strip the final FC layer so we get 512-d global average pooling features
        self.backbone = nn.Sequential(*list(base_model.children())[:-1]).to(self.device)
        self.backbone.eval()

    @torch.no_grad()
    def extract(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Extracts 512-d feature vector from standard ResNet-18 backbone."""
        tensor = preprocess_image_tensor(image, target_size=224, device=self.device)
        feat = self.backbone(tensor)  # [1, 512, 1, 1]
        feat = torch.flatten(feat, 1).squeeze(0).cpu().numpy()
        return feat.astype(np.float32)
