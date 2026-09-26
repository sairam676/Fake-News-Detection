"""Attribution MLP Classifier and Open-Set ('Unknown Generator') Thresholding."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributionMLP(nn.Module):
    """Lightweight 2-layer MLP Classifier for Generator Attribution."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OpenSetAttributor:
    """Wrapper that combines the trained MLP with a confidence threshold for Open-Set / Unknown generator detection."""

    def __init__(
        self,
        model: AttributionMLP,
        class_names: List[str],
        confidence_threshold: float = 0.50,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.device = device

    @torch.no_grad()
    def predict(self, feature_vector: Union[np.ndarray, torch.Tensor]) -> Dict[str, Union[str, float, bool, Dict[str, float]]]:
        """Predicts generator label or 'Unknown Generator' if confidence < threshold."""
        if isinstance(feature_vector, np.ndarray):
            x = torch.from_numpy(feature_vector.astype(np.float32))
        else:
            x = feature_vector.float()

        if x.ndim == 1:
            x = x.unsqueeze(0)

        x = x.to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        max_idx = int(np.argmax(probs))
        max_prob = float(probs[max_idx])

        if max_prob < self.confidence_threshold:
            predicted_label = "Unknown / Unidentified Generator"
            is_known = False
        else:
            predicted_label = self.class_names[max_idx]
            is_known = True

        class_probs = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}

        return {
            "predicted_label": predicted_label,
            "confidence": max_prob,
            "is_known_generator": is_known,
            "raw_prediction": self.class_names[max_idx],
            "class_probabilities": class_probs,
        }
