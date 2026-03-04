from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class Classifier(Protocol):
    def predict(self, image: Any) -> tuple[str, float]:
        """Predict class label and confidence for a single image."""


@dataclass(slots=True)
class RuleBasedWildlifeClassifier:
    """Map ImageNet predictions to broad wildlife labels."""

    model_name: str = "mobilenet_v3_large"

    def __post_init__(self) -> None:
        import torch
        from torchvision import models

        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        self._categories = weights.meta["categories"]
        self._transform = weights.transforms()
        self._model = models.mobilenet_v3_large(weights=weights)
        self._model.eval()
        self._torch = torch

    def predict(self, image: Any) -> tuple[str, float]:
        try:
            from PIL import Image
            import numpy as np

            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
        except Exception:
            pass

        tensor = self._transform(image).unsqueeze(0)
        with self._torch.no_grad():
            logits = self._model(tensor)
            probs = self._torch.softmax(logits[0], dim=0)
            score, idx = self._torch.max(probs, dim=0)

        raw_label = self._categories[idx.item()]
        mapped = self._map_label(raw_label)
        return mapped, float(score.item())

    @staticmethod
    def _map_label(raw_label: str) -> str:
        lowered = raw_label.lower()
        mapping = {
            "fox": "Fox",
            "deer": "Deer",
            "eagle": "Eagle",
            "owl": "Owl",
            "wolf": "Wolf",
            "bear": "Bear",
            "boar": "Boar",
            "rabbit": "Rabbit",
            "squirrel": "Squirrel",
            "otter": "Otter",
            "bird": "Bird",
            "duck": "Duck",
            "goose": "Goose",
            "cat": "Wildcat",
            "dog": "Canid",
        }
        for token, species in mapping.items():
            if token in lowered:
                return species
        return "Unknown"


@dataclass(slots=True)
class ConstantClassifier:
    label: str = "Unknown"
    score: float = 1.0

    def predict(self, image: Any) -> tuple[str, float]:
        _ = image
        return self.label, self.score
