"""Stage 1: is there an animal (vs. empty/person/vehicle), and where.

Wraps Microsoft's MegaDetector (via PytorchWildlife) -- a model trained
specifically to tell camera-trap frames apart as animal / person / vehicle /
empty. This is far better calibrated for that decision than a generic
ImageNet classifier, which is why it now gates what gets passed to the
species-hint classifier in classifier.py.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from PytorchWildlife.models import detection as pw_detection


@dataclass
class Detection:
    category: str  # "animal" | "person" | "vehicle"
    confidence: float
    bbox: tuple[int, int, int, int]  # xyxy pixel coordinates


class AnimalDetector:
    def __init__(self, confidence_threshold: float = 0.2, device: str = "cpu"):
        self.confidence_threshold = confidence_threshold
        self.model = pw_detection.MegaDetectorV6(
            device=device, pretrained=True, version="MDV6-yolov9-c"
        )

    def detect(self, image_path: Path) -> list[Detection]:
        image = np.array(Image.open(image_path).convert("RGB"))
        result = self.model.single_image_detection(
            image, img_path=str(image_path), det_conf_thres=self.confidence_threshold
        )
        detections = result["detections"]
        return [
            Detection(
                category=label.split()[0],
                confidence=float(conf),
                bbox=tuple(int(v) for v in box),
            )
            for label, conf, box in zip(result["labels"], detections.confidence, detections.xyxy)
        ]

    def best(self, image_path: Path) -> Detection | None:
        detections = self.detect(image_path)
        return max(detections, key=lambda d: d.confidence, default=None)
