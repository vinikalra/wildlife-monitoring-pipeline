from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .classifier import Classifier
from .config import PipelineConfig
from .organizer import iter_images, relocate_image
from .preprocess import load_image_rgb


@dataclass(slots=True)
class ProcessResult:
    source: Path
    destination: Path
    label: str
    confidence: float


class WildlifePipeline:
    def __init__(self, config: PipelineConfig, classifier: Classifier) -> None:
        self.config = config
        self.classifier = classifier

    def run(self) -> list[ProcessResult]:
        self.config.ensure_directories()
        results: list[ProcessResult] = []

        for image_path in iter_images(self.config.input_dir, recursive=self.config.recursive):
            image = load_image_rgb(image_path)
            label, confidence = self.classifier.predict(image)
            if confidence < self.config.confidence_threshold:
                label = "Unknown"

            destination = relocate_image(
                source=image_path,
                output_dir=self.config.output_dir,
                label=label,
                copy_files=self.config.copy_files,
            )
            results.append(
                ProcessResult(
                    source=image_path,
                    destination=destination,
                    label=label,
                    confidence=confidence,
                )
            )

        return results
