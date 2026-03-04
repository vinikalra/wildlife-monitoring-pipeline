from pathlib import Path

from wildlife_monitoring.classifier import ConstantClassifier
from wildlife_monitoring.config import PipelineConfig
from wildlife_monitoring.pipeline import WildlifePipeline


def test_pipeline_routes_images_with_constant_classifier(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    image_path = input_dir / "frame1.jpg"
    image_path.write_bytes(b"not-a-real-image")

    pipeline = WildlifePipeline(
        config=PipelineConfig(input_dir=input_dir, output_dir=output_dir, copy_files=True),
        classifier=ConstantClassifier(label="Fox", score=0.9),
    )

    results = pipeline.run()

    assert len(results) == 1
    assert (output_dir / "Fox" / "frame1.jpg").exists()
    assert image_path.exists(), "copy_files=True should keep source image"


def test_pipeline_uses_unknown_for_low_confidence(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    image_path = input_dir / "frame2.jpg"
    image_path.write_bytes(b"not-a-real-image")

    pipeline = WildlifePipeline(
        config=PipelineConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            confidence_threshold=0.8,
            copy_files=True,
        ),
        classifier=ConstantClassifier(label="Deer", score=0.1),
    )

    pipeline.run()

    assert (output_dir / "Unknown" / "frame2.jpg").exists()
