from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    """Configuration values for the wildlife image processing pipeline."""

    input_dir: Path
    output_dir: Path
    confidence_threshold: float = 0.2
    copy_files: bool = False
    recursive: bool = True

    def ensure_directories(self) -> None:
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
