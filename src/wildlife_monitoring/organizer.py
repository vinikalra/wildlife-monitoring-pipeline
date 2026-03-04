from pathlib import Path
import shutil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(input_dir: Path, recursive: bool = True):
    globber = input_dir.rglob if recursive else input_dir.glob
    for path in globber("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def relocate_image(source: Path, output_dir: Path, label: str, copy_files: bool = False) -> Path:
    destination_dir = output_dir / label
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination = destination_dir / source.name
    if copy_files:
        shutil.copy2(source, destination)
    else:
        shutil.move(source, destination)
    return destination
