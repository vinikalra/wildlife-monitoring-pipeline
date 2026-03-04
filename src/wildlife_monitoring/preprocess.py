from pathlib import Path
from typing import Any


def load_image_rgb(path: Path) -> Any:
    """Load image data.

    Prefer PIL (RGB), then OpenCV, and finally raw bytes as a fallback.
    """
    try:
        from PIL import Image

        image = Image.open(path)
        return image.convert("RGB")
    except Exception:
        pass

    try:
        import cv2

        image = cv2.imread(str(path))
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    return path.read_bytes()
