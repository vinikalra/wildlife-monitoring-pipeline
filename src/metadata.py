import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

# e.g. "S1_B04_R1_PICT0012.JPG" -> season=S1, station=B04, roll=R1, frame=PICT0012
FILENAME_PATTERN = re.compile(
    r"^(?P<season>[^_]+)_(?P<station>[^_]+)_(?P<roll>[^_]+)_(?P<frame>[^.]+)\.\w+$"
)


@dataclass
class ImageMetadata:
    filename: str
    season: str | None
    station: str | None
    roll: str | None
    frame: str | None
    captured_at: str | None  # ISO 8601, from EXIF, if present


def parse_filename(path: Path) -> tuple[str | None, str | None, str | None, str | None]:
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        return None, None, None, None
    return match["season"], match["station"], match["roll"], match["frame"]


def read_capture_time(path: Path) -> str | None:
    try:
        exif = Image.open(path).getexif()
    except Exception:
        return None

    raw = exif.get(306)  # DateTime tag
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").isoformat()
    except ValueError:
        return None


def extract_metadata(path: Path) -> ImageMetadata:
    season, station, roll, frame = parse_filename(path)
    return ImageMetadata(
        filename=path.name,
        season=season,
        station=station,
        roll=roll,
        frame=frame,
        captured_at=read_capture_time(path),
    )
