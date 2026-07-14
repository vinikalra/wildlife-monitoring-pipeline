import argparse
import logging
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

from airtable_logger import AirtableLogger
from classifier import WildlifeClassifier
from config import Config
from metadata import extract_metadata

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).resolve().parent.parent / "logs" / "pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pipeline")


def find_images(source_dir: Path) -> list[Path]:
    return sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def destination_for(category: str, config: Config) -> Path:
    return {
        "bird": config.birds_dir,
        "animal": config.animals_dir,
        "empty": config.rejected_dir,
    }[category]


def run(config: Config, dry_run: bool = False) -> None:
    for directory in (config.birds_dir, config.animals_dir, config.rejected_dir):
        directory.mkdir(parents=True, exist_ok=True)

    images = find_images(config.source_dir)
    if not images:
        log.info("No images found in %s", config.source_dir)
        return

    classifier = WildlifeClassifier(confidence_threshold=config.confidence_threshold)
    airtable = None if dry_run else AirtableLogger(
        config.airtable_api_key, config.airtable_base_id, config.airtable_table_name
    )

    counts = {"bird": 0, "animal": 0, "empty": 0, "error": 0}

    for image_path in tqdm(images, desc="Classifying"):
        try:
            result = classifier.classify(image_path)
        except ValueError as exc:
            log.warning("Skipping unreadable image %s: %s", image_path, exc)
            counts["error"] += 1
            continue

        metadata = extract_metadata(image_path)
        dest_dir = destination_for(result.category, config)
        dest_path = dest_dir / image_path.name

        log.info(
            "%s -> %s (%.2f confidence, label=%s)",
            image_path.name, result.category, result.confidence, result.label,
        )

        if not dry_run:
            shutil.move(str(image_path), str(dest_path))
            airtable.log(metadata, result, str(image_path), str(dest_path))

        counts[result.category] += 1

    log.info(
        "Done. birds=%d animals=%d empty=%d errors=%d",
        counts["bird"], counts["animal"], counts["empty"], counts["error"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify camera trap images and log to Airtable.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Classify and print results without moving files or writing to Airtable.",
    )
    args = parser.parse_args()

    config = Config.load()
    run(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
