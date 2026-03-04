from __future__ import annotations

import argparse
from pathlib import Path

from .classifier import ConstantClassifier, RuleBasedWildlifeClassifier
from .config import PipelineConfig
from .pipeline import WildlifePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify and organize camera trap images.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving.")
    parser.add_argument("--flat", action="store_true", help="Do not recurse into subdirectories.")
    parser.add_argument(
        "--dry-run-class",
        default=None,
        help="Use a constant label classifier for dry runs/tests (e.g., Deer).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        copy_files=args.copy,
        recursive=not args.flat,
    )

    classifier = (
        ConstantClassifier(label=args.dry_run_class)
        if args.dry_run_class
        else RuleBasedWildlifeClassifier()
    )

    pipeline = WildlifePipeline(config=config, classifier=classifier)
    results = pipeline.run()

    print(f"Processed {len(results)} image(s).")
    for result in results:
        print(
            f"{result.source.name} -> {result.label} "
            f"({result.confidence:.2%}) -> {result.destination}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
