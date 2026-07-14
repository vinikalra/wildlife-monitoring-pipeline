# Wildlife Monitoring Pipeline

Sorts camera trap images from a national park into `bird` / `animal` / `empty`,
moves each file into the matching folder, and logs metadata for every image
(station, roll, capture time, category, confidence) to an Airtable base.

## How it works

Classification is a two-stage pipeline:

1. **Detection** ([src/detector.py](src/detector.py)) — Microsoft's
   [MegaDetector](https://github.com/microsoft/CameraTraps) (via
   `PytorchWildlife`) decides whether a frame is empty, or contains an
   `animal`, `person`, or `vehicle`, with a well-calibrated confidence score.
   This is a model trained specifically on camera trap imagery, so it's what
   drives the empty-vs-not-empty decision.
2. **Species hint** ([src/classifier.py](src/classifier.py)) — for frames
   MegaDetector confirms contain an animal, the detected region is cropped
   and run through a torchvision ResNet50 (ImageNet-1k) to guess whether it's
   a bird or something else. ImageNet has no real African savanna species
   classes, so this is only a loose label hint (logged as `Predicted Label`)
   — it does **not** affect the empty/animal/bird category, which is
   controlled entirely by MegaDetector's detection plus the bird/mammal
   class mapping in `classifier.py`.

`person` and `vehicle` detections, and frames with no detection at all, are
categorized as `empty` and routed to the rejected folder.

For each image, [src/metadata.py](src/metadata.py) also parses the camera
station/roll/frame from the filename (expects the convention
`SEASON_STATION_ROLL_FRAME.jpg`, e.g. `S1_B04_R1_PICT0012.JPG`) and reads the
capture timestamp from EXIF data, if present.

[src/pipeline.py](src/pipeline.py) orchestrates all of it: classify each
image in the source folder, move it into the matching destination folder,
and write one row per image to Airtable.

## Folder layout

```
data/
  download/              # drop new, unsorted images here
  classified/
    birds/                # -> bird
    animals/               # -> animal
  rejected/                # -> empty (no detection, or person/vehicle)
```

All paths are configurable via `.env` (see below).

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

| Variable | Description |
|---|---|
| `AIRTABLE_API_KEY` | Personal access token with `data.records:read/write` + `schema.bases:read/write` |
| `AIRTABLE_BASE_ID` | Base ID (`app...`) |
| `AIRTABLE_TABLE_NAME` | Table name (default `Camera Trap Images`) |
| `SOURCE_DIR` | Folder to read unsorted images from (default `data/download`) |
| `BIRDS_DIR` / `ANIMALS_DIR` / `REJECTED_DIR` | Destination folders |
| `CONFIDENCE_THRESHOLD` | Min. confidence for the ResNet50 species hint to be trusted as "bird" (default `0.3`) |
| `DETECTION_CONFIDENCE_THRESHOLD` | Min. MegaDetector confidence to count as a real detection (default `0.2`) |

The Airtable table needs these fields: `File Name`, `Season`, `Station`,
`Roll`, `Frame`, `Captured At`, `Category`, `Predicted Label`, `Confidence`,
`Original Path`, `New Path`, `Processed At`.

## Usage

```bash
cd src
../venv/bin/python pipeline.py --dry-run   # preview: classify without moving files or writing to Airtable
../venv/bin/python pipeline.py             # move files + log to Airtable
```

## Data

No training happens in this repo — both models are used pretrained and
off-the-shelf. Two different datasets are relevant, for different reasons:

**What the models were trained on** (explains their behavior/limits):
- **MegaDetector** ([src/detector.py](src/detector.py)) is trained by its
  maintainers on a large aggregated corpus of camera trap images contributed
  by partner organizations across many ecosystems worldwide, labeled only for
  `animal` / `person` / `vehicle` presence — not species. That's why it's
  reliable for the empty-vs-not-empty call but has no concept of species.
- **The ResNet50 species hint** ([src/classifier.py](src/classifier.py)) is
  trained on **ImageNet-1k** (~1.28M images, 1,000 general object
  categories — everyday objects, dog breeds, 59 bird species, a handful of
  other animals). It has no African savanna species classes, which is why
  `Predicted Label` is only a loose hint (see Known limitations).

**The data this pipeline actually runs against:**
- [data/download/](data/download) — a staging folder for new, unsorted
  images; the pipeline reads from here (configurable via `SOURCE_DIR`).
- [data/S1/](data/S1) — a local, ungitignored-from-tracking raw archive of
  406,526 images across 167 camera station folders (~243GB), organized by
  station code (e.g. `B04`, `C03`) in a way consistent with a
  Snapshot-Serengeti-style camera trap survey. It has no accompanying label
  file in this repo — it's unlabeled raw footage, not a training set. Images
  get pulled from here into `data/download` to be processed.
- Neither of these is committed to git — `data/` is excluded via
  `.gitignore` because of size.

## Known limitations

- **Species labels are a hint, not an ID.** The classifier is a generic
  ImageNet model, not fine-tuned on this park's species — expect the
  `Predicted Label` field to occasionally show nonsensical guesses (e.g.
  "lens cap") for animals ImageNet has no equivalent class for. The
  bird/animal/empty *category* is unaffected by this.
- **CPU inference is slow at scale.** MegaDetector takes roughly 1-1.5s per
  image on CPU. Fine for batches in the hundreds; the full `data/S1` archive
  (400k+ images) would take many hours without a GPU or batching changes.
- **`data/` is not checked into git** (see `.gitignore`) — it's local-only,
  including the large `S1` archive and the `download`/`classified`/`rejected`
  working folders.

## Tech stack

Python · PyTorch / torchvision (ResNet50) · MegaDetector via `PytorchWildlife`
· Pillow (EXIF) · `pyairtable` · `python-dotenv`
