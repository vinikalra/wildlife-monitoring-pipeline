# Wildlife Monitoring Pipeline

## Overview
This project automates the classification and organization of camera trap images featuring animals and birds. The pipeline predicts a species label for each image and places files into output folders named after the detected species.

## Features
- **Animal Detection and Classification:** Uses a pre-trained deep learning model (MobileNetV3 on ImageNet) as a baseline.
- **Automated Folder Organization:** Creates folders by predicted species and moves/copies images there.
- **Confidence Thresholding:** Low-confidence predictions are automatically routed to `Unknown`.
- **Minimal Configuration:** Point at an input and output directory and run.
- **Extensible Design:** Swap in custom classifiers or retrained models.

## End Goal
Provide a robust, scalable workflow for wildlife researchers, conservationists, and hobbyists so they can spend less time sorting files and more time analyzing biodiversity.

## Technologies
- **Python 3.8+**
- **PyTorch / torchvision** (classification model)
- **Pillow / NumPy** (image handling)
- **OpenCV and scikit-learn** listed as project dependencies for broader CV/ML extensions

## Project Structure

```text
src/wildlife_monitoring/
  classifier.py   # model-backed and constant classifiers
  config.py       # pipeline configuration
  organizer.py    # image discovery + move/copy behavior
  pipeline.py     # orchestration
  preprocess.py   # image loading utilities
  cli.py          # command line entry point
tests/
  test_pipeline.py
  test_organizer.py
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Run

```bash
wildlife-pipeline --input-dir ./camera_trap_images --output-dir ./organized_images
```

### Useful Options

```bash
# copy files instead of moving
wildlife-pipeline --input-dir ./in --output-dir ./out --copy

# treat low-confidence predictions as Unknown at a stricter threshold
wildlife-pipeline --input-dir ./in --output-dir ./out --confidence-threshold 0.45

# dry run classifier that labels everything as Deer
wildlife-pipeline --input-dir ./in --output-dir ./out --dry-run-class Deer
```

## How It Works
1. Discover images in the input directory.
2. Load and preprocess each image to RGB.
3. Classify with the model-backed classifier.
4. Map predictions to wildlife-focused labels.
5. Move/copy each image into a species-named folder.

## Notes
- The first model-backed run may download model weights.
- The ImageNet baseline is a starting point; for field deployment, train/fine-tune on camera trap datasets for better species-level accuracy.
