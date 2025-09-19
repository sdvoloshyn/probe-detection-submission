## Overview

YOLO-based probe detection. Includes a simple folder inference script that saves annotated outputs to `output/`. Extensive documentation can be found in report.pdf.


## Setup

Create and activate a virtual environment, then install the project in editable mode.

```bash
python3 -m venv venv
source venv/bin/activate && pip install -e .
source venv/bin/activate
```

Requirements (both are on GitHub):
- `final.pt` at the repository root (trained YOLO weights)
- `configs/final_config.yaml` present (used by the inference script)


## Inference

Run inference on a folder of images (processed one-by-one). Results ("probe" box, or a "no_pred" overlay) are written to `output/`.

```bash
python inference.py /path/to/your/images
```

Optional confidence threshold (default is the best from training; tune if your dataset differs):

```bash
python inference.py /path/to/your/images --conf 0.10
```

Notes:
- The script is hardcoded to use `configs/final_config.yaml` and weights at `final.pt`.


## Repository tree (key items)

- `inference.py`: Folder inference (YOLO). Loads `final.pt`, reads `configs/final_config.yaml`, saves to `output/`.
- `train.py`: Training entrypoint.
- `configs/`
  - `final_config.yaml`: Main config used by training/inference.
  - other configs: Experiment variations.
- `src/`
  - `config.py`: Dataclass configs loader.
  - `trainers/yolo_trainer.py`: YOLO training pipeline and validation selection.
  - `eval/`
    - `op_select.py`: Operating point selection (threshold sweep logic).
    - `viz.py`: Validation visualization helpers.
    - `metrics.py`, `filters.py`, `sweep.py`: Evaluation utilities.
  - `datasets/`
    - `canonical.py`: Canonical dataset representation.
    - `augment.py`: Augmentation helpers.
    - `adapters/yolo_export.py`: Export to Ultralytics YOLO format.
    - `adapters/rcnn_dataset.py`: RCNN dataset adapter (not used by inference).
  - `cv/lomo.py`, `cv/runner.py`: Cross-validation utilities.
  - `utils/run_logger.py`: Run logging utilities.
- `scripts/`
  - `bench_infer.py`, `bench_plot.py`, `visualize_augmentations.py`, `reporting.py`: Utilities for benchmarking and analysis.
- `data/`
  - `prepare_data_single_split.py`, `prepare_data_lomo_splits.py`, `prepare_negatives.py`: Data prep utilities.
- `output/`: Inference outputs (created by `inference.py`).
- `runs/`: Training/eval artifacts (created by training).