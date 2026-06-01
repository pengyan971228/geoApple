# GeoApple-Seg

**Geometry-Aware Apple Instance Segmentation with RGB-D Fusion and Amodal Reasoning**

Source code and training scripts accompanying the paper submitted to *Image and Vision Computing* (Elsevier).

## Overview

GeoApple-Seg is an RGB-D instance segmentation framework for apple detection in orchards. It introduces:

- **Depth Dropout regularizer** — stochastically zeros the depth channel during training, the strongest contributor in our ablation study (−6.5 IoU when removed).
- **4-channel early-fusion backbone** built on YOLO-seg, jointly modeling RGB appearance and SfM/MVS depth.
- **Dual modal / amodal segmentation heads** for occlusion-aware reasoning.
- **Precision-favoring operating point** that achieves a favorable P-R Pareto trade-off compared to YOLOv8/v11 baselines under heavy occlusion (53% mean occlusion ratio on the test set).

On the public **AmodalAppleSize_RGB-D** dataset (Gené-Mola et al., 2023), GeoApple-Seg attains **IoU = 0.7753 ± 0.0117** and **Precision = 0.8659 ± 0.0133** across 5 random seeds, with statistically significant precision gains over YOLOv11s-seg (paired *t*-test, *p* = 0.0002).

## Repository Layout

```
geoApple/
├── src/
│   ├── data_module/        # RGB-D dataset loader, augmentation
│   ├── model_module/       # Depth encoder, fusion, modal/amodal heads, GeoApple model
│   ├── trainer_module/     # Training loop, losses, schedulers
│   └── utils/              # Logging, metrics, visualization helpers
├── scripts/
│   ├── train_geoapple.py           # Main training entry point
│   ├── train_baseline.py           # YOLOv8/v11 baseline training
│   ├── train_ablation.py           # Ablation study runs (A1–A4)
│   ├── evaluate.py                 # Standard evaluation
│   ├── eval_pr_curve.py            # P-R curve comparison (supports --replot)
│   ├── eval_occlusion_stratified.py# Occlusion-stratified evaluation
│   ├── eval_diameter.py            # Diameter MAE evaluation
│   ├── run_multi_seed.py           # Multi-seed training + paired t-test
│   ├── run_yolo_baselines.py       # Batch baseline training
│   ├── visualize_failures.py       # Qualitative failure-case visualization
│   ├── make_composites.py          # Paper-figure composites
│   └── setup_server.sh             # One-shot Linux + CUDA + conda setup
├── pyproject.toml
├── CITATION.cff
├── LICENSE
└── README.md
```

## Requirements

- Python ≥ 3.12
- CUDA 12.8+ (for RTX 5090 / Blackwell) or CUDA 12.1+ (Ampere / Ada / Hopper)
- PyTorch ≥ 2.4
- See `pyproject.toml` for the full dependency list.

## Installation

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
git clone https://github.com/pengyan971228/geoApple.git
cd geoApple
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Or with standard pip:

```bash
git clone https://github.com/pengyan971228/geoApple.git
cd geoApple
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For a fresh Linux server, `scripts/setup_server.sh` handles CUDA + conda + dependencies in one shot.

## Dataset

GeoApple-Seg is trained and evaluated on the publicly available **AmodalAppleSize_RGB-D** dataset:

> Gené-Mola, J., Sanz-Cortiella, R., Rosell-Polo, J. R., Escolà, A., & Gregorio, E. (2023).
> *AmodalAppleSize_RGB-D dataset: RGB-D images of apple trees annotated with modal and amodal segmentation masks for fruit detection, visibility and size estimation.*
> Data in Brief, 50, 109498. <https://doi.org/10.1016/j.dib.2023.109498>

Download the dataset from the Zenodo link referenced in the Data in Brief article, then update the data root path in your training config.

## Quick Start

Train GeoApple-Seg from scratch:

```bash
python scripts/train_geoapple.py \
  --data-root /path/to/AmodalAppleSize_RGB-D \
  --epochs 300 \
  --batch-size 16 \
  --depth-dropout 0.5 \
  --seed 42
```

Reproduce multi-seed evaluation (seeds 42, 123, 2024, 7, 314):

```bash
python scripts/run_multi_seed.py --data-root /path/to/AmodalAppleSize_RGB-D
```

Generate the P-R curve figure used in the paper:

```bash
python scripts/eval_pr_curve.py --replot
```

## Citation

If you use this code, please cite both the paper and the dataset:

```bibtex
@article{yan2026geoapple,
  author  = {Yan, Peng},
  title   = {{GeoApple-Seg}: Geometry-Aware Apple Instance Segmentation with {RGB-D} Fusion and Amodal Reasoning},
  journal = {Image and Vision Computing},
  year    = {2026},
  note    = {Under review}
}

@article{genemola2023dataset,
  author  = {Gen{\'e}-Mola, Jordi and Sanz-Cortiella, Ricardo and Rosell-Polo, Joan R. and Escol{\`a}, Alexandre and Gregorio, Eduard},
  title   = {{AmodalAppleSize\_RGB-D} dataset: {RGB-D} images of apple trees annotated with modal and amodal segmentation masks for fruit detection, visibility and size estimation},
  journal = {Data in Brief},
  volume  = {50},
  pages   = {109498},
  year    = {2023},
  doi     = {10.1016/j.dib.2023.109498}
}
```

## License

Released under the [MIT License](LICENSE).

## Contact

Peng Yan — Longdong University

For questions or issues, please open a GitHub issue.
