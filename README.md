# Per-Group DRO (Per-GDRO)

This repository contains the implementation used in our **Optimization Letters** submission.

- Submission snapshot (tag): `v1.0-optimlett`

## Overview

Implemented methods:
- ERM
- GroupDRO (Sagawa et al., 2020)
- Per-GDRO (Per-Group Distributionally Robust Optimization)

Supported Datasets:
- **Synthetic**: Multidimensional Gaussian mixture with distribution shifts.
- **CMNIST**: Modified Colored MNIST with variable spurious correlations and geometric shifts.

The repository includes scripts for:
- Single-run training: `train_single.py`
- Bilevel Hyperparameter Optimization (HPO): `train_bilevel.py`

## Requirements

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Quickstart

### 1. Training (Single Run)

Run a single experiment with specified hyperparameters.

**Synthetic Dataset (ERM):**
```bash
python run.py --dataset synthetic --algorithm ERM --n_epochs 50 --lr 0.01
```

**CMNIST (Per-Group DRO):**
```bash
# Note: Data will be downloaded/generated in the specified --data_dir
python run.py --dataset cmnist --algorithm PerGroupDRO --data_dir "./data" --n_epochs 20 --rho 0.1 --eps 0.1 0.1 0.1 0.1
```

### 2. Bilevel Optimization (HPO)

Run hyperparameter optimization using `scikit-optimize` to find the best configuration.

```bash
python run.py --dataset synthetic --algorithm GroupDRO --bilevel --n_epochs 100 --lr 0.01
```

## Repository Structure

- `run.py`: Main entry point for experiments.
- `train_single.py`: Training loop for a single configuration.
- `train_bilevel.py`: Bilevel optimization loop.
- `models.py`: Model architectures (MLP for Synthetic, ResNet18 for CMNIST).
- `per_gdro.py`: Implementation of the Per-Group DRO algorithm.
- `groupdro.py`: Implementation of GroupDRO.
- `erm.py`: Implementation of Empirical Risk Minimization.
- `data/`: Data loaders and generators.
  - `synthetic_data.py`: Generates synthetic Gaussian data.
  - `cmnist_data.py`: Handles Modified colored MNIST.

## Reproducibility

Random seeds can be set via the `--seed` argument (default: 42) to ensure deterministic behavior.

## Contact

- Seobeom Jung — nasnaga1628@gmail.com
