# Per-Group Distributionally Robust Optimization with Learnable Ambiguity Set Sizes via Bilevel Optimization

This repository implements **Per-Group Distributionally Robust Optimization (Per-Group DRO)**, a method for improving worst-group performance by learning group-specific ambiguity set sizes (epsilons) via **Bilevel Optimization**.

It compares the performance of the following algorithms:
*   **ERM** (Empirical Risk Minimization)
*   **GroupDRO** (Group Distributionally Robust Optimization)
*   **PerGroupDRO** (Proposed Method: Per-Group DRO)

## Project Structure

```
Per_GroupDRO/
├── data/                      # Data loading scripts
│   ├── cmnist_data.py         # Colored MNIST dataset loader
│   ├── compas_data.py         # COMPAS dataset loader # not included in the paper
│   ├── waterbirds_data.py     # Waterbirds dataset loader # not included in the paper
│   ├── synthetic_data.py      # Synthetic dataset generation 
│   ├── data_loader.py         # General data loading utilities
│   └── data.py                # Core data handling functions
├── run.py                     # Main entry point for training and evaluation
├── train_single.py            # Training loop for standard single-run experiments
├── train_bilevel.py           # Implementation of Bilevel Optimization for hyperparameter search
├── models.py                  # Neural network architectures (MLP, ResNet, etc.)
├── groupdro.py                # GroupDRO algorithm implementation
├── per_gdro.py                # PerGroupDRO algorithm implementation
├── erm.py                     # ERM algorithm implementation
└── utils.py                   # Utility functions (seeding, logging, etc.)
```

## Requirements

*   Python 3.8+
*   PyTorch
*   NumPy
*   scikit-optimize (for bilevel optimization)
*   pandas

## Usage

Measurement and training are primarily triggered via `run.py`.

### Arguments

*   `-d`, `--dataset`: Dataset to use. Choices: `synthetic`, `compas`, `adult`, `cmnist`.
*   `-a`, `--algorithm`: Algorithm to run. Choices: `ERM`, `GroupDRO`, `PerGroupDRO`.
*   `--bilevel`: Add this flag to run bilevel hyperparameter optimization.
*   `--data_dir`: Root directory containing the datasets (default: `D:\seobeom`).
*   `--n_epochs`: Number of training epochs.
*   `--batch_size`: Batch size for training.
*   `--seed`: Random seed for reproducibility.
*   `--lr`: Learning rate.
*   `--weight_decay`: Weight decay (L2 regularization).

### Examples

Below are example commands to run experiments across different datasets and algorithms.

**1. ERM on Colored MNIST (CMNIST)**
Runs ERM with bilevel optimization on the CMNIST dataset.
```bash
python run.py -d cmnist -a ERM --bilevel --n_epochs 50 --seed 42 --data_dir "D:\seobeom" --batch_size 128
```

**2. GroupDRO on Synthetic Data**
Runs GroupDRO with bilevel optimization on a synthetic dataset.
```bash
python run.py -d synthetic -a GroupDRO --bilevel --n_epochs 50 --seed 42 --data_dir "D:\seobeom" --batch_size 128
```

**3. Per-Group DRO on CMNIST**
Runs Per-Group DRO with bilevel optimization, specifying specific learning rate and weight decay values.
```bash
python run.py -d cmnist -a PerGroupDRO --bilevel --n_epochs 50 --seed 42 --data_dir "D:\seobeom" --batch_size 128 --lr 0.001 --weight_decay 0.1
```
