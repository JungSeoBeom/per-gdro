import os
import argparse
import torch
import numpy as np

from utils import set_seed
from train_single import train_single
from train_bilevel import run_bilevel_optimization
from test import run_comparison_or_eval
from data.data_loader import get_all_loaders
from models import algorithm_attributes


DATASET_CHOICES = ['synthetic', 'cmnist']
ALGORITHM_CHOICES = list(algorithm_attributes.keys())

def main():
    parser = argparse.ArgumentParser(description='Distributionally Robust Optimization Experiments')

    # -------------------------------------------------------------------------
    # 1. Data Settings
    # -------------------------------------------------------------------------
    parser.add_argument('-d', '--dataset', choices=DATASET_CHOICES, required=True, 
                        help='Dataset name (e.g., synthetic, cmnist)')
    parser.add_argument('--data_dir', type=str, default='D:\seobeom', 
                        help='Root directory containing datasets')
    parser.add_argument('--n_groups', type=int, default=4, 
                        help='Number of groups (will be auto-detected if None)')
    parser.add_argument('--batch_size', type=int, default=None, 
                        help='Batch size (None for full-batch on tabular data, required for images)')
    
    # -------------------------------------------------------------------------
    # 2. Algorithm Settings
    # -------------------------------------------------------------------------
    parser.add_argument('-a', '--algorithm', choices=ALGORITHM_CHOICES, required=True, 
                        help='Algorithm to run')
    parser.add_argument('--bilevel', action='store_true', 
                        help='Run bilevel hyperparameter optimization (HPO) using skopt')

    # -------------------------------------------------------------------------
    # 3. Model Architecture & Optimization
    # -------------------------------------------------------------------------
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 regularization)')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of hidden layers (MLP only)')
    parser.add_argument('--hsize', type=int, default=32, help='Hidden unit size (MLP only)')
    parser.add_argument('--eta_min_ratio', type=float, default=0.0, help='Cosine annealing LR min ratio')

    # -------------------------------------------------------------------------
    # 4. Algorithm Specific Hyperparameters
    # -------------------------------------------------------------------------
    # GroupDRO
    parser.add_argument('--eta_q', type=float, default=0.01, help='GroupDRO step size for q update')
    
    # PerGroupDRO / Robust PGD
    parser.add_argument('--pgd_steps', type=int, default=5, help='Number of PGD steps for inner maximization')
    parser.add_argument('--rho', type=float, default=1.0, help='Constraint radius (rho) for PerGroupDRO')
    parser.add_argument('--eps', type=float, nargs='+', default=[0.1], 
                        help='Per-group epsilon values (e.g. --eps 0.1 0.1 0.2 0.2)')

    # -------------------------------------------------------------------------
    # 5. System & Logging
    # -------------------------------------------------------------------------
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu/mps)')
    parser.add_argument('--track_curve', action='store_true', help='Save training curve plot')

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    set_seed(args.seed)
    print("="*60)
    print(f" Experiment: {args.algorithm} on {args.dataset}")
    print(f" Mode: {'Bilevel HPO' if args.bilevel else 'Single Run'}")
    print(f" Device: {args.device} | Seed: {args.seed}")
    print("="*60)

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------
    # get_all_loaders returns dicts: {'mixed': loader, 'by_group': [loaders], ...}
    train_loaders, val_loaders, test_loaders = get_all_loaders(args)
    
    # Auto-detect Metadata from loaders
    # 1. Number of Groups
    detected_n_groups = train_loaders['n_groups']
    if args.n_groups is None or args.n_groups != detected_n_groups:
        print(f"[Info] Setting n_groups to {detected_n_groups} (detected from data).")
        args.n_groups = detected_n_groups
        
    # 2. Input Dimension (For MLP)
    # Fetch one batch to check shape
    try:
        sample_batch = next(iter(train_loaders['mixed']))
        sample_x = sample_batch[0] # (x, y, g)
        
        if len(sample_x.shape) == 2: # Tabular [Batch, Dim]
            args.input_dim = sample_x.shape[1]
            print(f"[Info] Detected Input Dimension: {args.input_dim}")
        else: # Image [Batch, Channel, H, W]
            # ResNet/CNN handles this, but set to None or specific value if needed
            args.input_dim = None 
            print(f"[Info] Detected Image Data: {sample_x.shape}")
            
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    # 3. Adjust Epsilon List (if length mismatch)
    # If user provided 1 epsilon but we have 4 groups, expand it.
    if len(args.eps) == 1:
        args.eps = args.eps * args.n_groups
    elif len(args.eps) != args.n_groups:
        print(f"[Warning] Length of eps ({len(args.eps)}) != n_groups ({args.n_groups}).")
        # You might want to raise error or handle it. For now, just warning.

    # -------------------------------------------------------------------------
    # Training Phase
    # -------------------------------------------------------------------------
    model_for_eval = None

    if args.bilevel:

        run_bilevel_optimization(
            args, 
            train_loaders, 
            val_loaders, 
            test_loaders
        )

        model_for_eval = None
        
    else:

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        args.result_dir = os.path.join("results", args.dataset, args.algorithm, f"single_seed{args.seed}_{timestamp}")
        model, results = train_single(
            args, 
            train_loaders, 
            val_loaders, 
            test_loaders, 
            device=args.device
        )
        model_for_eval = model
    # -------------------------------------------------------------------------
    # Test / Comparison Phase
    # -------------------------------------------------------------------------

    run_comparison_or_eval(
        args, 
        test_loader_dict=test_loaders, 
        device=args.device, 
        current_model=model_for_eval
    )

    print('\n[Experiment Finished]')

if __name__ == '__main__':
    main()


