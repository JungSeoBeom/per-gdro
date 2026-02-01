import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import set_seed

def generate_synthetic_data(args):
    
    set_seed(args.seed)

    counts = {
        "train": [500, 420, 140, 140],   # 1200
        "valid": [80, 60, 30, 30],       # 200
        "test":  [240, 210, 75, 75],     # 600
    }
   
    base_means = torch.tensor([[5/3, 5/3],
                               [-5/3, -5/3],
                               [5/3, -5/3],
                               [-5/3, 5/3]], dtype=torch.float32)
    
    shift_means = torch.tensor([[0.0, 0.0],
                                [0.0, 0.0],
                                [0.0, 0.0],
                                [-2.0, -0.5]], dtype=torch.float32)
    
    taus = torch.tensor([0.0, 0.0, 0.0, 0.3], dtype=torch.float32)
    
    def _sample(n_counts, use_shift):
        Xs, ys, gs = [], [], []
        with torch.no_grad():
            for g, n_g in enumerate(n_counts):
                Z = base_means[g] + torch.randn(n_g, 2)
                if use_shift:
                    e = shift_means[g] + taus[g] * torch.randn(n_g, 2)
                else:
                    e = torch.zeros(n_g, 2)
                
                Xs.append(Z + e)
                ys.append(torch.full((n_g,), 1.0 if g in (1, 3) else 0.0, dtype=torch.float32))
                gs.append(torch.full((n_g,), g, dtype=torch.long))
        return torch.cat(Xs).float(), torch.cat(ys).float(), torch.cat(gs).long()
    
    X_train, y_train, g_train = _sample(counts["train"], use_shift=True)
    X_valid, y_valid, g_valid = _sample(counts["valid"], use_shift=True)
    X_test, y_test, g_test = _sample(counts["test"], use_shift=False)

    return (X_train, y_train, g_train, X_valid, y_valid, g_valid, X_test, y_test, g_test)