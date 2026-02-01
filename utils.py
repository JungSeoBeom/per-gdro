# seed setting, Logging utils
import sys
import os
import numpy as np
import torch
import csv
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Logger:
    def __init__(self, path=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.file = open(path, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
            self.file.flush()

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

class CSVBatchLogger(object):
    def __init__(self, csv_path, n_groups, mode='w'):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
       
        columns = ['epoch', 'batch']
        for idx in range(n_groups):
            columns.append(f'avg_loss_group:{idx}')
       
        self.path = csv_path
        self.file = open(csv_path, mode, newline='')
        self.columns = columns
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
       
        if mode == 'w':
            self.writer.writeheader()

    def log(self, epoch, batch, stats_dict):
        row = stats_dict.copy()
        row['epoch'] = epoch
        row['batch'] = batch
        self.writer.writerow(row)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate_loader(model, dataloader, n_groups, device):
    """
    Evaluate accuracy using a DataLoader.
    Efficient for large datasets (e.g., Image) as it doesn't load everything into memory.
    """
    model.eval()
    correct_by_group = torch.zeros(n_groups, device=device)
    total_by_group = torch.zeros(n_groups, device=device)
   
    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch (x, y, g)
            if len(batch) == 3:
                x, y, g = batch
            else:
                raise ValueError(f"DataLoader must yield (x, y, g). Got {len(batch)} elements.")

            x = x.to(device)
            y = y.to(device)
            g = g.to(device)
           
            logits = model(x)
           
            # Dimension check (B, 1) -> (B)
            if logits.dim() > 1 and logits.shape[1] == 1:
                logits = logits.squeeze(-1)
           
            # Binary Classification Assumption
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct = (preds == y).float()
           
            # Count per group
            for gi in range(n_groups):
                mask = (g == gi)
                if mask.sum() > 0:
                    correct_by_group[gi] += correct[mask].sum()
                    total_by_group[gi] += mask.sum()
                   
    # Calculate Metrics
    # Avoid division by zero
    per_group_acc = (correct_by_group / (total_by_group + 1e-12)).cpu().numpy()
   
    total_samples = total_by_group.sum()
    avg_acc = (correct_by_group.sum() / total_samples).item() if total_samples > 0 else 0.0
   
    # Worst group acc (only consider groups that exist in this set)
    valid_groups = total_by_group > 0
    if valid_groups.any():
        worst_acc = (correct_by_group[valid_groups] / total_by_group[valid_groups]).min().item()
    else:
        worst_acc = 0.0
   
    return avg_acc, worst_acc, per_group_acc

# Legacy function (kept for compatibility if needed, but evaluate_loader is preferred)
def evaluate(model, X, y, g, n_groups=None, device=None):
    if device is None:
        device = next(model.parameters()).device
   
    # Wrap in a temporary loader logic for consistency
    # But here we just implement the tensor logic directly
    model.eval()
    with torch.no_grad():
        Xd = X.to(device)
        yd = y.to(device)
        gd = g.to(device)
       
        logits = model(Xd)
        if logits.dim() > 1 and logits.shape[1] == 1:
            logits = logits.squeeze(-1)
           
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct = (preds == yd).float()
       
        avg_acc = correct.mean().item()

        if n_groups is None:
            if len(gd) > 0:
                n_groups = int(gd.max().item()) + 1
            else:
                n_groups = 0

        per_group_acc = []
        for gid in range(n_groups):
            mask = (gd == gid)
            if mask.sum() > 0:
                acc = correct[mask].mean().item()
                per_group_acc.append(acc)
            else:
                per_group_acc.append(0.0)

        worst_acc = min(per_group_acc) if per_group_acc else 0.0

    return avg_acc, worst_acc, per_group_acc