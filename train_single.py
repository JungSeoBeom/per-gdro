import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models import define_model
from erm import ERM
from groupdro import GroupDRO
from per_gdro import PerGroupDRO
from utils import evaluate_loader

# --------------------------------------------------------------------------
# Plotting Helper Functions (X-axis: Epoch)
# --------------------------------------------------------------------------
def save_loss_plot(train_curve, save_path):
    if not train_curve: return
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_curve) + 1), train_curve, label='Training Loss', marker='.')
    plt.title("Training Loss Curve (Per Epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_group_weights_plot(weight_history, save_path):
    if not weight_history: return
    weights = np.array(weight_history) # shape: (epochs, n_groups)
    
    plt.figure(figsize=(10, 6))
    epochs = range(1, weights.shape[0] + 1)
    for i in range(weights.shape[1]):
        plt.plot(epochs, weights[:, i], label=f'Group {i}', marker='.')
    
    plt.title("Group Weights History (p_dist, Avg per Epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_lhs_rhs_plot(results, save_path):
    # Extract histories
    lhs = results.get('lhs_history', [])
    rhs = results.get('rhs_history', [])
    
    # Filter valid values (remove NaNs/None)
    # Assuming synced lengths since we append per epoch
    valid_indices = [i for i, v in enumerate(lhs) if v is not None and not np.isnan(v)]
    if not valid_indices: return
    
    lhs_valid = [lhs[i] for i in valid_indices]
    rhs_valid = [rhs[i] for i in valid_indices]
    epochs_valid = [valid_indices[i] + 1 for i in range(len(valid_indices))]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_valid, lhs_valid, label='LHS (Robust Loss)', alpha=0.8, marker='.')
    plt.plot(epochs_valid, rhs_valid, label='RHS (Theoretical Bound)', alpha=0.8, linestyle='--', marker='.')
    
    plt.title("LHS vs RHS Bound Tracking (Avg per Epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --------------------------------------------------------------------------
# Main Train Single Function
# --------------------------------------------------------------------------
def train_single(args, train_loader_dict, val_loader_dict, test_loader_dict, device):
    """
    Runs a single training experiment.
    """
    # 1. Define Model
    input_dim = None
    if args.dataset in ['synthetic', 'compas', 'adult', 'insurance']:
        x_sample, _, _ = train_loader_dict['mixed'].dataset[0]
        input_dim = x_sample.shape[0]

    model = define_model(args, input_dim=input_dim).to(device)
    
    # 2. Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 3. Define Trainer
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    if args.algorithm == 'ERM':
        trainer = ERM(model, loss_fn, optimizer, device)
    elif args.algorithm == 'GroupDRO':
        # [Fix] Explicitly passed device as keyword arg to prevent mix-up
        trainer = GroupDRO(model, optimizer, n_groups=args.n_groups, eta_q=args.eta_q, device=device, loss_type='bce')
    elif args.algorithm == 'PerGroupDRO':
        trainer = PerGroupDRO(model, loss_fn, optimizer, device)
    else:
        raise NotImplementedError
    # 4. Training Loop
    train_loader = train_loader_dict['mixed']
    group_counts = train_loader_dict['group_counts']
    
    if torch.is_tensor(group_counts):
        group_ratios = (group_counts / group_counts.sum()).cpu().numpy()
    else:
        gc = np.array(group_counts)
        group_ratios = gc / gc.sum()

    n_groups = args.n_groups
    
    # Result Containers (Epoch-level)
    train_results = {
        'train_curve': [],       # Avg loss per epoch
        'group_weights': [],     # Avg p_dist per epoch
        'lhs_history': [],       # Avg LHS per epoch
        'rhs_history': [],       # Avg RHS per epoch
        'rhs_t1_history': [],
        'rhs_t2_history': [],
        'rhs_t3_history': []
    }

    print(f"\n[Train] Start training {args.algorithm} for {args.n_epochs} epochs...")
    
    for epoch in range(args.n_epochs):
        model.train()
        
        # Epoch Accumulators
        epoch_losses = []
        epoch_weights = []
        epoch_lhs = []
        epoch_rhs = []
        epoch_t1 = []
        epoch_t2 = []
        epoch_t3 = []
        
        for batch in train_loader:
            # Prepare arguments for train_step
            step_kwargs = {}
            if args.algorithm == 'PerGroupDRO':
                step_kwargs['group_ratios'] = group_ratios
                step_kwargs['eta_p'] = 1 - (epoch**(1/2))/args.n_epochs
                step_kwargs['rho_p'] = args.rho
                step_kwargs['phi'] = 'chi2'
                step_kwargs['dro_config_q'] = {
                    "epsilon": args.eps if isinstance(args.eps, list) else [0.0]*n_groups,
                    "pgd_steps": args.pgd_steps
                }
                step_kwargs['track_bound'] = True 
            
            # Update Step
            if args.algorithm == 'PerGroupDRO':
                loss_val, _, p_dist = trainer.train_step(batch, **step_kwargs)
                
                # Append step values to epoch list
                epoch_losses.append(loss_val)
                epoch_weights.append(p_dist)
                
                # Get last appended values from trainer history
                # (Trainer appends per step, so we take the last one)
                if trainer.lhs_history: epoch_lhs.append(trainer.lhs_history[-1])
                if trainer.rhs_history: epoch_rhs.append(trainer.rhs_history[-1])
                if trainer.rhs_t1_history: epoch_t1.append(trainer.rhs_t1_history[-1])
                if trainer.rhs_t2_history: epoch_t2.append(trainer.rhs_t2_history[-1])
                if trainer.rhs_t3_history: epoch_t3.append(trainer.rhs_t3_history[-1])
            
            elif args.algorithm == 'GroupDRO':
                loss_val, _, p_dist = trainer.train_step(batch, **step_kwargs)
                epoch_losses.append(loss_val)
                epoch_weights.append(p_dist)
                
            else: # ERM
                loss_val = trainer.train_step(batch)
                epoch_losses.append(loss_val)

        # --- End of Epoch Aggregation ---
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        train_results['train_curve'].append(avg_loss)
        
        if epoch_weights:
            avg_weights = np.mean(np.array(epoch_weights), axis=0)
            train_results['group_weights'].append(avg_weights)
            
        if args.algorithm == 'PerGroupDRO':
            # Safe mean calculation ignoring NaNs
            def safe_mean(lst):
                valid = [x for x in lst if x is not None and not np.isnan(x)]
                return np.mean(valid) if valid else np.nan

            train_results['lhs_history'].append(safe_mean(epoch_lhs))
            train_results['rhs_history'].append(safe_mean(epoch_rhs))
            train_results['rhs_t1_history'].append(safe_mean(epoch_t1))
            train_results['rhs_t2_history'].append(safe_mean(epoch_t2))
            train_results['rhs_t3_history'].append(safe_mean(epoch_t3))

        # Logging
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.n_epochs:
            print(f"   Epoch {epoch+1}/{args.n_epochs} | Loss: {avg_loss:.4f}")

    # 5. Final Evaluation & Saving Results
    if hasattr(args, 'result_dir') and args.result_dir:
        save_dir = args.result_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # (A) Save Plots
        save_loss_plot(train_results['train_curve'], os.path.join(save_dir, "train_loss.png"))
        
        if args.algorithm in ['GroupDRO', 'PerGroupDRO']:
            save_group_weights_plot(train_results['group_weights'], os.path.join(save_dir, "group_weights.png"))
            
        if args.algorithm == 'PerGroupDRO':
            save_lhs_rhs_plot(train_results, os.path.join(save_dir, "lhs_rhs_bound.png"))
        
        # (B) Evaluation Metrics
        t_avg, t_worst, t_pg = evaluate_loader(model, test_loader_dict['mixed'], n_groups, device)
        v_avg, v_worst, v_pg = evaluate_loader(model, val_loader_dict['mixed'], n_groups, device)
        
        final_metrics = {
            "val": {"avg": v_avg, "worst": v_worst, "per_group": v_pg},
            "test": {"avg": t_avg, "worst": t_worst, "per_group": t_pg},
            "params": vars(args)
        }
        
        # (C) Save Metrics JSON
        with open(os.path.join(save_dir, "final_metrics.json"), "w") as f:
            json.dump(final_metrics, f, indent=4, cls=NumpyEncoder)
            
        print(f"[Train Single] Plots and metrics saved to: {save_dir}")

    return model, train_results