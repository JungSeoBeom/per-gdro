import os
import json
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from utils import set_seed, evaluate_loader
from train_single import train_single
from per_gdro import PerGroupDRO

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_best_plot(train_results, save_path):
    curve = train_results.get('train_curve', [])
    plt.figure(figsize=(8, 6))
    plt.plot(curve, label='Training Loss')
    plt.title(f"Best Model Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_optimization_history_plot(func_vals, save_path):
    """
    Plots the history of function evaluations (Val Obj) over trials.
    """
    n_calls = len(func_vals)
    iterations = range(1, n_calls + 1)
    
    # Calculate accumulated minimum (Best so far)
    best_so_far = np.minimum.accumulate(func_vals)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, func_vals, 'bo', alpha=0.4, label='Trial Objective')
    plt.step(iterations, best_so_far, 'r-', linewidth=2, where='post', label='Best Found')
    
    plt.title("Hyperparameter Optimization History")
    plt.xlabel("Trial (Iteration)")
    plt.ylabel("Validation Objective")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_lhs_rhs_plot(train_results, save_path):
    lhs = train_results.get('lhs_history', [])
    rhs = train_results.get('rhs_history', [])
    term1 = train_results.get('rhs_t1_history', [])
    term2 = train_results.get('rhs_t2_history', [])
    term3 = train_results.get('rhs_t3_history', [])
    
    lhs = [v for v in lhs if not (v is None or np.isnan(v))]
    rhs = [v for v in rhs if not (v is None or np.isnan(v))]
    term1 = [v for v in term1 if not (v is None or np.isnan(v))]
    term2 = [v for v in term2 if not (v is None or np.isnan(v))]    
    term3 = [v for v in term3 if not (v is None or np.isnan(v))]   
    
    if not lhs or not rhs: return

    plt.figure(figsize=(8, 6))
    plt.plot(lhs, label='LHS (Robust Loss)', alpha=0.8)
    plt.plot(rhs, label='RHS (Bound)', alpha=0.8, linestyle='--')
    
    if term1: plt.plot(term1, label='Term1 (E_q[L])', alpha=0.6, linestyle=':')
    if term2: plt.plot(term2, label='Term2 (Var/Dev)', alpha=0.6, linestyle=':')
    if term3: plt.plot(term3, label='Term3 (Robustness)', alpha=0.6, linestyle=':')

    plt.title("LHS vs RHS Bound Tracking")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_val_group_losses(model, val_group_loaders, device, loss_fn):
    model.eval()
    losses = []
    with torch.no_grad():
        for loader in val_group_loaders:
            if hasattr(loader.dataset, "__len__") and len(loader.dataset) == 0:
                losses.append(0.0)
                continue
            
            total_loss = 0.0
            total_count = 0
            
            for batch in loader:
                if len(batch) == 3: x, y, _ = batch
                elif len(batch) == 2: x, y = batch
                else: continue
                
                x = x.to(device)
                y = y.to(device)
                
                out = model(x)
                if out.dim() > 1 and out.shape[1] == 1: out = out.squeeze(-1)
                
                batch_loss = loss_fn(out, y)
                bs = x.size(0)
                
                total_loss += batch_loss.item() * bs
                total_count += bs
            
            if total_count > 0:
                losses.append(total_loss / total_count)
            else:
                losses.append(0.0)
                
    return torch.tensor(losses, device=device)

def calculate_upper_level_objective(args, model, val_group_loaders, val_ratios, device):
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    L_val = get_val_group_losses(model, val_group_loaders, device, criterion)
    val_metrics = { "L_plain": L_val.cpu().numpy().tolist() }
    
    if args.algorithm == 'ERM':
        obj = torch.dot(torch.tensor(val_ratios, device=device), L_val).item()
    elif args.algorithm == 'GroupDRO':
        obj = L_val.max().item()
    elif args.algorithm == 'PerGroupDRO':
        alpha_ = 0.01
        beta_ = 0.01
        eps_vec = np.array(args.eps)
        sum_eps = np.sum(eps_vec)
        rho = args.rho
        avg_loss = torch.dot(torch.tensor(val_ratios, device=device), L_val).item()
        obj = avg_loss + alpha_*np.exp(-sum_eps) + beta_*np.exp(-rho)
    else: obj = float('inf')
    return obj, val_metrics

best_val_obj = float('inf')

def run_bilevel_optimization(args, train_loader_dict, val_loader_dict, test_loader_dict):
    print("\n[Bilevel] Starting Hyperparameter Optimization...")
    device = args.device
    
    val_group_loaders = val_loader_dict['by_group']
    group_counts = val_loader_dict['group_counts']
    
    if torch.is_tensor(group_counts):
        val_ratios = (group_counts / group_counts.sum()).cpu().numpy()
    else:
        gc = np.array(group_counts)
        val_ratios = gc / gc.sum()

    # -----------------------------------------------------------
    # [Mod] Construct Search Space based on Algorithm & Dataset
    # -----------------------------------------------------------
    space = []
    
    # Check if we should use Tabular specific params (MLP)
    is_tabular = args.dataset in ['synthetic', 'compas', 'adult', 'insurance']

    # Case 1: PerGroupDRO -> ONLY optimize Rho and Epsilon
    if args.algorithm == 'PerGroupDRO':
        # [Req] Only Rho and Eps (LR, WD, etc. are fixed from args)
        space.append(Real(1e-8, 0.5, name='rho'))
               
        if args.dataset == 'synthetic':# [Req] Eps range changed to 0.0 - 0.5
            for i in range(args.n_groups): 
                space.append(Real(0.0, 2.0, name=f'eps{i}'))
                eps_max = 2.0
        elif args.dataset == 'cmnist':
            for i in range(args.n_groups):
                space.append(Real(0.0, 0.05, name=f'eps{i}'))
                eps_max = 0.05
        else:
            for i in range(args.n_groups):
                space.append(Real(0.0, 1.0, name=f'eps{i}'))
                eps_max = 1.0
    # Case 2: Other Algorithms (ERM, GroupDRO) -> Optimize LR, WD, etc.
    
    # Standard HPO parameters
    space.extend([
        Real(1e-4, 1e-2, prior="log-uniform", name="lr"),
        Real(1e-5, 1e-1, prior="log-uniform", name="weight_decay"),
        Real(0.0, 0.5, name="eta_min_ratio"),
    ])
    
    # [Req] Add MLP params ONLY for tabular data
    if is_tabular:
        space.insert(0, Integer(16, 128, name="hsize"))
        space.insert(0, Integer(1, 3, name="n_layers"))
        
    if args.algorithm == 'GroupDRO':
        space.append(Real(1e-3, 1.0, prior="log-uniform", name='eta_q'))

    # -----------------------------------------------------------
    # [Mod] Construct Initial Points (x0) matching the space
    # -----------------------------------------------------------
    x0 = []
    
    # 1. Init Points for PerGroupDRO (Rho, Eps...)
    #args.algorithm == 'PerGroupDRO':
    # Create varying starting points for rho and eps
    # Point 1: rho=1.0, eps=0.1

    init_lrs = (10 ** np.random.uniform(np.log10(1e-4), np.log10(1e-2), size=10)).tolist()
    init_wds = (10 ** np.random.uniform(np.log10(1e-5), np.log10(1e-1), size=10)).tolist()
    init_eta_mins = np.random.uniform(0.0, 0.5, size=10).tolist()
    
    init_layers = [2, 2, 2, 3, 2, 2, 2, 3, 2, 2]
    init_hsizes = [128, 32, 64, 32, 128, 32, 64, 32, 128, 64]
    x0 = []
    for i in range(10):
        
        pt=([np.random.rand()*0.5] + [eps_max/np.random.randint(1,11) for _ in range(args.n_groups)])
        
        # 2. Init Points for Others (LR, WD...)

               
        # (A) MLP Params
        if is_tabular:
            pt.append(init_layers[i])
            pt.append(init_hsizes[i])
        
        # (B) Common Params
        pt.append(init_lrs[i])
        pt.append(init_wds[i])
        pt.append(init_eta_mins[i])
        
        # (C) GroupDRO Param
        if args.algorithm == 'GroupDRO':
            pt.append(args.eta_q if args.eta_q else 0.01) # eta_q default
        
        x0.append(pt)

    global trial_idx; trial_idx = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("results", args.dataset, args.algorithm, f"bilevel_seed{args.seed}_{timestamp}")
    plot_dir = os.path.join(result_dir, "plots")
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(plot_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)
    trial_csv_path = os.path.join(result_dir, "trials.csv")

    print(f"[Log] Results will be saved to: {result_dir}")

    with open(trial_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # [Mod] Added "test_groups" to header
        writer.writerow(["iter", "val_obj", "test_avg", "test_worst", "test_groups"] + [d.name for d in space])

    @use_named_args(space)
    def objective(**params):
        global trial_idx, best_val_obj
        trial_idx += 1
        
        # Apply params to args
        for k, v in params.items():
            if k.startswith('eps'): continue
            setattr(args, k, v)
            
        if args.algorithm == 'PerGroupDRO':
            # Collect eps values from params
            args.eps = [params[f'eps{i}'] for i in range(args.n_groups)]

        print(f"\n[Trial {trial_idx}] Params: {params}")
        
        # Run Training
        model, train_results = train_single(args, train_loader_dict, val_loader_dict, test_loader_dict, device)
        
        # Calculate Objective
        val_obj, val_metrics = calculate_upper_level_objective(args, model, val_group_loaders, val_ratios, device)
        
        # Evaluate on Test Set
        t_avg, t_worst, t_pg = evaluate_loader(model, test_loader_dict['mixed'], args.n_groups, device)
        
        print(f"   -> Val Obj: {val_obj:.4f} | Test Avg: {t_avg:.4f} | Test Worst: {t_worst:.4f}")
       
        with open(trial_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            # [Mod] Save per-group accuracy as string
            pg_str = str([round(acc, 4) for acc in t_pg])
            writer.writerow([trial_idx, val_obj, t_avg, t_worst, pg_str] + [params[d.name] for d in space])

        if val_obj < best_val_obj:
            print(f"   [!] New Best Found! (Obj: {best_val_obj:.4f} -> {val_obj:.4f})")
            best_val_obj = val_obj
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pt"))
            
            save_best_plot(train_results, os.path.join(plot_dir, "best_model_curve.png"))
            
            if args.algorithm == 'PerGroupDRO':
                save_lhs_rhs_plot(train_results, os.path.join(plot_dir, "best_model_lhs_rhs.png"))

            best_info = {
                "iter": trial_idx, "params": params, "val_obj": val_obj,
                "val_metrics": val_metrics,
                "test_metrics": {"avg": t_avg, "worst": t_worst, "per_group": t_pg},
                "group_weights_history": train_results.get('group_weights', []),
                "lhs_history": train_results.get('lhs_history', []),
                "rhs_history": train_results.get('rhs_history', []),
                "rhs_t1_history": train_results.get('rhs_t1_history', []),
                "rhs_t2_history": train_results.get('rhs_t2_history', []),
                "rhs_t3_history": train_results.get('rhs_t3_history', [])
            }
            with open(os.path.join(result_dir, "best_results.json"), "w") as f:
                json.dump(best_info, f, indent=4, cls=NumpyEncoder)
        return val_obj

    print(f"Starting gp_minimize with {len(x0)} initial points...")
    
    if args.dataset == 'synthetic':
        n_calls=100
    elif args.algorithm == 'PerGroupDRO':
        n_calls=30
    else: n_calls=50

    res = gp_minimize(func=objective, dimensions=space, n_calls=n_calls, n_initial_points=0, x0=x0, acq_func="EI", random_state=args.seed)
    
    print("\n[Bilevel] Optimization Finished."); print(f"Best Val Obj: {res.fun:.4f}")
    
    # Save Optimization History
    save_optimization_history_plot(res.func_vals, os.path.join(plot_dir, "optimization_history.png"))
    print(f" -> Optimization history plot saved to: {plot_dir}")

