import os
import glob
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from models import define_model, MLP
from utils import evaluate_loader

ALGORITHMS = ['ERM', 'GroupDRO', 'PerGroupDRO']

def _get_latest_run_path(dataset, algorithm, seed):
    base_dir = os.path.join("results", dataset, algorithm)
    if not os.path.exists(base_dir): return None
    
    pattern = os.path.join(base_dir, f"bilevel_seed{seed}_*")
    candidates = glob.glob(pattern)
    if not candidates: return None
    
    candidates.sort(reverse=True) # Latest first
    latest_dir = candidates[0]
    
    ckpt_path = os.path.join(latest_dir, "checkpoints", "best_model.pt")
    json_path = os.path.join(latest_dir, "best_results.json")
    
    if os.path.exists(ckpt_path) and os.path.exists(json_path):
        return latest_dir, ckpt_path, json_path
    return None

def _load_model_from_file(ckpt_path, json_path, args, device):
    """
    Load model architecture and weights.
    Reconstructs MLP with correct hyperparameters from json if needed.
    """
    with open(json_path, 'r') as f: 
        info = json.load(f)
    
    params = info.get('params', {})
    
    # 1. Tabular (MLP) - Needs specific hsize/n_layers from saved run
    if args.dataset in ['synthetic', 'compas', 'adult']:
        # Fallback to args if not in json (e.g. single run without HPO)
        n_layers = params.get('n_layers', args.n_layers)
        hsize = params.get('hsize', args.hsize)
        input_dim = getattr(args, 'input_dim', None)
        
        if input_dim is None:
            raise ValueError("args.input_dim is None. Cannot reconstruct MLP.")
            
        model = MLP(input_dim, hsize, n_layers).to(device)
        
    # 2. Image (ResNet/CNN) - Architecture usually fixed per dataset
    else:
        model = define_model(args, input_dim=None).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return model, info

def _plot_comparison_bar(dataset, seed, metrics_dict, output_dir):
    """
    metrics_dict: { 'ERM': [acc_g0, acc_g1, ...], ... }
    """
    if not metrics_dict: return
    n_groups = len(next(iter(metrics_dict.values())))
    groups = np.arange(n_groups)
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, accs in metrics_dict.items():
        offset = width * multiplier
        rects = ax.bar(groups + offset, accs, width, label=algo)
        multiplier += 1

    ax.set_ylabel('Accuracy')
    ax.set_title(f'Per-Group Accuracy Comparison ({dataset}, Seed {seed})')
    ax.set_xticks(groups + width)
    ax.set_xticklabels([f'G{i}' for i in groups])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "group_accuracy_comparison.png")
    plt.savefig(save_path)
    plt.close()

def _plot_weight_histories(dataset, seed, history_dict, output_dir):
    """
    history_dict: { 'GroupDRO': [[w_g0, ...], ...], 'PerGroupDRO': ... }
    """
    for algo, hist in history_dict.items():
        if not hist or len(hist) == 0: continue
        
        hist_np = np.array(hist) # [Steps, Groups]
        steps = np.arange(len(hist_np))
        n_groups = hist_np.shape[1]

        plt.figure(figsize=(8, 5))
        for g in range(n_groups):
            plt.plot(steps, hist_np[:, g], label=f'Group {g}')
        
        plt.title(f'{algo} Group Weights (p/q) History')
        plt.xlabel('Step')
        plt.ylabel('Weight')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"{algo}_weight_history.png")
        plt.savefig(save_path)
        plt.close()

def run_comparison_or_eval(args, test_loader_dict, device, current_model=None):
    test_loader = test_loader_dict['mixed']
    n_groups = args.n_groups
    
    # Comparison Output Directory
    comp_out_dir = os.path.join("results", args.dataset, "test_comparison")
    os.makedirs(comp_out_dir, exist_ok=True)

    print("\n" + "="*60)
    print(" [Test Phase] Checking for existing models to compare...")
    
    found_paths = {}
    missing = []
    
    for algo in ALGORITHMS:
        res = _get_latest_run_path(args.dataset, algo, args.seed)
        if res: found_paths[algo] = res
        else: missing.append(algo)
            
    # Condition: Compare only if ALL models are found
    if not missing:
        print(f" -> Found latest models for all algorithms (Seed {args.seed}).")
        print(" -> Running Comparative Evaluation...\n")
        print(f"{'Algorithm':<15} | {'Avg Acc':<10} | {'Worst Acc':<10} | {'Per-Group Acc'}")
        print("-" * 90)
        
        per_group_metrics = {}
        weight_histories = {}

        best_algo = None
        best_worst_acc = -1.0
        
        for algo in ALGORITHMS:
            run_dir, ckpt_path, json_path = found_paths[algo]
            try:
                model, info = _load_model_from_file(ckpt_path, json_path, args, device)
                
                avg, worst, per_group = evaluate_loader(model, test_loader, n_groups, device)
                
                pg_str = ", ".join([f"{a:.3f}" for a in per_group])
                print(f"{algo:<15} | {avg:.4f}     | {worst:.4f}     | [{pg_str}]")
                
                per_group_metrics[algo] = per_group
                if 'group_weights_history' in info:
                    weight_histories[algo] = info['group_weights_history']

                if worst > best_worst_acc:
                    best_worst_acc = worst; best_algo = algo
            except Exception as e:
                print(f"{algo:<15} | Error: {e}")

        print("-" * 90)
        print(f"Winner (Worst Group Acc): {best_algo} ({best_worst_acc:.4f})")
        
        # Save Plots
        _plot_comparison_bar(args.dataset, args.seed, per_group_metrics, comp_out_dir)
        _plot_weight_histories(args.dataset, args.seed, weight_histories, comp_out_dir)
        print(f" -> Comparison plots saved to: {comp_out_dir}")
        print("="*60 + "\n")
        
    else:
        # Fallback: Evaluate only current
        print(f" -> Missing models for: {missing}")
        print(f" -> Skipping comparison. Evaluating current target: {args.algorithm}")
        print("-" * 60)
        
        target_model = current_model
        
        # If current model is None (e.g. from bilevel run), try loading the one just saved
        if target_model is None:
            res = _get_latest_run_path(args.dataset, args.algorithm, args.seed)
            if res:
                _, ckpt_path, json_path = res
                print(f" -> Loading best model from: {ckpt_path}")
                target_model, _ = _load_model_from_file(ckpt_path, json_path, args, device)
        
        if target_model is not None:
            avg, worst, per_group = evaluate_loader(target_model, test_loader, n_groups, device)
            
            print(f"Algorithm: {args.algorithm}")
            print(f"Avg Acc:   {avg:.4f}")
            print(f"Worst Acc: {worst:.4f}")
            pg_str = ", ".join([f"{a:.3f}" for a in per_group])
            print(f"Per-Group: [{pg_str}]")
        else:
            print("[Error] No model available in memory or disk to evaluate.")
        print("="*60 + "\n")


