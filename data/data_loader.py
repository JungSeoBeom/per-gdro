import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


from .synthetic_data import generate_synthetic_data
try:
    from .waterbirds_data import get_waterbirds_loader_dict
except ImportError:
    pass
try:
    from .cmnist_data import get_cmnist_data
except ImportError:
    pass

from .compas_data import load_compas_data


# -------------------------------------------------------------------------
# Common Helper: Tensor -> Loader Dict
# -------------------------------------------------------------------------
def make_loader_dict_from_tensors(X, y, g, batch_size=None, shuffle=True, n_groups_force=None):
    """
    Converts (X, y, g) tensors into the standardized loader dictionary format.
    Used for small tabular datasets (Synthetic, COMPAS, Adult).
    """
    if n_groups_force is None:
        n_groups = int(g.max().item()) + 1 if len(g) > 0 else 0
    else:
        n_groups = n_groups_force
       
    N = len(X)
    if batch_size is None:
        batch_size = N # Full batch training
       
    # 1. Mixed Loader
    ds = TensorDataset(X, y, g)
    mixed_loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
   
    # 2. Group Loaders
    group_loaders = []
    group_counts = torch.zeros(n_groups)
   
    for gid in range(n_groups):
        mask = (g == gid)
        count = mask.sum().item()
        group_counts[gid] = count
       
        if count > 0:
            Xg, yg = X[mask], y[mask]
            # Group loader batch size: use same batch_size or full group
            bs = min(batch_size, count)
            g_ds = TensorDataset(Xg, yg)
            g_loader = DataLoader(g_ds, batch_size=bs, shuffle=shuffle, drop_last=False)
            group_loaders.append(g_loader)
        else:
            # Dummy loader for empty group
            ds_dummy = TensorDataset(torch.empty(0, X.shape[1]), torch.empty(0))
            group_loaders.append(DataLoader(ds_dummy, batch_size=1))
           
    return {
        "mixed": mixed_loader,
        "by_group": group_loaders,
        "n_groups": n_groups,
        "group_counts": group_counts
    }

# -------------------------------------------------------------------------
# Main Interface: get_all_loaders
# -------------------------------------------------------------------------
def get_all_loaders(args):
    """
    Returns (train_loaders, val_loaders, test_loaders)
    Each is a dict: {'mixed': ..., 'by_group': ..., 'n_groups': ..., 'group_counts': ...}
    """
   
    # 1. Synthetic
    if args.dataset == 'synthetic':
        # Generate Tensors
        data = generate_synthetic_data(args)
        (X_tr, y_tr, g_tr, X_va, y_va, g_va, X_te, y_te, g_te) = data
       
        # Wrap in Loaders
        # Synthetic is small, so we can use full batch if batch_size is None
        train_loaders = make_loader_dict_from_tensors(X_tr, y_tr, g_tr, args.batch_size, shuffle=True)
        val_loaders = make_loader_dict_from_tensors(X_va, y_va, g_va, args.batch_size, shuffle=False)
        test_loaders = make_loader_dict_from_tensors(X_te, y_te, g_te, args.batch_size, shuffle=False)
       
        return train_loaders, val_loaders, test_loaders

    # 2. COMPAS
    elif args.dataset == 'compas':
        # load_compas_data returns ((X,y,g), (X,y,g), (X,y,g))
        if 'load_compas_data' not in globals():
             raise NotImplementedError("compas_data.py is missing or load_compas_data not imported.")
       
        train_data, val_data, test_data = load_compas_data(args.data_dir)
       
        train_loaders = make_loader_dict_from_tensors(*train_data, batch_size=args.batch_size, shuffle=True)
        val_loaders = make_loader_dict_from_tensors(*val_data, batch_size=args.batch_size, shuffle=False)
        test_loaders = make_loader_dict_from_tensors(*test_data, batch_size=args.batch_size, shuffle=False)
       
        return train_loaders, val_loaders, test_loaders
    
    # 3. CMNIST (Added)
    elif args.dataset == 'cmnist':
        if 'get_cmnist_data' not in globals():
             raise NotImplementedError("cmnist_data.py is missing or not imported.")
        
        # Data Generation (Tensors)
        # cmnist_data.py returns ((X,y,g), (X,y,g), (X,y,g)) tuple
        # But wait, get_cmnist_data returns 3 tuples of tensors directly.
        train_data, val_data, test_data = get_cmnist_data(args)
        
        # Wrap in Loaders
        train_loaders = make_loader_dict_from_tensors(*train_data, batch_size=args.batch_size, shuffle=True)
        val_loaders = make_loader_dict_from_tensors(*val_data, batch_size=args.batch_size, shuffle=False)
        test_loaders = make_loader_dict_from_tensors(*test_data, batch_size=args.batch_size, shuffle=False)
        
        return train_loaders, val_loaders, test_loaders

    # 4. Waterbirds
    elif args.dataset == 'waterbirds':
        if 'get_waterbirds_loader_dict' not in globals():
            raise NotImplementedError("waterbirds_data.py is missing or function not imported.")
       
        # Data directory path
        #wb_dir = os.path.join(args.data_dir, 'waterbird_complete95_forest2water2')

        wb_dir = "D:\seobeom\waterbird_complete95_forest2water2\waterbird_complete95_forest2water2"
       
        # Image datasets need explicit batch size
        bs = args.batch_size if args.batch_size is not None else 64
       
        train_loaders = get_waterbirds_loader_dict(wb_dir, bs, split='train', shuffle=True)
        val_loaders = get_waterbirds_loader_dict(wb_dir, bs, split='val', shuffle=False)
        test_loaders = get_waterbirds_loader_dict(wb_dir, bs, split='test', shuffle=False)
       
        return train_loaders, val_loaders, test_loaders

    # 4. CelebA
    elif args.dataset == 'celeba':
        # Similar structure to Waterbirds, to be implemented
        raise NotImplementedError("CelebA not implemented yet.")

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

# Compatibility alias
create_data_loaders = make_loader_dict_from_tensors