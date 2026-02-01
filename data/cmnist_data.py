import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import TensorDataset

def get_cmnist_data(args):
    """
    Generate Modified CMNIST dataset as described in the paper:
    "Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets"
    
    Paper Settings [cite: 451-456, 490-493]:
    1. Classes: Binary (0-4 -> 0, 5-9 -> 1).
    2. Label Noise: Flip probability 0.25.
    3. Groups: 4 groups based on (Color, Label).
       - g0: Green, y=0
       - g1: Green, y=1
       - g2: Red,   y=0
       - g3: Red,   y=1
    4. Spurious Correlation (Color Bias):
       - Train: y=0 is 80% Red, y=1 is 80% Green.
       - Test:  y=0 is 10% Red, y=1 is 90% Red. (Distribution Shift)
    5. Intra-group Shift (Modified CMNIST):
       - The minority group from training (y=1, Red) is rotated by 90 degrees
         ONLY in the Test set.
    
    Data Sizes [cite: 454-455]:
       - Train: 30,000
       - Val:   10,000
       - Test:  20,000
    """
    
    # 1. Load Original MNIST (Train + Test) & Pool them
    # We pool 60k + 10k = 70k images to split exactly as per paper
    data_dir = args.data_dir
    mnist_train = datasets.MNIST(data_dir, train=True, download=True)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True)

    X_all = torch.cat([mnist_train.data, mnist_test.data], dim=0).float() / 255.0 # (70000, 28, 28)
    y_all = torch.cat([mnist_train.targets, mnist_test.targets], dim=0)           # (70000,)

    # 2. Split into Train (30k) / Val (10k) / Test (20k)
    # Note: We use the first 60k for train/val/test split logic to be consistent with random shuffling
    # But strictly following paper counts:
    n_train = 30000
    n_val = 10000
    n_test = 20000
    
    # Shuffle indices
    perm = torch.randperm(len(X_all), generator=torch.Generator().manual_seed(args.seed))
    
    idx_tr = perm[:n_train]
    idx_va = perm[n_train : n_train + n_val]
    idx_te = perm[n_train + n_val : n_train + n_val + n_test]

    # 3. Process each split
    # Configs for Color Ratios (Prob of Red)
    # Train: y=0 -> 80% Red, y=1 -> 20% Red 
    conf_tr = {0: 0.8, 1: 0.2} 
    # Val:   50:50 Balanced 
    conf_va = {0: 0.5, 1: 0.5}
    # Test:  y=0 -> 10% Red, y=1 -> 90% Red 
    conf_te = {0: 0.1, 1: 0.9}

    train_data = _make_cmnist_split(X_all[idx_tr], y_all[idx_tr], conf_tr, 
                                    label_flip_p=0.25, rotate_minority=False, seed=args.seed)
    
    val_data = _make_cmnist_split(X_all[idx_va], y_all[idx_va], conf_va, 
                                  label_flip_p=0.25, rotate_minority=False, seed=args.seed+1)
    
    #  Apply rotation shift to minority group (y=1, Red) in Test set
    test_data = _make_cmnist_split(X_all[idx_te], y_all[idx_te], conf_te, 
                                   label_flip_p=0.25, rotate_minority=True, seed=args.seed+2)

    return train_data, val_data, test_data

def _make_cmnist_split(images, targets, color_probs, label_flip_p=0.25, rotate_minority=False, seed=42):
    """
    Helper to construct (X, y, g) tensors.
    """
    n = len(images)
    rng = np.random.RandomState(seed)
    
    # 1. Binarize Labels: 0-4 -> 0, 5-9 -> 1 [cite: 453]
    y_binary = (targets >= 5).long()
    
    # 2. Label Noise: Flip with prob 0.25 [cite: 456]
    flip_mask = torch.from_numpy(rng.rand(n) < label_flip_p)
    y_final = torch.where(flip_mask, 1 - y_binary, y_binary)
    
    # 3. Assign Color (Spurious Attribute)
    # color_probs[lbl] = Probability of being RED (1)
    # A = 0 (Green), A = 1 (Red)
    color_assign = torch.zeros(n, dtype=torch.long)
    
    for y_c in [0, 1]:
        mask = (y_final == y_c)
        count = mask.sum().item()
        if count > 0:
            # Probability of Red for this class
            p_red = color_probs[y_c]
            # Randomly assign Red (1) or Green (0)
            is_red = (rng.rand(count) < p_red).astype(int)
            color_assign[mask] = torch.tensor(is_red, dtype=torch.long)
            
    # 4. Define Groups
    # We use the structure: Group = Color * 2 + Label
    # g0: Green(0), y=0 -> 0
    # g1: Green(0), y=1 -> 1
    # g2: Red(1),   y=0 -> 2
    # g3: Red(1),   y=1 -> 3 (The Minority Group in Train)
    groups = color_assign * 2 + y_final

    # 5. Apply Modified CMNIST Shift (Rotation) 
    # "Shifted CMNIST: Rotate all images in the minority group (label 1, red) by 90 deg at test time"
    # Minority group is y=1, Red => Group 3.
    images_processed = images.clone()
    
    if rotate_minority:
        # Find indices of Group 3
        mask_g3 = (groups == 3)
        if mask_g3.sum() > 0:
            # Rotate 90 degrees (k=1) in spatial dims (last two dims)
            # images shape: (N, 28, 28)
            images_to_rot = images_processed[mask_g3]
            # torch.rot90 rotates in the plane formed by dims.
            # For (N, H, W), dims are (1, 2)
            rotated = torch.rot90(images_to_rot, k=1, dims=[1, 2])
            images_processed[mask_g3] = rotated

    # 6. Colorize Images (2 Channels)
    # Channel 0: Green, Channel 1: Red
    # If Red (A=1): Ch0=0, Ch1=Image
    # If Green(A=0): Ch0=Image, Ch1=0
    # Shape: (N, 2, 28, 28)
    
    images_stacked = torch.stack([images_processed, images_processed], dim=1) # (N, 2, 28, 28)
    
    # Zero out the other channel
    # If color=1 (Red), we want Ch0 (Green) to be 0.
    # If color=0 (Green), we want Ch1 (Red) to be 0.
    # Logic: x[i, 1-color[i], :, :] = 0
    
    for i in range(n):
        c = color_assign[i].item()
        # if c=1 (Red), zero out index 0 (Green)
        # if c=0 (Green), zero out index 1 (Red)
        channel_to_zero = 1 - c
        images_stacked[i, channel_to_zero, :, :] = 0.0

    return images_stacked, y_final.float(), groups