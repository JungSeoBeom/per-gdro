import torch
import torch.nn as nn

class ERM:
    """
    Empirical Risk Minimization (ERM) Trainer.
    Processes a single batch per step.
    """
    def __init__(self, model, loss_fn, optimizer, device="cpu"):
        self.model = model.to(device)
        self.loss_fn = loss_fn  # Usually reduction='mean'
        self.optimizer = optimizer
        self.device = device

    def train_step(self, batch):
        """
        Single training step for ERM.
        
        Args:
            batch: Tuple of (inputs, targets, groups) or (inputs, targets) from DataLoader
            
        Returns:
            loss (float): The loss value for this batch
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Unpack batch safely
        if len(batch) == 3:
            inputs, targets, groups = batch
        elif len(batch) == 2:
            inputs, targets = batch
            # groups are ignored in ERM
        else:
            raise ValueError(f"ERM train_step received batch of length {len(batch)}. Expected 2 or 3.")
        
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        logits = self.model(inputs)
        
        # Adjust dimensions for BCE loss (B, 1) -> (B)
        if logits.dim() > 1 and logits.shape[1] == 1:
            logits = logits.squeeze(-1)
            
        loss = self.loss_fn(logits, targets)
        if loss.dim() > 0:
            loss = loss.mean()
            
        loss.backward()
        self.optimizer.step()
        
        return loss.item()