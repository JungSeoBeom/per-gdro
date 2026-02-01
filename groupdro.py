import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupDRO:
    """
    Group Distributionally Robust Optimization (Group DRO).
    References: Sagawa et al., "Distributionally Robust Neural Networks for Group Shifts", ICLR 2020.
    
    Maintains a distribution 'q' over groups and updates it to maximize the weighted loss,
    while the model parameters 'theta' minimize the weighted loss.
    """
    def __init__(self, model, optimizer, n_groups, eta_q, device, loss_type='bce'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.n_groups = int(n_groups)
        self.eta_q = float(eta_q) # Step size for q update
        self.device = device
        self.loss_type = loss_type

        # Initialize q uniformly
        self.q = torch.ones(self.n_groups, device=self.device) / self.n_groups

        # History for logging
        self.group_losses_history = []
        self.group_weights_history = []

    def _per_sample_loss(self, logits, targets):
        """Computes loss per sample (reduction='none')."""
        if self.loss_type == 'bce':
            targets = targets.float()
            return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        elif self.loss_type == 'ce':
            return F.cross_entropy(logits, targets.long(), reduction="none")
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    def _update_q(self, group_losses, present_mask):
        """
        Exponentiated Gradient Ascent on q.
        Only updates q for groups present in the current batch.
        """
        # q <- q * exp(eta * loss)
        # Only apply to groups present in this batch to avoid numerical issues
        
        idx = torch.where(present_mask)[0]
        if idx.numel() == 0:
            return

        # Update
        self.q[idx] = self.q[idx] * torch.exp(self.eta_q * group_losses[idx])
        
        # Normalize
        self.q = self.q / (self.q.sum() + 1e-12)

    def train_step(self, batch):
        """
        Single training step for GroupDRO using a mixed batch.
        
        Args:
            batch: Tuple of (inputs, targets, groups)
        
        Returns:
            loss (float): The weighted robust loss value
            group_losses (np.array): Average loss per group in this batch
            q (np.array): Current group weights
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Unpack batch
        if len(batch) == 3:
            inputs, targets, groups = batch
        else:
            raise ValueError(f"GroupDRO requires groups in batch (len=3). Got len {len(batch)}")

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        groups = groups.to(self.device)

        # 1. Forward pass
        logits = self.model(inputs)
        if logits.dim() > 1 and logits.shape[1] == 1:
            logits = logits.squeeze(-1)

        # 2. Compute per-sample loss
        per_sample_losses = self._per_sample_loss(logits, targets)

        # 3. Aggregate losses per group
        group_losses = torch.zeros(self.n_groups, device=self.device)
        present_mask = torch.zeros(self.n_groups, dtype=torch.bool, device=self.device)

        for gid in range(self.n_groups):
            mask = (groups == gid)
            if mask.sum() > 0:
                group_losses[gid] = per_sample_losses[mask].mean()
                present_mask[gid] = True
            else:
                # If group is missing in batch, loss is 0 (won't affect q update logic)
                group_losses[gid] = 0.0

        # 4. Update q (distribution over groups)
        with torch.no_grad():
            self._update_q(group_losses, present_mask)

        # 5. Compute robust loss (weighted sum)
        # L_robust = sum(q_g * L_g)
        # Note: q is normalized over all groups, even those not present in batch.
        robust_loss = torch.dot(self.q, group_losses)

        # 6. Backward
        robust_loss.backward()
        self.optimizer.step()

        # 7. Logging
        self.group_losses_history.append(group_losses.detach().cpu().numpy())
        self.group_weights_history.append(self.q.detach().cpu().numpy())

        return robust_loss.item(), group_losses.detach().cpu().numpy(), self.q.detach().cpu().numpy()