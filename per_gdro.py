import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cvxpy as cp

class PerGroupDRO:
    """
    Per-Group Distributionally Robust Optimization (Per-Group DRO).
    """
    def __init__(self, model, loss_fn, optimizer, device="cpu"):
        self.model = model.to(device)
        self.loss_fn = loss_fn # Expected to have reduction='none'
        self.optimizer = optimizer
        self.device = device

        self.p_dist = None  
        
        # History for plotting
        self.group_losses_history = []
        self.group_weights_history = []
        
        # LHS/RHS History
        self.lhs_history = []
        self.rhs_history = []
        self.rhs_t1_history = [] 
        self.rhs_t2_history = [] 
        self.rhs_t3_history = [] 

    def solve_w1_dro_pgd_lp(self, inputs, labels, epsilon, steps, clip_min=None, clip_max=None):
        # Fast exit setup
        step_size = 2.0 * epsilon / steps if steps > 0 else 0.0
        
        # Helper: Get loss vector (reduction='none')
        def _get_loss_vec(x, y):
            out = self.model(x)
            if out.dim() > 1 and out.shape[1] == 1: out = out.squeeze(-1)
            return self.loss_fn(out, y)

        # 1. PGD Setup (Eval Mode for Attack Generation & Consistent Comparison)
        was_training = self.model.training
        self.model.eval() 
        
        # Freeze model params for efficiency during attack generation
        prev_req = [param.requires_grad for param in self.model.parameters()]
        for param in self.model.parameters(): param.requires_grad_(False)

        try:
            x0 = inputs.detach()
            
            # [Fix] Calculate Original Loss in Eval Mode (Baseline)
            # This ensures logical consistency (Adv Loss >= Orig Loss)
            with torch.no_grad():
                orig_loss_vec = _get_loss_vec(x0, labels)

            if epsilon <= 0.0 or steps <= 0:
                # If no attack, adv loss is same as orig loss
                # We still need gradients for training, so we re-run forward on x0 after restoring grads
                
                # Restore gradients first
                for param, r in zip(self.model.parameters(), prev_req): param.requires_grad_(r)
                
                # Re-compute to attach graph
                final_loss_vec = _get_loss_vec(x0, labels)
                return final_loss_vec, orig_loss_vec

            # Initialize Best using x0
            adv = x0.detach().clone()
            adv.requires_grad = True
            
            best_loss_vec = orig_loss_vec.clone()
            best_adv = adv.detach().clone()

            # 2. PGD Loop
            for _ in range(int(steps)):
                adv.requires_grad_(True)
                loss_vec = _get_loss_vec(adv, labels)
                loss_mean = loss_vec.mean()
                
                self.model.zero_grad(set_to_none=True)
                loss_mean.backward()

                with torch.no_grad():
                    # Update Best
                    improved = loss_vec > best_loss_vec
                    if improved.any():
                        best_loss_vec[improved] = loss_vec[improved]
                        best_adv[improved] = adv.detach()[improved]

                    g = adv.grad
                    if g is None: break 
                    
                    # Update adv
                    adv = adv + step_size * g.sign()
                    delta = (adv - x0).clamp(-float(epsilon), float(epsilon))
                    adv = x0 + delta
                    
                    if clip_min is not None and clip_max is not None:
                        adv.clamp_(clip_min, clip_max)

            # 3. Final Check (Last Step)
            with torch.no_grad():
                last_loss_vec = _get_loss_vec(adv, labels)
                improved = last_loss_vec > best_loss_vec
                if improved.any():
                    best_adv[improved] = adv.detach()[improved]
                    # best_loss_vec is updated here but it doesn't have gradients.
                    # We will re-compute final_loss_vec below.

            # 4. Final Return (Restore Gradients & Re-compute)
            
            # [Fix] Restore model gradients BEFORE final forward pass
            for param, r in zip(self.model.parameters(), prev_req): param.requires_grad_(r)
            
            # [Fix] Re-compute loss on the best adversarial example to attach the computation graph
            # This ensures 'total_loss.backward()' works in train_step
            final_loss_vec = _get_loss_vec(best_adv, labels)

        finally:
            # Ensure model is put back in original mode and gradients are restored
            if was_training: self.model.train()
            
            # Just in case we exited early or hit an error before step 4
            # (Double check to ensure requires_grad is restored)
            for param, r in zip(self.model.parameters(), prev_req): 
                param.requires_grad_(r)

        return final_loss_vec, orig_loss_vec

    def solve_phi_divergence_primal(self, loss_vec, group_ratios, rho, phi="chi2"):
        n = len(loss_vec)
        q = np.asarray(group_ratios, dtype=np.float64)
        q = q / (np.sum(q) + 1e-12)
        q = np.clip(q, 1e-8, 1.0)

        if rho <= 1e-12: return q

        loss_vec = np.asarray(loss_vec, dtype=np.float64)
        p = cp.Variable(n)
        eps = 1e-8

        if phi == "chi2": div_expr = cp.sum(cp.square(p - q) / (q + eps))
        elif phi == "mod_chi2": div_expr = cp.sum(cp.square(p - q) / (p + eps))
        elif phi == "kl": div_expr = cp.sum(cp.kl_div(p, q))
        elif phi == "burg": div_expr = cp.sum(cp.kl_div(q, p))
        elif phi == "tv": div_expr = 0.5 * cp.norm(p - q, 1)
        elif phi == "hellinger": div_expr = cp.sum(cp.square(cp.sqrt(p + eps) - cp.sqrt(q)))
        else: raise ValueError(f"Unsupported phi-divergence: {phi}")

        constraints = [cp.sum(p) == 1, p >= 0, div_expr <= rho]
        prob = cp.Problem(cp.Maximize(loss_vec @ p), constraints)
        
        try: prob.solve(solver=cp.SCS, verbose=False)
        except: prob.solve(verbose=False)

        if prob.status != cp.OPTIMAL or p.value is None: return q
        return np.asarray(p.value).reshape(-1)

    def train_step(self, batch, group_ratios, eta_p, rho_p, phi, dro_config_q, track_bound=False, lip_loss=1.0):
        self.model.train()
        inputs, labels, groups = batch
        inputs, labels, groups = inputs.to(self.device), labels.to(self.device), groups.to(self.device)
        n_groups = len(group_ratios)

        if self.p_dist is None:
            q0 = torch.from_numpy(np.asarray(group_ratios, dtype=np.float32)).to(self.device)
            self.p_dist = q0 / (q0.sum() + 1e-12)

        # 1. Compute Robust Losses
        group_losses_tilde = []
        group_losses_original = [] 

        for gi in range(n_groups):
            mask = (groups == gi)
            if mask.sum() == 0:
                zero = torch.tensor(0.0, device=self.device)
                group_losses_tilde.append(zero)
                group_losses_original.append(zero)
                continue

            Xg = inputs[mask]
            yg = labels[mask]

            # [Removed] Manual calculation of out_orig here (it was Train mode)
            
            eps_g = float(dro_config_q["epsilon"][gi])
            gcfg = {
                "epsilon": eps_g, 
                "steps": int(dro_config_q["pgd_steps"]),
            }
            
            # [Modified] Receive BOTH losses from PGD function
            l_tilde_vec, l_orig_vec = self.solve_w1_dro_pgd_lp(Xg, yg, **gcfg)
            
            group_losses_tilde.append(l_tilde_vec.mean())
            group_losses_original.append(l_orig_vec.mean())

        L_tilde = torch.stack(group_losses_tilde)
        L_original = torch.stack(group_losses_original)

        # Debug Prints (Now Consistent)
        L_tilde_np = L_tilde.detach().cpu().numpy()
        L_orig_np = L_original.detach().cpu().numpy()
        

        # 2. Solve for p*
        p_star_np = self.solve_phi_divergence_primal(L_tilde_np, group_ratios, rho=rho_p, phi=phi)
        p_star = torch.from_numpy(p_star_np).float().to(self.device)


        # 3. Update p_dist
        self.p_dist = (1.0 - eta_p) * self.p_dist + eta_p * p_star
        self.p_dist = self.p_dist / (self.p_dist.sum() + 1e-12)
        

        # 4. Update Theta
        self.optimizer.zero_grad(set_to_none=True)
        # Note: We minimize the Robust Loss (L_tilde)
        total_loss = torch.dot(self.p_dist, L_tilde)
        total_loss.backward()
        self.optimizer.step()

        # 5. Logging
        self.group_losses_history.append(L_tilde_np)
        self.group_weights_history.append(self.p_dist.detach().cpu().numpy())

        # Track LHS vs RHS Bound
        if track_bound:
            q = torch.from_numpy(group_ratios / (np.sum(group_ratios) + 1e-12)).float().to(self.device)
            
            # EqL uses L_original (Calculated in Eval mode, consistent with L_tilde)
            EqL = torch.dot(q, L_original)
            dev = L_original - EqL
            eps_vec = torch.tensor(dro_config_q["epsilon"], dtype=torch.float32, device=self.device)
            
            term1 = EqL.item()
            term2 = (torch.linalg.norm(p_star, 2) * torch.linalg.norm(dev, 2)).item()
            term3 = (lip_loss * torch.dot(p_star, eps_vec)).item()
            
            rhs = term1 + term2 + term3
            lhs = torch.dot(p_star, L_tilde.detach()).item()
            
            self.lhs_history.append(lhs)
            self.rhs_history.append(rhs)
            self.rhs_t1_history.append(term1)
            self.rhs_t2_history.append(term2)
            self.rhs_t3_history.append(term3)
        else:
            # (Nan filling code...)
            self.lhs_history.append(np.nan)
            self.rhs_history.append(np.nan)
            self.rhs_t1_history.append(np.nan)
            self.rhs_t2_history.append(np.nan)
            self.rhs_t3_history.append(np.nan)

        return float(total_loss.item()), L_tilde_np, self.p_dist.detach().cpu().numpy()