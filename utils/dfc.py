from __future__ import absolute_import

import torch
import torch.nn.functional as F
from collections import deque, defaultdict
import numpy as np


class DynamicFeedbackController(object):
    """
    Dynamic Feedback Controller (DFC) for adaptive loss weighting.
    
    Dynamically adjusts loss weights (lambda_IDG, lambda_PMR) based on:
    - Gradient variance/norm variance
    - Prediction entropy
    
    Update rule: lambda_t = lambda_0 * (1 + beta * Var(∇L_total))
    Optionally modulated by entropy (increase weights when entropy is high).
    """
    
    def __init__(self, lambda_idg_0=1.0, lambda_pmr_0=1.0, lambda_dfc_0=1.0, beta=0.1, 
                 lambda_max=10.0, use_entropy_modulation=True, 
                 entropy_window_size=100, grad_window_size=100,
                 entropy_weight=0.5,
                 entropy_target=None, gradvar_target=None,
                 w_entropy=None, w_grad=None):
        """
        Initialize Dynamic Feedback Controller.
        
        Args:
            lambda_idg_0: Base weight for IDG loss
            lambda_pmr_0: Base weight for PMR loss
            lambda_dfc_0: Base weight for DFC loss
            beta: Scaling factor for gradient variance term
            lambda_max: Maximum allowed weight value (clamping upper bound)
            use_entropy_modulation: If True, modulate weights by entropy
            entropy_window_size: Window size for running entropy estimate
            grad_window_size: Window size for running gradient variance estimate
            entropy_weight: Weight for entropy modulation (0-1)
            entropy_target: Target entropy value for DFC loss (default: 2.0)
            gradvar_target: Target gradient variance for DFC loss (default: 0.1)
            w_entropy: Weight for entropy term in DFC loss (default: 0.1)
            w_grad: Weight for gradient variance term in DFC loss (default: 0.1)
        """
        # Base weights
        self.lambda_idg_0 = lambda_idg_0
        self.lambda_pmr_0 = lambda_pmr_0
        self.lambda_dfc_0 = lambda_dfc_0
        
        # Current adaptive weights (initialized to base weights)
        self.lambda_idg_t = lambda_idg_0
        self.lambda_pmr_t = lambda_pmr_0
        self.lambda_dfc_t = lambda_dfc_0
        
        # Hyperparameters
        self.beta = beta
        self.lambda_max = lambda_max
        self.use_entropy_modulation = use_entropy_modulation
        self.entropy_weight = entropy_weight
        
        # DFC loss hyperparameters (with backwards-compatible defaults)
        self.entropy_target = entropy_target if entropy_target is not None else 2.0
        self.gradvar_target = gradvar_target if gradvar_target is not None else 0.1
        self.w_entropy = w_entropy if w_entropy is not None else 0.1
        self.w_grad = w_grad if w_grad is not None else 0.1
        
        # Internal state for running estimates
        # Use deque for efficient windowed statistics
        self.entropy_history = deque(maxlen=entropy_window_size)
        self.grad_norm_history = deque(maxlen=grad_window_size)
        
        # Running statistics
        self.running_entropy = 0.0
        self.running_grad_variance = 0.0
        self.running_grad_mean = 0.0
        self.update_count = 0
        
        # DFC loss value (computed after update)
        self.current_dfc_loss = 0.0
        
        # Per-domain statistics for MS-IDG alpha control
        # Track entropy and loss per domain to adaptively adjust mixing ratios
        self.domain_entropy_history = defaultdict(lambda: deque(maxlen=entropy_window_size))
        self.domain_loss_history = defaultdict(lambda: deque(maxlen=entropy_window_size))
        self.running_domain_entropy = defaultdict(float)
        self.running_domain_loss = defaultdict(float)
    
    def update(self, entropy_value, grad_norm, domain_ids=None, domain_entropies=None, domain_losses=None):
        """
        Update internal statistics with new entropy and gradient norm values.
        
        Args:
            entropy_value: Current prediction entropy (scalar or tensor)
            grad_norm: Current gradient norm (scalar or tensor)
            domain_ids: Optional [B] tensor of domain IDs for per-domain tracking
            domain_entropies: Optional [B] tensor of per-sample entropies
            domain_losses: Optional [B] tensor of per-sample losses (e.g., L_IDG per sample)
        """
        # Convert to scalar if tensor
        if torch.is_tensor(entropy_value):
            entropy_value = entropy_value.item()
        if torch.is_tensor(grad_norm):
            grad_norm = grad_norm.item()
        
        # Update history
        self.entropy_history.append(entropy_value)
        self.grad_norm_history.append(grad_norm)
        
        # Update per-domain statistics if provided
        if domain_ids is not None and domain_entropies is not None:
            domain_ids_np = domain_ids.cpu().numpy() if torch.is_tensor(domain_ids) else domain_ids
            domain_entropies_np = domain_entropies.cpu().numpy() if torch.is_tensor(domain_entropies) else domain_entropies
            
            for domain_id, entropy_val in zip(domain_ids_np, domain_entropies_np):
                domain_id = int(domain_id)
                self.domain_entropy_history[domain_id].append(float(entropy_val))
                # Update running average
                if len(self.domain_entropy_history[domain_id]) == 1:
                    self.running_domain_entropy[domain_id] = float(entropy_val)
                else:
                    alpha_ema = 0.1
                    self.running_domain_entropy[domain_id] = (
                        (1 - alpha_ema) * self.running_domain_entropy[domain_id] + 
                        alpha_ema * float(entropy_val)
                    )
        
        if domain_ids is not None and domain_losses is not None:
            domain_ids_np = domain_ids.cpu().numpy() if torch.is_tensor(domain_ids) else domain_ids
            domain_losses_np = domain_losses.cpu().numpy() if torch.is_tensor(domain_losses) else domain_losses
            
            for domain_id, loss_val in zip(domain_ids_np, domain_losses_np):
                domain_id = int(domain_id)
                self.domain_loss_history[domain_id].append(float(loss_val))
                # Update running average
                if len(self.domain_loss_history[domain_id]) == 1:
                    self.running_domain_loss[domain_id] = float(loss_val)
                else:
                    alpha_ema = 0.1
                    self.running_domain_loss[domain_id] = (
                        (1 - alpha_ema) * self.running_domain_loss[domain_id] + 
                        alpha_ema * float(loss_val)
                    )
        
        # Update running statistics
        self.update_count += 1
        
        # Running entropy: exponential moving average
        if self.update_count == 1:
            self.running_entropy = entropy_value
        else:
            alpha_entropy = 0.1  # EMA decay factor
            self.running_entropy = (1 - alpha_entropy) * self.running_entropy + alpha_entropy * entropy_value
        
        # Running gradient variance: compute variance over window
        if len(self.grad_norm_history) > 1:
            grad_norms = np.array(self.grad_norm_history)
            self.running_grad_mean = grad_norms.mean()
            self.running_grad_variance = grad_norms.var()
        else:
            self.running_grad_mean = grad_norm
            self.running_grad_variance = 0.0
        
        # Update adaptive weights
        self._update_lambdas()
        
        # Compute DFC loss
        self.current_dfc_loss = self.compute_loss(
            self.running_entropy, 
            self.running_grad_variance
        )
    
    def _update_lambdas(self):
        """
        Update adaptive weights based on current statistics.
        Uses rule: lambda_t = lambda_0 * (1 + beta * Var(∇L_total))
        Optionally modulated by entropy.
        """
        # Base update: lambda_t = lambda_0 * (1 + beta * Var(∇L_total))
        # We use gradient norm variance as proxy for gradient variance
        grad_var_term = 1.0 + self.beta * self.running_grad_variance
        
        # Entropy modulation (optional)
        if self.use_entropy_modulation and len(self.entropy_history) > 0:
            # Normalize entropy (assuming typical range 0-10 for log(num_classes))
            # Higher entropy -> increase weights
            entropy_normalized = self.running_entropy / 10.0  # Rough normalization
            entropy_modulation = 1.0 + self.entropy_weight * entropy_normalized
        else:
            entropy_modulation = 1.0
        
        # Compute adaptive weights
        self.lambda_idg_t = self.lambda_idg_0 * grad_var_term * entropy_modulation
        self.lambda_pmr_t = self.lambda_pmr_0 * grad_var_term * entropy_modulation
        self.lambda_dfc_t = self.lambda_dfc_0 * grad_var_term * entropy_modulation
        
        # Clamp to safe range [0, lambda_max]
        self.lambda_idg_t = max(0.0, min(self.lambda_idg_t, self.lambda_max))
        self.lambda_pmr_t = max(0.0, min(self.lambda_pmr_t, self.lambda_max))
        self.lambda_dfc_t = max(0.0, min(self.lambda_dfc_t, self.lambda_max))
    
    def compute_loss(self, entropy, grad_var):
        """
        Compute DFC loss term based on entropy and gradient variance.
        
        The DFC loss encourages entropy and gradient variance to stay close to
        target values, providing additional regularization.
        
        Args:
            entropy: Current prediction entropy (scalar)
            grad_var: Current gradient variance (scalar)
        
        Returns:
            L_dfc: Scalar DFC loss value
        """
        # Convert to scalar if tensor
        if torch.is_tensor(entropy):
            entropy = entropy.item()
        if torch.is_tensor(grad_var):
            grad_var = grad_var.item()
        
        # Compute squared differences from targets
        L_entropy = (entropy - self.entropy_target) ** 2
        L_grad = (grad_var - self.gradvar_target) ** 2
        
        # Weighted combination
        L_dfc = self.w_entropy * L_entropy + self.w_grad * L_grad
        
        return L_dfc
    
    def get_lambdas(self, return_alpha=False):
        """
        Get current adaptive weights.
        
        Args:
            return_alpha: If True, also return alpha mixing coefficient for MS-IDG
                         (currently returns None as placeholder)
        
        Returns:
            lambda_idg_t: Current adaptive weight for IDG loss
            lambda_pmr_t: Current adaptive weight for PMR loss
            lambda_dfc_t: Current adaptive weight for DFC loss
            alpha: Optional mixing coefficient (None if return_alpha=False)
        """
        if return_alpha:
            # Placeholder for MS-IDG alpha mixing coefficient
            # Could be computed based on current statistics if needed
            alpha = None
            return self.lambda_idg_t, self.lambda_pmr_t, self.lambda_dfc_t, alpha
        else:
            return self.lambda_idg_t, self.lambda_pmr_t, self.lambda_dfc_t
    
    def get_dfc_loss(self):
        """
        Get the current DFC loss value.
        
        This should be called after update() to get the most recent loss.
        
        Returns:
            L_dfc: Current DFC loss value (scalar)
        """
        return self.current_dfc_loss
    
    def get_alpha_scale(self, num_domains, use_entropy=True, use_loss=True, scale_factor=1.5):
        """
        Get per-domain scaling factors for MS-IDG mixing coefficients.
        
        DFC-controlled αᵢ for MS-IDG, matching the AIDA paper description.
        Adaptively biases mixing towards domains that need more alignment
        (e.g., higher entropy or higher loss).
        
        Args:
            num_domains: Number of domains (K)
            use_entropy: If True, use per-domain entropy to compute scaling
            use_loss: If True, use per-domain loss to compute scaling
            scale_factor: Maximum scaling factor (domains with high entropy/loss get higher scale)
        
        Returns:
            alpha_scale: [num_domains] tensor of scaling factors (defaults to 1.0 if no data)
        """
        # Initialize with ones (no scaling by default)
        alpha_scale = np.ones(num_domains, dtype=np.float32)
        
        # Collect statistics for all domains
        domain_scores = {}
        for domain_id in range(num_domains):
            score = 0.0
            count = 0
            
            # Use entropy if available
            if use_entropy and domain_id in self.running_domain_entropy:
                entropy_val = self.running_domain_entropy[domain_id]
                # Higher entropy -> higher score (needs more alignment)
                score += entropy_val / 10.0  # Normalize (assuming max entropy ~10)
                count += 1
            
            # Use loss if available
            if use_loss and domain_id in self.running_domain_loss:
                loss_val = self.running_domain_loss[domain_id]
                # Higher loss -> higher score (needs more alignment)
                # Normalize loss (assuming typical range 0-10)
                score += loss_val / 10.0
                count += 1
            
            if count > 0:
                domain_scores[domain_id] = score / count
            else:
                domain_scores[domain_id] = 0.0
        
        # Compute scaling factors
        if len(domain_scores) > 0:
            # Normalize scores to [0, 1] range
            max_score = max(domain_scores.values()) if domain_scores.values() else 1.0
            min_score = min(domain_scores.values()) if domain_scores.values() else 0.0
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            for domain_id in range(num_domains):
                if domain_id in domain_scores:
                    # Normalize score to [0, 1]
                    normalized_score = (domain_scores[domain_id] - min_score) / score_range if score_range > 0 else 0.5
                    # Scale: 1.0 (no change) to scale_factor (maximum boost)
                    # Higher score -> higher scale -> more mixing weight
                    alpha_scale[domain_id] = 1.0 + (scale_factor - 1.0) * normalized_score
                else:
                    alpha_scale[domain_id] = 1.0  # Default: no scaling
        else:
            # No domain statistics available, return all ones
            alpha_scale = np.ones(num_domains, dtype=np.float32)
        
        return torch.tensor(alpha_scale, dtype=torch.float32)
    
    def get_statistics(self):
        """
        Get current internal statistics for monitoring.
        
        Returns:
            dict with current statistics
        """
        return {
            'running_entropy': self.running_entropy,
            'running_grad_mean': self.running_grad_mean,
            'running_grad_variance': self.running_grad_variance,
            'lambda_idg_t': self.lambda_idg_t,
            'lambda_pmr_t': self.lambda_pmr_t,
            'lambda_dfc_t': self.lambda_dfc_t,
            'dfc_loss': self.current_dfc_loss,
            'entropy_target': self.entropy_target,
            'gradvar_target': self.gradvar_target,
            'update_count': self.update_count
        }
    
    def reset(self):
        """Reset internal state to initial values."""
        self.entropy_history.clear()
        self.grad_norm_history.clear()
        self.running_entropy = 0.0
        self.running_grad_variance = 0.0
        self.running_grad_mean = 0.0
        self.update_count = 0
        self.lambda_idg_t = self.lambda_idg_0
        self.lambda_pmr_t = self.lambda_pmr_0
        self.lambda_dfc_t = self.lambda_dfc_0
        self.current_dfc_loss = 0.0
        # Reset per-domain statistics
        self.domain_entropy_history.clear()
        self.domain_loss_history.clear()
        self.running_domain_entropy.clear()
        self.running_domain_loss.clear()
    
    def compute_entropy_from_logits(self, logits):
        """
        Utility method to compute prediction entropy from logits.
        
        Args:
            logits: [B, num_classes] logits tensor
        
        Returns:
            entropy: Scalar entropy value (average over batch)
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        return entropy
    
    def compute_grad_norm(self, parameters):
        """
        Utility method to compute gradient norm from model parameters.
        
        Args:
            parameters: Iterable of model parameters (e.g., model.parameters())
        
        Returns:
            grad_norm: Scalar gradient norm
        """
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

