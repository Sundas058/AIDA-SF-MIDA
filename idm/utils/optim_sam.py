from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.optim import Optimizer


class SAM(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer wrapper.
    
    SAM optimizes for flat minima by performing two forward-backward passes per step:
    1. First step: Move in the direction of the gradient (ascend to find sharp region)
    2. Second step: Recompute loss and move in the direction of the new gradient (descend)
    
    This implementation wraps any base optimizer (e.g., AdamW) and adds SAM functionality.
    
    Reference: "Foret et al., Sharpness-Aware Minimization for Efficiently Improving 
                Generalization, ICLR 2021"
    
    Usage:
        # Create base optimizer
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        # Wrap with SAM
        optimizer = SAM(model.parameters(), base_optimizer, rho=0.05, adaptive=True)
        
        # Training loop:
        for batch in dataloader:
            # Forward pass
            loss = compute_loss(model, batch)
            
            # Backward pass
            loss.backward()
            
            # First step: ascend to sharp region
            optimizer.first_step(zero_grad=True)
            
            # Recompute loss at new point
            loss = compute_loss(model, batch)
            loss.backward()
            
            # Second step: descend from sharp region
            optimizer.second_step(zero_grad=True)
    """
    
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Initialize SAM optimizer.
        
        Args:
            params: Model parameters (iterable of parameters or dicts defining parameter groups)
            base_optimizer: Base optimizer to wrap (e.g., AdamW, Adam, SGD)
            rho: Neighborhood size (radius) for sharpness-aware minimization (default: 0.05)
            adaptive: If True, use adaptive rho per parameter (default: False)
            **kwargs: Additional arguments passed to base optimizer
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        # Store base optimizer
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        
        # Track if we're in first_step or second_step
        self.state_step = 0  # 0: before first_step, 1: after first_step, 2: after second_step
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First step: Move parameters in the direction of the gradient (ascend).
        
        This finds a point in the neighborhood that maximizes the loss (sharp region).
        The gradient is normalized and scaled by rho.
        
        Args:
            zero_grad: If True, zero gradients after first step (default: False)
        """
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Store original parameter values
                self.state[p]["old_p"] = p.data.clone()
                
                # Compute adaptive rho if enabled
                if self.adaptive:
                    # Adaptive rho: scale by parameter magnitude
                    param_norm = p.data.norm(2)
                    adaptive_scale = scale * (param_norm / (grad_norm + 1e-12))
                    e_w = adaptive_scale * p.grad
                else:
                    # Fixed rho: use same scale for all parameters
                    e_w = scale * p.grad
                
                # First step: ascend (move in direction of gradient)
                p.add_(e_w)
        
        self.state_step = 1
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Second step: Move parameters using the base optimizer from the sharp region.
        
        This step uses the base optimizer (e.g., AdamW) to update parameters
        based on the gradient computed at the sharp region found in first_step.
        
        Args:
            zero_grad: If True, zero gradients after second step (default: False)
        """
        # Restore original parameters before applying base optimizer step
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Restore original parameter values
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]
        
        # Apply base optimizer step (e.g., AdamW step)
        self.base_optimizer.step()
        
        self.state_step = 2
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Standard step method (not used in SAM workflow).
        
        SAM requires first_step() and second_step() to be called separately.
        This method raises an error to prevent accidental use.
        
        Args:
            closure: Optional closure for recomputing loss (not used)
        
        Raises:
            RuntimeError: Always raises, as SAM requires first_step() and second_step()
        """
        raise RuntimeError(
            "SAM optimizer requires first_step() and second_step() to be called separately. "
            "Do not call step() directly. See docstring for usage example."
        )
    
    def _grad_norm(self):
        """
        Compute the L2 norm of gradients across all parameters.
        
        Returns:
            grad_norm: Scalar L2 norm of all gradients
        """
        # Handle case where no gradients are available
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grads.append(p.grad.detach().norm(p=2))
        
        if len(grads) == 0:
            # Return zero norm if no gradients
            if len(self.param_groups) > 0 and len(self.param_groups[0]["params"]) > 0:
                device = next(iter(self.param_groups[0]["params"])).device
                return torch.tensor(0.0, device=device)
            else:
                return torch.tensor(0.0)
        
        # Concatenate all gradients and compute norm
        all_grads = torch.stack(grads)
        norm = all_grads.norm(p=2)
        return norm
    
    def zero_grad(self, set_to_none=False):
        """
        Zero gradients for both SAM and base optimizer.
        
        Args:
            set_to_none: If True, set gradients to None instead of zero (default: False)
        """
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
        super().zero_grad(set_to_none=set_to_none)
    
    def state_dict(self):
        """
        Get state dict including base optimizer state.
        
        Returns:
            state_dict: Dictionary containing SAM and base optimizer state
        """
        state_dict = super().state_dict()
        state_dict['base_optimizer'] = self.base_optimizer.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict):
        """
        Load state dict including base optimizer state.
        
        Args:
            state_dict: Dictionary containing SAM and base optimizer state
        """
        base_optimizer_state = state_dict.pop('base_optimizer', None)
        super().load_state_dict(state_dict)
        if base_optimizer_state is not None:
            self.base_optimizer.load_state_dict(base_optimizer_state)


def create_sam_optimizer(model, base_optimizer_class, lr, weight_decay=0, 
                         rho=0.05, adaptive=False, **base_optimizer_kwargs):
    """
    Convenience function to create a SAM-wrapped optimizer.
    
    Args:
        model: PyTorch model
        base_optimizer_class: Base optimizer class (e.g., torch.optim.AdamW)
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        rho: SAM neighborhood size (default: 0.05)
        adaptive: Use adaptive rho (default: False)
        **base_optimizer_kwargs: Additional arguments for base optimizer
    
    Returns:
        SAM optimizer instance
    
    Example:
        optimizer = create_sam_optimizer(
            model, torch.optim.AdamW, lr=0.001, weight_decay=5e-4,
            rho=0.05, adaptive=True
        )
    """
    # Get model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Create base optimizer
    base_optimizer = base_optimizer_class(
        params, lr=lr, weight_decay=weight_decay, **base_optimizer_kwargs
    )
    
    # Wrap with SAM
    sam_optimizer = SAM(params, base_optimizer, rho=rho, adaptive=adaptive)
    
    return sam_optimizer


