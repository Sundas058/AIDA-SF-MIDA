"""Detached feedback controller implementing paper Eqs. (18)-(24)."""
from collections import deque
import math
import numpy as np
import torch


class DynamicFeedbackController:
    def __init__(self, num_sources=1, lambda_idg_0=1.0, lambda_pmr_0=1.0,
                 beta=0.1, gamma=0.5, lambda_max=10.0, entropy_max=10.0,
                 window_size=100, scale_max=1.5):
        if num_sources < 1 or window_size < 2 or entropy_max <= 0:
            raise ValueError("Invalid controller dimensions or normalization.")
        self.num_sources = num_sources
        self.lambda_idg_0, self.lambda_pmr_0 = lambda_idg_0, lambda_pmr_0
        self.beta, self.gamma = beta, gamma
        self.lambda_max, self.entropy_max, self.scale_max = lambda_max, entropy_max, scale_max
        self.grad_norms = deque(maxlen=window_size)
        self.variances = deque(maxlen=window_size)
        self.lambda_idg, self.lambda_pmr = lambda_idg_0, lambda_pmr_0
        self.alpha = torch.full((num_sources,), 1.0 / num_sources)
        self.entropy = self.variance = 0.0
        self.scale = 1.0
        self.updates = 0

    @staticmethod
    def prediction_entropy(logits):
        log_p = logits.detach().log_softmax(1)
        return -(log_p.exp() * log_p).sum(1)

    def update(self, entropy_value, grad_norm, domain_ids=None, domain_entropies=None):
        entropy_value, grad_norm = float(entropy_value), float(grad_norm)
        if not math.isfinite(entropy_value) or not math.isfinite(grad_norm):
            raise ValueError("DFC feedback must be finite.")
        self.entropy = entropy_value
        self.grad_norms.append(grad_norm)
        self.variance = float(np.var(self.grad_norms))
        self.variances.append(self.variance)
        vmax = max(self.variances)
        vbar = self.variance / vmax if vmax > 0 else 0.0
        ebar = self.entropy / self.entropy_max
        modulation = (1 + self.beta * vbar) * (1 + self.gamma * ebar)
        self.lambda_idg = float(np.clip(self.lambda_idg_0 * modulation, 0, self.lambda_max))
        self.lambda_pmr = float(np.clip(self.lambda_pmr_0 * modulation, 0, self.lambda_max))
        self.scale = float(np.clip(modulation, 1, self.scale_max))
        if domain_ids is not None and domain_entropies is not None:
            ids = domain_ids.detach().cpu()
            entropies = domain_entropies.detach().cpu()
            if any(not (ids == k).any() for k in range(self.num_sources)):
                raise ValueError("DFC mixing requires feedback for all source domains.")
            scores = torch.stack([entropies[ids == k].mean() for k in range(self.num_sources)])
            self.alpha = (self.scale * scores / self.entropy_max).softmax(0)
        self.updates += 1
        return self.lambda_idg, self.lambda_pmr

    def state_dict(self):
        return {
            "grad_norms": list(self.grad_norms), "variances": list(self.variances),
            "lambda_idg": self.lambda_idg, "lambda_pmr": self.lambda_pmr,
            "alpha": self.alpha.tolist(), "entropy": self.entropy,
            "variance": self.variance, "scale": self.scale, "updates": self.updates,
        }

    def load_state_dict(self, state):
        self.grad_norms.extend(state["grad_norms"])
        self.variances.extend(state["variances"])
        for key in ("lambda_idg", "lambda_pmr", "entropy", "variance", "scale", "updates"):
            setattr(self, key, state[key])
        self.alpha = torch.tensor(state["alpha"], dtype=torch.float32)

