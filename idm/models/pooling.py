"""Generalized mean pooling for DG and source-free embeddings."""
import torch
from torch import nn


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor([p]))
        self.eps = eps

    def forward(self, x):
        exponent = self.p.clamp(min=self.eps)
        return x.clamp(min=self.eps).pow(exponent).mean((-1, -2)).pow(1 / exponent)
