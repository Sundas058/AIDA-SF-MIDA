"""Explicit source/target batch-normalization branches for UDA."""
from copy import deepcopy
from torch import nn


class DomainBatchNorm(nn.Module):
    def __init__(self, batch_norm):
        super().__init__()
        self.source = deepcopy(batch_norm)
        self.target = deepcopy(batch_norm)
        self.domain = "source"

    def forward(self, x):
        return getattr(self, self.domain)(x)


def convert_domain_bn(module):
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            setattr(module, name, DomainBatchNorm(child))
        else:
            convert_domain_bn(child)


def set_domain(module, domain):
    for child in module.modules():
        if isinstance(child, DomainBatchNorm):
            child.domain = domain
