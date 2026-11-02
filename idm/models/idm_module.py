"""Intermediate feature generation, AdaIN, and normalized pseudo-mirror losses."""
import torch
from torch import nn
from .pooling import GeM
from .aida import MSIDG, adain, target_mirror
from ..loss.aida import PseudoMirrorLoss

def adaptive_instance_normalization(content_feat, style_mean, style_std):
    if style_mean.ndim == 1:
        style_mean = style_mean[None, :, None, None]
    if style_std.ndim == 1:
        style_std = style_std[None, :, None, None]
    return adain(content_feat, style_mean, style_std)

class IDM(nn.Module):
    def __init__(self, channel=64):
        super(IDM, self).__init__()
        self.channel = channel
        self.adaptiveFC1 = nn.Linear(2 * channel, channel)
        self.adaptiveFC2 = nn.Linear(channel, int(channel / 2))
        self.adaptiveFC3 = nn.Linear(int(channel / 2), 2)
        self.softmax = nn.Softmax(dim=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        if not self.training:
            return x

        bs = x.size(0)
        assert (bs % 2 == 0)
        split = torch.split(x, int(bs / 2), 0)
        x_s = split[0].contiguous()
        x_t = split[1].contiguous()

        x_embd_s = torch.cat(
            (self.avg_pool(x_s.detach()).flatten(1), self.max_pool(x_s.detach()).flatten(1)), 1
        )
        x_embd_t = torch.cat(
            (self.avg_pool(x_t.detach()).flatten(1), self.max_pool(x_t.detach()).flatten(1)), 1
        )

        x_embd_s, x_embd_t = self.adaptiveFC1(x_embd_s), self.adaptiveFC1(x_embd_t)
        x_embd = x_embd_s + x_embd_t
        x_embd = self.adaptiveFC2(x_embd)
        lam = self.adaptiveFC3(x_embd)
        lam = self.softmax(lam)
        x_inter = lam[:, 0].reshape(-1, 1, 1, 1) * x_s + lam[:, 1].reshape(-1, 1, 1, 1) * x_t
        out = torch.cat((x_s, x_t, x_inter), 0)
        return out, lam


class MultiSourceIntermediateDomainGenerator(MSIDG):
    """Eq. (13). Identity supervision is computed by the trainer after embedding."""
    def __init__(self, channel=2048, num_sources=3, learnable_alpha=False):
        super().__init__()
        self.num_sources = num_sources

    def forward(self, features, domain_ids, alpha=None):
        if alpha is None:
            alpha = features.new_full((self.num_sources,), 1 / self.num_sources)
        return super().forward(features, domain_ids, alpha), alpha


class MirrorGenerationModule(nn.Module):
    def __init__(self, channel=2048):
        super().__init__()

    def forward(self, features, style_mean=None, style_std=None):
        if style_mean is None:
            return target_mirror(features)
        return adaptive_instance_normalization(features, style_mean, style_std)


class PseudoMirrorRegularization(PseudoMirrorLoss):
    """Eqs. (15)-(17), applied to original and mirrored identity embeddings."""

