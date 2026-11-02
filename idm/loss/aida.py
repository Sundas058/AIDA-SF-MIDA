"""Normalized Re-ID objectives: Eqs. (6)-(8) and (15)-(17)."""
import torch
from torch import nn
from torch.nn import functional as F


def batch_hard_triplet(embeddings, labels, margin=0.3):
    z = F.normalize(embeddings, dim=1)
    distances = torch.cdist(z, z, p=2)
    same = labels[:, None].eq(labels[None, :])
    positives = same & ~torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    negatives = ~same
    valid = positives.any(1) & negatives.any(1)
    if not valid.any():
        return embeddings.sum() * 0
    hardest_positive = distances.masked_fill(~positives, -torch.inf).max(1).values
    hardest_negative = distances.masked_fill(~negatives, torch.inf).min(1).values
    return F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()


def supervised_loss(logits, embeddings, labels, margin=0.3, lambda_tri=1.0):
    return F.cross_entropy(logits, labels) + lambda_tri * batch_hard_triplet(embeddings, labels, margin)


class PseudoMirrorLoss(nn.Module):
    def __init__(self, lambda_rel=1.0, identity=True, relational=True):
        super().__init__()
        self.lambda_rel = lambda_rel
        self.identity = identity
        self.relational = relational

    def forward(self, original, mirrored):
        z = F.normalize(original, dim=1)
        mirror = F.normalize(mirrored, dim=1)
        pointwise = (z - mirror).square().sum(1).mean()
        if len(z) > 1:
            pairs = torch.triu_indices(len(z), len(z), offset=1, device=z.device)
            d1 = torch.linalg.vector_norm(z[pairs[0]] - z[pairs[1]], dim=1)
            d2 = torch.linalg.vector_norm(mirror[pairs[0]] - mirror[pairs[1]], dim=1)
            relational = (d1 - d2).abs().mean()
        else:
            relational = z.sum() * 0
        return (pointwise if self.identity else pointwise * 0) + self.lambda_rel * (
            relational if self.relational else relational * 0
        )

