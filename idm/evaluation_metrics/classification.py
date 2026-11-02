from __future__ import absolute_import

import torch
from ..utils import to_torch


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k.
    Makes it robust if output and target batch sizes mismatch."""
    with torch.no_grad():
        output, target = to_torch(output), to_torch(target)
        maxk = max(topk)

        # output: [N, num_classes]
        # target: [N]
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # [maxk, N_pred]

        # --- NEW: align batch sizes if they mismatch ---
        batch_pred = pred.size(1)
        batch_tgt = target.size(0)
        if batch_pred != batch_tgt:
            min_bs = min(batch_pred, batch_tgt)
            pred = pred[:, :min_bs]
            target = target[:min_bs]
        else:
            min_bs = batch_tgt
        batch_size = min_bs
        # ------------------------------------------------

        target_expanded = target.view(1, -1).expand_as(pred)
        correct = pred.eq(target_expanded)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

