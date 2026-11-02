from __future__ import absolute_import

import torch
from torch import nn


class DivLoss(nn.Module):
    def __init__(self):
        super(DivLoss, self).__init__()

    def forward(self, scores):
        mu = scores.mean(0)
        std = ((scores - mu) ** 2).mean(0, keepdim=True).clamp(min=1e-12).sqrt()
        loss_std = -std.sum()
        return loss_std


class BridgeFeatLoss(nn.Module):
    def __init__(self):
        super(BridgeFeatLoss, self).__init__()

    def forward(self, feats_s, feats_t, feats_mixed, lam):
        dist_mixed2s = ((feats_mixed - feats_s) ** 2).sum(1, keepdim=True)
        dist_mixed2t = ((feats_mixed - feats_t) ** 2).sum(1, keepdim=True)

        dist_mixed2s = dist_mixed2s.clamp(min=1e-12).sqrt()
        dist_mixed2t = dist_mixed2t.clamp(min=1e-12).sqrt()

        dist_mixed = torch.cat((dist_mixed2s, dist_mixed2t), 1)
        lam_dist_mixed = (lam * dist_mixed).sum(1, keepdim=True)
        loss = lam_dist_mixed.mean()
        return loss


class BridgeProbLoss(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super(BridgeProbLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        # IMPORTANT: do NOT force cuda here
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, lam):
        # Use the device of inputs (works for CPU/GPU)
        device = inputs.device
        logsoftmax = self.logsoftmax.to(device)

        # device_num must never be 0 on CPU
        device_num = max(1, torch.cuda.device_count())

        # ---- reshape inputs into (device_num, batch_per_device, dim) ----
        inputs = inputs.view(device_num, -1, inputs.size(-1))

        # Expect 3 chunks: source / target / mixed
        inputs_s, inputs_t, inputs_mixed = inputs.split(inputs.size(1) // 3, dim=1)

        inputs_ori = torch.cat((inputs_s, inputs_t), 1).contiguous().view(-1, inputs.size(-1))
        inputs_mixed = inputs_mixed.contiguous().view(-1, inputs.size(-1))

        log_probs_ori = logsoftmax(inputs_ori)
        log_probs_mixed = logsoftmax(inputs_mixed)

        # ---- build one-hot targets for "ori" (source+target) ----
        # targets here must correspond to inputs_ori length
        onehot = torch.zeros_like(log_probs_ori).scatter_(1, targets.unsqueeze(1), 1)

        # reshape using SAFE device_num (NOT self.device_num)
        onehot = onehot.view(device_num, -1, onehot.size(-1))

        targets_s, targets_t = onehot.split(onehot.size(1) // 2, dim=1)
        targets_s = targets_s.contiguous().view(-1, onehot.size(-1))
        targets_t = targets_t.contiguous().view(-1, onehot.size(-1))

        onehot_flat = onehot.contiguous().view(-1, onehot.size(-1))

        # label smoothing
        soft_targets = (1 - self.epsilon) * onehot_flat + self.epsilon / self.num_classes

        lam = lam.view(-1, 1).to(device)

        # Mixed soft labels
        soft_targets_mixed = lam * targets_s + (1.0 - lam) * targets_t
        soft_targets_mixed = (1 - self.epsilon) * soft_targets_mixed + self.epsilon / self.num_classes

        loss_ori = (-soft_targets * log_probs_ori).mean(0).sum()
        loss_bridge_prob = (-soft_targets_mixed * log_probs_mixed).mean(0).sum()

        return loss_ori, loss_bridge_prob

