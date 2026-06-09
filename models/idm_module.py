from __future__ import absolute_import

from torch import nn
import torch
import torch.nn.functional as F
from collections import defaultdict


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    """
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps)
        x = x.pow(self.p).mean(dim=(-1, -2))  # [B, C]
        return x.pow(1.0 / self.p)


def adaptive_instance_normalization(content_feat, style_mean, style_std):
    """
    Apply AdaIN to content features using style statistics.

    FIX: use unbiased=False in std() to avoid "degrees of freedom <= 0"
    when H*W == 1 (e.g., [B,C,1,1]).
    """
    size = content_feat.size()

    content_mean = content_feat.view(size[0], size[1], -1).mean(dim=2).view(size[0], size[1], 1, 1)

    # IMPORTANT FIX HERE
    content_std = (
        content_feat.view(size[0], size[1], -1)
        .std(dim=2, unbiased=False)   # <-- FIX
        .view(size[0], size[1], 1, 1)
        + 1e-8
    )

    normalized = (content_feat - content_mean) / content_std

    if style_mean.dim() == 1:
        style_mean = style_mean.view(1, -1, 1, 1)
    if style_std.dim() == 1:
        style_std = style_std.view(1, -1, 1, 1)

    style_std = style_std.clamp(min=1e-8)
    return normalized * style_std + style_mean


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
            (self.avg_pool(x_s.detach()).squeeze(), self.max_pool(x_s.detach()).squeeze()), 1
        )
        x_embd_t = torch.cat(
            (self.avg_pool(x_t.detach()).squeeze(), self.max_pool(x_t.detach()).squeeze()), 1
        )

        x_embd_s, x_embd_t = self.adaptiveFC1(x_embd_s), self.adaptiveFC1(x_embd_t)
        x_embd = x_embd_s + x_embd_t
        x_embd = self.adaptiveFC2(x_embd)
        lam = self.adaptiveFC3(x_embd)
        lam = self.softmax(lam)
        x_inter = lam[:, 0].reshape(-1, 1, 1, 1) * x_s + lam[:, 1].reshape(-1, 1, 1, 1) * x_t
        out = torch.cat((x_s, x_t, x_inter), 0)
        return out, lam


class MultiSourceIntermediateDomainGenerator(nn.Module):
    """
    Multi-Source Intermediate Domain Generator (MS-IDG).
    """
    def __init__(self, channel=64, num_sources=3, learnable_alpha=True):
        super(MultiSourceIntermediateDomainGenerator, self).__init__()
        self.channel = channel
        self.num_sources = num_sources
        self.learnable_alpha = learnable_alpha

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        if learnable_alpha:
            self.alpha_fc1 = nn.Linear(2 * channel, channel)
            self.alpha_fc2 = nn.Linear(channel, int(channel / 2))
            self.alpha_fc3 = nn.Linear(int(channel / 2), num_sources)
            self.softmax = nn.Softmax(dim=1)
        else:
            self.register_buffer("alpha_placeholder", torch.zeros(1))

    def _compute_domain_statistics(self, features, domain_ids):
        domain_features = defaultdict(list)
        domain_means = {}
        domain_stds = {}

        for i, domain_id in enumerate(domain_ids):
            domain_id = domain_id.item()
            domain_features[domain_id].append(features[i : i + 1])

        for domain_id, feat_list in domain_features.items():
            if len(feat_list) == 0:
                continue

            domain_feat = torch.cat(feat_list, dim=0)
            N_d, C, H, W = domain_feat.size()
            feat_flat = domain_feat.view(N_d, C, -1)

            domain_mean = feat_flat.mean(dim=(0, 2))
            # FIX: unbiased=False (safer if spatial size is tiny)
            domain_std = feat_flat.std(dim=(0, 2), unbiased=False).clamp(min=1e-8)

            domain_means[domain_id] = domain_mean
            domain_stds[domain_id] = domain_std

        return domain_means, domain_stds, domain_features

    def _generate_mixing_coefficients(self, features, domain_ids):
        B = features.size(0)

        if self.learnable_alpha:
            feat_embd = torch.cat(
                (
                    self.avg_pool(features.detach()).squeeze(-1).squeeze(-1),
                    self.max_pool(features.detach()).squeeze(-1).squeeze(-1),
                ),
                dim=1,
            )

            alpha = self.alpha_fc1(feat_embd)
            alpha = self.alpha_fc2(alpha)
            alpha = self.alpha_fc3(alpha)
            alpha = self.softmax(alpha)
        else:
            alpha = torch.zeros(B, self.num_sources, device=features.device)
            unique_domains = torch.unique(domain_ids)
            num_domains = len(unique_domains)

            if num_domains > 0:
                uniform_weight = 1.0 / num_domains
                for domain_id in unique_domains:
                    mask = (domain_ids == domain_id)
                    for d in unique_domains:
                        alpha[mask, d.item()] = uniform_weight
            else:
                alpha.fill_(1.0 / self.num_sources)

        return alpha

    def forward(self, features, domain_ids, alpha=None, alpha_scale=None):
        if not self.training:
            if alpha is None:
                B = features.size(0)
                alpha = torch.zeros(B, self.num_sources, device=features.device)
                alpha.fill_(1.0 / self.num_sources)
            return features, alpha, torch.tensor(0.0, device=features.device)

        B, C, H, W = features.size()

        domain_means, domain_stds, _ = self._compute_domain_statistics(features, domain_ids)
        unique_domains = sorted(domain_means.keys())
        K_batch = len(unique_domains)

        if K_batch == 0:
            if alpha is None:
                alpha = torch.zeros(B, self.num_sources, device=features.device)
                alpha.fill_(1.0 / self.num_sources)
            return features, alpha, torch.tensor(0.0, device=features.device)

        if alpha is None:
            alpha = self._generate_mixing_coefficients(features, domain_ids)

        if alpha_scale is not None:
            if alpha_scale.device != alpha.device:
                alpha_scale = alpha_scale.to(alpha.device)
            if alpha_scale.dim() == 1:
                alpha_scale = alpha_scale.unsqueeze(0)
            alpha_scale = alpha_scale.expand_as(alpha)

            alpha = alpha * alpha_scale
            alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)

        domain_to_alpha_idx = {domain_id: idx for idx, domain_id in enumerate(unique_domains)}

        alpha_cols = min(alpha.size(1), K_batch)
        if alpha.size(1) > K_batch:
            alpha = alpha[:, :K_batch]
            alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)

        intermediate_features_list = []
        for b in range(B):
            mixed_mean_b = torch.zeros(C, device=features.device)
            mixed_std_b = torch.zeros(C, device=features.device)

            for domain_id in unique_domains:
                alpha_idx = domain_to_alpha_idx[domain_id]
                if alpha_idx < alpha_cols:
                    weight = alpha[b, alpha_idx]
                    mixed_mean_b = mixed_mean_b + weight * domain_means[domain_id]
                    mixed_std_b = mixed_std_b + weight * domain_stds[domain_id]

            mixed_std_b = mixed_std_b.clamp(min=1e-8)

            sample_feat = features[b : b + 1]
            sample_inter = adaptive_instance_normalization(sample_feat, mixed_mean_b, mixed_std_b)
            intermediate_features_list.append(sample_inter)

        intermediate_features = torch.cat(intermediate_features_list, dim=0)
        loss_idg = F.mse_loss(intermediate_features, features.detach())

        return intermediate_features, alpha, loss_idg


class MirrorGenerationModule(nn.Module):
    def __init__(self, channel=64):
        super(MirrorGenerationModule, self).__init__()
        self.channel = channel

    def forward(self, features, style_mean, style_std):
        mirror_features = adaptive_instance_normalization(features, style_mean, style_std)
        return mirror_features


class PseudoMirrorRegularization(nn.Module):
    def __init__(self, channel=64, lambda_sd=1.0):
        super(PseudoMirrorRegularization, self).__init__()
        self.channel = channel
        self.lambda_sd = lambda_sd
        self.mgm = MirrorGenerationModule(channel=channel)

    def generate_mirror(self, features, style_stats_or_codes):
        if isinstance(style_stats_or_codes, dict):
            style_mean = style_stats_or_codes["mean"]
            style_std = style_stats_or_codes["std"]
        elif isinstance(style_stats_or_codes, tuple):
            style_mean, style_std = style_stats_or_codes
        elif isinstance(style_stats_or_codes, torch.Tensor):
            style_mean = style_stats_or_codes
            B, C, H, W = features.size()
            feat_flat = features.view(B, C, -1)
            # FIX: unbiased=False
            style_std = feat_flat.std(dim=2, unbiased=False).mean(dim=0).clamp(min=1e-8)
        else:
            raise ValueError("style_stats_or_codes must be dict, tuple, or tensor")

        mirror_features = self.mgm(features, style_mean, style_std)
        return mirror_features

    def _compute_feature_loss(self, original_features, mirror_features):
        return F.mse_loss(original_features, mirror_features)

    def _compute_kl_loss(self, original_logits, mirror_logits):
        p_s = F.softmax(original_logits, dim=1)
        p_t = F.softmax(mirror_logits, dim=1)

        p_s = (p_s + 1e-8)
        p_t = (p_t + 1e-8)
        p_s = p_s / p_s.sum(dim=1, keepdim=True)
        p_t = p_t / p_t.sum(dim=1, keepdim=True)

        kl_div = p_t * torch.log(p_t / p_s)
        return kl_div.sum(dim=1).mean()

    def forward(self, features, logits, style_stats, classifier_fn=None):
        mirror_features = self.generate_mirror(features, style_stats)

        if classifier_fn is not None:
            mirror_logits = classifier_fn(mirror_features)
        else:
            mirror_logits = None

        loss_feat = self._compute_feature_loss(features, mirror_features)

        if mirror_logits is not None and logits is not None:
            loss_kl = self._compute_kl_loss(logits, mirror_logits)
        else:
            loss_kl = torch.tensor(0.0, device=features.device)

        loss_pmr = loss_feat + self.lambda_sd * loss_kl
        return mirror_features, mirror_logits, loss_pmr

    def forward_with_mirror_logits(self, features, original_logits, mirror_logits, style_stats):
        mirror_features = self.generate_mirror(features, style_stats)
        loss_feat = self._compute_feature_loss(features, mirror_features)
        loss_kl = self._compute_kl_loss(original_logits, mirror_logits)
        loss_pmr = loss_feat + self.lambda_sd * loss_kl
        return mirror_features, mirror_logits, loss_pmr
