"""ResNet-50 embeddings and channel-statistics domain transitions."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights
from .pooling import GeM


def channel_statistics(features):
    return features.mean((2, 3), keepdim=True), features.std((2, 3), keepdim=True, unbiased=False)


def adain(features, mean, std, eps=1e-6):
    own_mean, own_std = channel_statistics(features)
    return (features - own_mean) / (own_std + eps) * std + mean


class MSIDG(nn.Module):
    """Eq. (13); statistics are averaged within each source mini-batch."""
    def forward(self, features, domain_ids, alpha):
        domains = torch.unique(domain_ids, sorted=True)
        if domains.tolist() != list(range(len(alpha))):
            raise ValueError("Each MS-IDG batch must include every source domain, numbered 0..K-1.")
        if alpha.ndim != 1 or not torch.isfinite(alpha).all() or (alpha < 0).any():
            raise ValueError("Mixing coefficients must be a finite nonnegative vector.")
        if not torch.allclose(alpha.sum(), alpha.new_tensor(1.0), atol=1e-5):
            raise ValueError("Mixing coefficients must sum to one.")
        means, stds = channel_statistics(features)
        domain_means = torch.stack([means[domain_ids == d].mean(0) for d in domains])
        domain_stds = torch.stack([stds[domain_ids == d].mean(0) for d in domains])
        weights = alpha[:, None, None, None]
        return adain(features, (weights * domain_means).sum(0), (weights * domain_stds).sum(0))


def target_mirror(features):
    """Target-only statistics permutation; source statistics are never loaded."""
    mean, std = channel_statistics(features)
    # A nonzero cyclic shift prevents self-pairing for batches larger than one.
    shift = int(torch.randint(1, len(features), ()).item()) if len(features) > 1 else 0
    return adain(features, mean.roll(shift, 0), std.roll(shift, 0))


class InteractionMemory(nn.Module):
    """Trainable experience slots and value reliability for global/local tokens."""
    def __init__(self, dim, slots=32):
        super().__init__()
        self.experience = nn.Parameter(torch.randn(slots, dim) * 0.01)
        self.values = nn.Parameter(torch.zeros(slots))
        self.gate = nn.Parameter(torch.tensor(-4.0))

    def forward(self, tokens):
        scores = F.normalize(tokens, dim=-1) @ F.normalize(self.experience, dim=-1).T
        attention = (scores + F.logsigmoid(self.values)).softmax(-1)
        return tokens + self.gate.sigmoid() * (attention @ self.experience)


class AIDAEmbedding(nn.Module):
    def __init__(self, num_classes, pretrained=True, use_gem=True, num_parts=4, memory_slots=32, pairwise_idm=False):
        super().__init__()
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        net.layer4[0].conv2.stride = (1, 1)
        net.layer4[0].downsample[0].stride = (1, 1)
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.pool = GeM() if use_gem else nn.AdaptiveAvgPool2d(1)
        self.num_parts = num_parts
        self.global_memory = InteractionMemory(2048, memory_slots) if memory_slots else nn.Identity()
        self.local_memory = InteractionMemory(2048, memory_slots) if memory_slots else nn.Identity()
        self.part_gate = nn.Parameter(torch.tensor(0.0))
        self.feat_bn = nn.BatchNorm1d(2048)
        self.feat_bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        if pairwise_idm:
            from .idm_module import IDM
            self.idm = IDM(channel=2048)

    def feature_maps(self, images):
        return self.backbone(images)

    def embed(self, maps):
        global_token = self.global_memory(self.pool(maps).flatten(1))
        if self.num_parts:
            parts = F.adaptive_avg_pool2d(maps, (self.num_parts, 1)).squeeze(-1).transpose(1, 2)
            local = self.local_memory(parts).mean(1)
            global_token = global_token + self.part_gate.sigmoid() * local
        return F.normalize(self.feat_bn(global_token), dim=1)

    def forward(self, images):
        z = self.embed(self.feature_maps(images))
        return (self.classifier(z), z) if self.training else z

    def reset_classifier(self, centers):
        self.classifier = nn.Linear(2048, len(centers), bias=False).to(centers.device)
        with torch.no_grad():
            self.classifier.weight.copy_(F.normalize(centers, dim=1))

