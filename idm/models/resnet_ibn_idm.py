from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from collections import OrderedDict

from .idm_module import IDM, GeM
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from .vit_blocks import ViTStack

__all__ = ['ResNetIBN', 'resnet_ibn50a_idm', 'resnet_ibn101a_idm']


class ResNetIBN(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, use_gem=False,
                 use_vit_blocks=False, vit_num_blocks=2, vit_num_heads=4, vit_mlp_ratio=4.0):
        super(ResNetIBN, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.use_gem = use_gem
        self.use_vit_blocks = use_vit_blocks

        resnet = ResNetIBN.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.conv = nn.Sequential(OrderedDict([
            ('conv1', resnet.conv1),
            ('bn1', resnet.bn1),
            ('relu', resnet.relu),
            ('maxpool', resnet.maxpool)]))

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Get output channel dimension from ResNet
        out_planes = resnet.fc.in_features  # Typically 2048 for ResNet-50/101

        # ViT blocks on top of CNN features (if enabled)
        if self.use_vit_blocks:
            self.vit_stack = ViTStack(
                embed_dim=out_planes,
                num_blocks=vit_num_blocks,
                num_heads=vit_num_heads,
                mlp_ratio=vit_mlp_ratio,
                dropout=dropout
            )
        else:
            self.vit_stack = None

        # Use GeM pooling if enabled, otherwise use GAP (backward compatible)
        if self.use_gem:
            self.pool = GeM(p=3.0, eps=1e-6)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

        self.idm1 = IDM(channel=64)
        self.idm2 = IDM(channel=256)
        self.idm3 = IDM(channel=512)
        self.idm4 = IDM(channel=1024)
        self.idm5 = IDM(channel=2048)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x, stage=0):
        x = self.conv(x)
        if stage == 0 and self.training:
            x, attention_lam = self.idm1(x)
        x = self.layer1(x)
        if stage == 1 and self.training:
            x, attention_lam = self.idm2(x)
        x = self.layer2(x)
        if stage == 2 and self.training:
            x, attention_lam = self.idm3(x)
        x = self.layer3(x)
        if stage == 3 and self.training:
            x, attention_lam = self.idm4(x)
        x = self.layer4(x)
        if stage == 4 and self.training:
            x, attention_lam = self.idm5(x)

        # Apply ViT blocks on top of CNN features (if enabled)
        # This transforms the feature maps before pooling
        if self.use_vit_blocks and self.vit_stack is not None:
            x = self.vit_stack(x)  # [B, C, H, W] -> [B, C, H, W] (transformed)

        # Apply pooling (GeM or GAP)
        if self.use_gem:
            x = self.pool(x)  # GeM returns [B, C] directly
        else:
            x = self.pool(x)  # GAP returns [B, C, 1, 1]
            x = x.view(x.size(0), -1)  # Flatten to [B, C]

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if self.training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x

        return prob, x, attention_lam

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def resnet_ibn50a_idm(**kwargs):
    return ResNetIBN('50a', **kwargs)


def resnet_ibn101a_idm(**kwargs):
    return ResNetIBN('101a', **kwargs)


