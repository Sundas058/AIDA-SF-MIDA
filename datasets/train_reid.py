#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import glob
from collections import OrderedDict
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

# =========================
# Dataset paths (your setup)
# =========================
DUKE_ROOT   = "/home/tq_sundas/dataset/dukemtmcreid/DukeMTMC-reID"
MARKET_ROOT = "/home/tq_sundas/dataset/market1501/Market-1501-v15.09.15"

DATASETS = {
    "duke": {
        "train":   os.path.join(DUKE_ROOT,   "bounding_box_train"),
        "query":   os.path.join(DUKE_ROOT,   "query"),
        "gallery": os.path.join(DUKE_ROOT,   "bounding_box_test"),
    },
    "market": {
        "train":   os.path.join(MARKET_ROOT, "bounding_box_train"),
        "query":   os.path.join(MARKET_ROOT, "query"),
        "gallery": os.path.join(MARKET_ROOT, "bounding_box_test"),
    },
}

# =========================
# Utilities
# =========================
def list_images(root_dir):
    exts = ("*.jpg", "*.JPG", "*.jpeg", "*.png")
    files = []
    for pat in exts:
        files.extend(glob.glob(os.path.join(root_dir, pat)))
    return sorted(files)

def parse_pid_cam(filename):
    base = os.path.basename(filename)
    pid = int(base.split("_")[0])
    camid = int(base.split("_")[1][1])  # c1 -> 1
    return pid, camid

# =========================
# Dataset (with PID reindexing)
# =========================
class ReIDDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

        self.images = list_images(root_dir)
        if len(self.images) == 0:
            raise FileNotFoundError(f"No images found in {root_dir}. Expected .jpg/.jpeg/.png files.")

        raw_pids, camids = [], []
        for p in self.images:
            pid, cam = parse_pid_cam(p)
            raw_pids.append(pid)
            camids.append(cam)

        unique = sorted(set(raw_pids))
        self.pid2label = {pid: idx for idx, pid in enumerate(unique)}
        labels = [self.pid2label[pid] for pid in raw_pids]

        self.pids = np.asarray(labels, dtype=np.int64)
        self.camids = np.asarray(camids, dtype=np.int64)
        self.num_classes = len(self.pid2label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = Image.open(path).convert("RGB")
        pid = int(self.pids[idx])
        cam = int(self.camids[idx])
        if self.transform:
            img = self.transform(img)
        return img, pid, cam

# =========================
# Models
# =========================
class ReIDHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.fc = nn.Linear(in_dim, num_classes)

        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out")
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.bn(x)
        logits = self.fc(x)
        return x, logits

class ReIDResNet(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()
        if backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            feat_dim = 2048
        elif backbone_name == "resnet101":
            backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
            feat_dim = 2048
        else:
            raise ValueError("backbone must be 'resnet50' or 'resnet101'")

        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.head = ReIDHead(feat_dim, num_classes)

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.head(x)

# =========================
# Loss (CE + Triplet-hardest)
# =========================
class MultiLoss(nn.Module):
    def __init__(self, alpha=0.7, margin=0.3):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()

    def pairwise(self, x):
        n = (x ** 2).sum(1, keepdim=True)
        d = n + n.t() - 2 * (x @ x.t())
        return torch.clamp(d, min=1e-6)

    def triplet(self, feats, pids):
        d = self.pairwise(feats)
        loss = feats.new_tensor(0.)
        count = 0
        for i in range(feats.size(0)):
            pos = d[i][pids == pids[i]]
            neg = d[i][pids != pids[i]]
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            loss = loss + F.relu(pos.max() - neg.min() + self.margin)
            count += 1
        return loss / max(count, 1)

    def forward(self, feats, logits, pids):
        ce = self.ce(logits, pids)
        tri = self.triplet(feats, pids)
        return self.alpha * ce + (1 - self.alpha) * tri

# =========================
# Pretrained loader (logs/pretrained)
# =========================
def load_pretrained_if_available(model, device, prefer_name="resnet101_ibn_a.pth"):
    base = os.path.join("logs", "pretrained")
    pref = os.path.join(base, prefer_name)

    ckpt_path = None
    if os.path.isfile(pref):
        ckpt_path = pref
    else:
        cands = sorted(glob.glob(os.path.join(base, "*.pth")))
        if cands:
            ckpt_path = cands[0]

    if ckpt_path is None:
        print("⚠️  No pretrained weights found in logs/pretrained/. Training from scratch.")
        return

    print(f"✅ Loading pretrained weights: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location=device)

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    new_sd = OrderedDict()
    for k, v in sd.items():
        new_sd[k[7:]] = v if k.startswith("module.") else v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print("ℹ️  Missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("ℹ️  Unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")
    if not missing and not unexpected:
        print("🎯 Weights loaded perfectly.")

# =========================
# Evaluation helpers
# =========================
@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, pids, camids = [], [], []
    for imgs, pid, cam in tqdm(loader, desc="Extract"):
        imgs = imgs.to(device, non_blocking=True)
        f, _ = model(imgs)
        feats.append(f.cpu().numpy())
        pids.extend(pid.numpy())
        camids.extend(cam.numpy())
    return np.concatenate(feats, 0), np.array(pids), np.array(camids)

def compute_distmat(Q, G):
    # ✅ fixed keepdim → keepdims (NumPy uses plural)
    qn = (Q ** 2).sum(1, keepdims=True)
    gn = (G ** 2).sum(1, keepdims=True)
    d = qn + gn.T - 2 * (Q @ G.T)
    d = np.maximum(d, 0)
    return np.sqrt(d)

def evaluate(distmat, q_pids, g_pids, q_cam, g_cam, max_rank=20):
    num_q, _ = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, None])

    all_cmc, all_ap = [], []
    for i in range(num_q):
        order = indices[i]
        keep = ~((g_pids[order] == q_pids[i]) & (g_cam[order] == q_cam[i]))
        m = matches[i][keep]
        if not np.any(m):
            continue

        cmc = m.cumsum()
        cmc[cmc > 1] = 1
        if len(cmc) < max_rank:
            cmc = np.pad(cmc, (0, max_rank - len(cmc)))
        all_cmc.append(cmc[:max_rank])

        prec = np.cumsum(m) / (np.arange(len(m)) + 1)
        ap = float((prec * m).sum() / m.sum())
        all_ap.append(ap)

    if len(all_cmc) == 0:
        return np.zeros(max_rank), 0.0

    return np.mean(np.stack(all_cmc), axis=0), float(np.mean(all_ap))

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["duke", "market"], default="duke")
    parser.add_argument("--target", choices=["duke", "market"], default="market")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backbone", choices=["resnet50", "resnet101"], default="resnet50")
    parser.add_argument("--prefer_ckpt", type=str, default="resnet101_ibn_a.pth")
    args = parser.parse_args()

    # transforms
    train_t = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    eval_t = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # datasets
    s = DATASETS[args.source]
    t = DATASETS[args.target]

    print("Source train:", s["train"])
    print("Target eval (query/gallery):", t["query"], "|", t["gallery"])

    ds_train = ReIDDataset(s["train"], transform=train_t, is_train=True)
    dt_query = ReIDDataset(t["query"], transform=eval_t)
    dt_gall  = ReIDDataset(t["gallery"], transform=eval_t)

    print(f"[check] {len(ds_train)} source images | classes={ds_train.num_classes}")
    print(f"[check] {len(dt_query)} target query images")
    print(f"[check] {len(dt_gall)} target gallery images")

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_query = DataLoader(dt_query, batch_size=128, shuffle=False,
                          num_workers=args.workers, pin_memory=True)
    dl_gall  = DataLoader(dt_gall,  batch_size=128, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    num_classes = ds_train.num_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReIDResNet(args.backbone, num_classes=num_classes).to(device)

    load_pretrained_if_available(model, device=device, prefer_name=args.prefer_ckpt)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    loss_fn = MultiLoss(alpha=0.7, margin=0.3)

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for imgs, pids, _ in tqdm(dl_train, desc=f"Epoch {ep}/{args.epochs}"):
            imgs = imgs.to(device, non_blocking=True)
            pids = pids.to(device, non_blocking=True).long()

            feats, logits = model(imgs)
            loss = loss_fn(feats, logits, pids)

            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item()

        print(f"Epoch {ep}: loss={running/len(dl_train):.4f}")

        if ep % 5 == 0 or ep == args.epochs:
            print("Evaluating…")
            qf, qpid, qcam = extract_features(model, dl_query, device)
            gf, gpid, gcam = extract_features(model, dl_gall,  device)
            dist = compute_distmat(qf, gf)
            cmc, mAP = evaluate(dist, qpid, gpid, qcam, gcam, max_rank=20)
            print(f"mAP: {mAP:.4f} | R1: {cmc[0]:.4f} | R5: {cmc[4]:.4f} | R10: {cmc[9]:.4f}")

    # save final (target_to_source format)
    os.makedirs("/home/tq_sundas/main/checkpoints", exist_ok=True)
    out = f"/home/tq_sundas/main/checkpoints/reid_{args.target}_to_{args.source}.pth"
    torch.save(model.state_dict(), out)
    print("Saved:", out)

if __name__ == "__main__":
    main()
