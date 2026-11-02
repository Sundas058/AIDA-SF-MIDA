"""Camera-filtered CMC and mAP from normalized query/gallery embeddings."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score, silhouette_score


@torch.no_grad()
def extract_features(model, data_loader, print_freq=50):
    model.eval()
    features, labels = OrderedDict(), OrderedDict()
    device = next(model.parameters()).device
    for batch in data_loader:
        embeddings = model(batch[0].to(device)).detach().cpu()
        for name, feature, pid in zip(batch[1], embeddings, batch[2]):
            features[name], labels[name] = feature, int(pid)
    return features, labels


def retrieval_metrics(query_features, gallery_features, query, gallery, chunk_size=256):
    q = torch.nn.functional.normalize(torch.as_tensor(query_features).float(), dim=1)
    g = torch.nn.functional.normalize(torch.as_tensor(gallery_features).float(), dim=1)
    pids = np.asarray([row[1] for row in gallery])
    cameras = np.asarray([row[2] for row in gallery])
    cmcs, aps = [], []
    for start in range(0, len(q), chunk_size):
        distances = (2 - 2 * q[start:start + chunk_size] @ g.T).numpy()
        for local, distance in enumerate(distances):
            _, identity, camera, *_ = query[start + local]
            order = np.argsort(distance, kind="stable")
            keep = (pids[order] != -1) & ~((pids[order] == identity) & (cameras[order] == camera))
            match = pids[order[keep]] == identity
            if not match.any():
                continue
            cumulative = match.cumsum()
            aps.append(float(((cumulative / np.arange(1, len(match) + 1)) * match).sum() / match.sum()))
            cmcs.append([float(match[:rank].any()) for rank in (1, 5, 10)])
    if not aps:
        raise ValueError("No valid cross-camera query matches in the gallery.")
    mean_cmc = np.asarray(cmcs).mean(0)
    return dict(rank1=float(mean_cmc[0] * 100), rank5=float(mean_cmc[1] * 100),
                rank10=float(mean_cmc[2] * 100), mAP=float(np.mean(aps) * 100),
                valid_queries=len(aps), total_queries=len(query))


def representation_metrics(features, labels, clusters, seed=1):
    keep = np.asarray(clusters) >= 0
    if not keep.any():
        raise ValueError("No non-noise clusters for representation analysis.")
    result = {"NMI": float(normalized_mutual_info_score(np.asarray(labels)[keep], np.asarray(clusters)[keep]))}
    z = np.asarray(features)[keep]
    c = np.asarray(clusters)[keep]
    if 1 < len(set(c)) < len(c):
        result["silhouette"] = float(silhouette_score(z, c, metric="euclidean", sample_size=min(5000, len(z)), random_state=seed))
    return result


class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, output=None):
        features, _ = extract_features(self.model, data_loader)
        q = torch.stack([features[row[0]] for row in query])
        g = torch.stack([features[row[0]] for row in gallery])
        metrics = retrieval_metrics(q, g, query, gallery)
        print(json.dumps(metrics, indent=2))
        if output:
            Path(output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return (metrics["rank1"] / 100, metrics["mAP"] / 100) if cmc_flag else metrics["mAP"] / 100

