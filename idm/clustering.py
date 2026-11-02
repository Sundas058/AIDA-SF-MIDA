"""Epoch-wise target pseudo-labels using k-reciprocal Jaccard distances."""
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.cluster import DBSCAN
from .utils.rerank import compute_jaccard_distance


def cluster_target(embeddings, eps=0.6, min_samples=4, k1=30, k2=6):
    if len(embeddings) < min_samples:
        raise ValueError("Target split has fewer samples than DBSCAN min_samples.")
    z = F.normalize(embeddings.detach().cpu(), dim=1)
    distances = compute_jaccard_distance(z, k1=min(k1, len(z) - 1), k2=min(k2, len(z)), print_flag=False)
    distances = distances.numpy()
    distances = np.maximum((distances + distances.T) * 0.5, 0)
    np.fill_diagonal(distances, 0)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1).fit_predict(distances)
    clusters = sorted(set(labels) - {-1})
    if len(clusters) < 2:
        raise ValueError(f"DBSCAN found {len(clusters)} usable clusters; at least two are needed for identity/triplet training.")
    centers = torch.stack([z[labels == label].mean(0) for label in clusters])
    return labels, F.normalize(centers, dim=1)

