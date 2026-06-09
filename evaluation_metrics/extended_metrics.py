"""
Extended Evaluation Metrics for Person Re-ID

Additional metrics beyond Rank-1 and mAP:
- CMC (Cumulative Matching Characteristics) - already computed, but extended reporting
- NMI (Normalized Mutual Information)
- Silhouette Score
- FID (Fréchet Inception Distance) - optional
"""

from __future__ import absolute_import
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
import warnings

try:
    from scipy.linalg import sqrtm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available, FID computation will be disabled")


def compute_nmi(features, labels, n_clusters=None):
    """
    Compute Normalized Mutual Information (NMI) between true labels and cluster assignments.
    
    Args:
        features: [N, D] feature matrix
        labels: [N] true identity labels
        n_clusters: Number of clusters for KMeans. If None, uses number of unique labels.
    
    Returns:
        float: NMI score (0-1, higher is better)
    """
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    
    # Normalize features
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # Cluster features using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_norm)
    
    # Compute NMI
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    return nmi


def compute_silhouette(features, labels, sample_size=None):
    """
    Compute Silhouette Score for feature clustering quality.
    
    Args:
        features: [N, D] feature matrix
        labels: [N] true identity labels (used as ground truth clusters)
        sample_size: If not None, randomly sample this many points for efficiency
    
    Returns:
        float: Silhouette score (-1 to 1, higher is better)
    """
    # Normalize features
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # Sample if requested (for large datasets)
    if sample_size is not None and len(features_norm) > sample_size:
        indices = np.random.choice(len(features_norm), sample_size, replace=False)
        features_norm = features_norm[indices]
        labels = labels[indices]
    
    # Compute silhouette score using true labels as clusters
    try:
        silhouette = silhouette_score(features_norm, labels, metric='cosine')
    except ValueError:
        # Fallback to euclidean if cosine fails
        silhouette = silhouette_score(features_norm, labels, metric='euclidean')
    
    return silhouette


def compute_fid(features1, features2):
    """
    Compute Fréchet Inception Distance (FID) between two feature distributions.
    
    FID measures the distance between two multivariate Gaussians fitted to the features.
    Lower is better (0 means identical distributions).
    
    Args:
        features1: [N1, D] feature matrix from distribution 1
        features2: [N2, D] feature matrix from distribution 2
    
    Returns:
        float: FID score (lower is better)
    """
    if not SCIPY_AVAILABLE:
        warnings.warn("scipy not available, cannot compute FID")
        return None
    
    # Normalize features
    features1 = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
    features2 = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)
    
    # Compute mean and covariance
    mu1 = np.mean(features1, axis=0)
    sigma1 = np.cov(features1, rowvar=False)
    
    mu2 = np.mean(features2, axis=0)
    sigma2 = np.cov(features2, rowvar=False)
    
    # Compute FID
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    
    # Check for complex numbers (shouldn't happen, but handle it)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def compute_extended_metrics(query_features, gallery_features, 
                            query_labels, gallery_labels,
                            distmat=None,
                            compute_fid_flag=False,
                            silhouette_sample_size=5000):
    """
    Compute extended evaluation metrics.
    
    Args:
        query_features: [N_q, D] query feature matrix
        gallery_features: [N_g, D] gallery feature matrix
        query_labels: [N_q] query identity labels
        gallery_labels: [N_g] gallery identity labels
        distmat: [N_q, N_g] distance matrix (optional, for efficiency)
        compute_fid_flag: If True, compute FID (requires scipy)
        silhouette_sample_size: Sample size for silhouette computation
    
    Returns:
        dict: Dictionary containing all computed metrics
    """
    metrics = {}
    
    # Combine features and labels for clustering metrics
    all_features = np.vstack([query_features, gallery_features])
    all_labels = np.concatenate([query_labels, gallery_labels])
    
    # NMI: Normalized Mutual Information
    try:
        metrics['nmi'] = compute_nmi(all_features, all_labels)
    except Exception as e:
        warnings.warn(f"Failed to compute NMI: {e}")
        metrics['nmi'] = None
    
    # Silhouette Score
    try:
        metrics['silhouette'] = compute_silhouette(
            all_features, all_labels, 
            sample_size=silhouette_sample_size
        )
    except Exception as e:
        warnings.warn(f"Failed to compute Silhouette: {e}")
        metrics['silhouette'] = None
    
    # FID (optional): Compare query and gallery distributions
    if compute_fid_flag:
        try:
            metrics['fid'] = compute_fid(query_features, gallery_features)
        except Exception as e:
            warnings.warn(f"Failed to compute FID: {e}")
            metrics['fid'] = None
    else:
        metrics['fid'] = None
    
    return metrics


def format_extended_metrics(metrics_dict):
    """
    Format extended metrics for printing.
    
    Args:
        metrics_dict: Dictionary from compute_extended_metrics
    
    Returns:
        str: Formatted string
    """
    lines = []
    
    if metrics_dict.get('nmi') is not None:
        lines.append(f"NMI: {metrics_dict['nmi']:.4f}")
    
    if metrics_dict.get('silhouette') is not None:
        lines.append(f"Silhouette: {metrics_dict['silhouette']:.4f}")
    
    if metrics_dict.get('fid') is not None:
        lines.append(f"FID: {metrics_dict['fid']:.4f}")
    
    return "\n".join(lines) if lines else "Extended metrics: N/A"

