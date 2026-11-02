from __future__ import absolute_import

import numpy as np
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Using manual t-test implementation.")


def paired_t_test(baseline_scores, aida_scores, alpha=0.05, verbose=True):
    """
    Perform a paired t-test to compare baseline (IDM++) and AIDA scores.
    
    This test is appropriate when the same experiments/runs are compared
    between baseline and AIDA methods (e.g., same random seeds, same datasets).
    
    Args:
        baseline_scores: List or array of baseline scores (e.g., mAP values from multiple runs)
        aida_scores: List or array of AIDA scores (same length as baseline_scores)
        alpha: Significance level (default: 0.05)
        verbose: If True, print test results (default: True)
    
    Returns:
        dict: Dictionary containing:
            - 't_statistic': t-value
            - 'p_value': p-value
            - 'mean_diff': Mean difference (aida - baseline)
            - 'std_diff': Standard deviation of differences
            - 'is_significant': Boolean indicating if p < alpha
            - 'confidence_interval': 95% confidence interval for the mean difference
    """
    baseline_scores = np.array(baseline_scores)
    aida_scores = np.array(aida_scores)
    
    # Check that arrays have the same length
    if len(baseline_scores) != len(aida_scores):
        raise ValueError(
            f"baseline_scores and aida_scores must have the same length. "
            f"Got {len(baseline_scores)} and {len(aida_scores)}"
        )
    
    # Check minimum sample size
    n = len(baseline_scores)
    if n < 2:
        raise ValueError("Need at least 2 paired samples for t-test")
    
    # Compute differences
    differences = aida_scores - baseline_scores
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)  # Sample standard deviation
    
    # Perform paired t-test
    if SCIPY_AVAILABLE:
        # Use SciPy's paired t-test
        t_statistic, p_value = stats.ttest_rel(aida_scores, baseline_scores)
        
        # Compute confidence interval
        se_diff = std_diff / np.sqrt(n)
        df = n - 1
        t_critical = stats.t.ppf(1 - alpha / 2, df)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
    else:
        # Manual implementation
        se_diff = std_diff / np.sqrt(n)
        t_statistic = mean_diff / se_diff if se_diff > 0 else 0.0
        
        # Compute p-value using t-distribution
        df = n - 1
        if abs(t_statistic) < 1e-10:
            p_value = 1.0
        else:
            # Approximate p-value using t-distribution CDF
            # For simplicity, we'll use a basic approximation
            # In practice, you'd use scipy.stats.t.cdf
            p_value = 2 * (1 - _t_cdf_approx(abs(t_statistic), df))
        
        # Compute confidence interval
        t_critical = _t_ppf_approx(1 - alpha / 2, df)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
    
    is_significant = p_value < alpha
    
    results = {
        't_statistic': float(t_statistic),
        'p_value': float(p_value),
        'mean_diff': float(mean_diff),
        'std_diff': float(std_diff),
        'is_significant': is_significant,
        'confidence_interval': (float(ci_lower), float(ci_upper)),
        'n_samples': n,
        'alpha': alpha
    }
    
    if verbose:
        print("\n" + "="*60)
        print("Paired t-test Results")
        print("="*60)
        print(f"Baseline mean: {np.mean(baseline_scores):.4f} ± {np.std(baseline_scores, ddof=1):.4f}")
        print(f"AIDA mean:     {np.mean(aida_scores):.4f} ± {np.std(aida_scores, ddof=1):.4f}")
        print(f"Mean difference: {mean_diff:.4f} (AIDA - Baseline)")
        print(f"Std of differences: {std_diff:.4f}")
        print(f"\nt-statistic: {t_statistic:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"\nSignificance level: α = {alpha}")
        if is_significant:
            print(f"✓ Statistically significant difference (p < {alpha})")
            if mean_diff > 0:
                print("  → AIDA performs significantly better than baseline")
            else:
                print("  → Baseline performs significantly better than AIDA")
        else:
            print(f"✗ No statistically significant difference (p ≥ {alpha})")
        print("="*60 + "\n")
    
    return results


def _t_cdf_approx(t, df):
    """
    Approximate CDF of t-distribution.
    This is a simple approximation; for production use, SciPy is recommended.
    """
    # Simple approximation using normal distribution for large df
    if df > 30:
        from scipy.special import erf
        return 0.5 * (1 + erf(t / np.sqrt(2)))
    else:
        # For small df, use a rough approximation
        # This is not very accurate but better than nothing
        z = t / np.sqrt(df / (df - 2))
        from scipy.special import erf
        return 0.5 * (1 + erf(z / np.sqrt(2)))


def _t_ppf_approx(prob, df):
    """
    Approximate inverse CDF (percent point function) of t-distribution.
    This is a simple approximation; for production use, SciPy is recommended.
    """
    # Simple approximation using normal distribution for large df
    if df > 30:
        from scipy.special import erfinv
        return np.sqrt(2) * erfinv(2 * prob - 1)
    else:
        # For small df, use a rough approximation
        from scipy.special import erfinv
        z = np.sqrt(2) * erfinv(2 * prob - 1)
        return z * np.sqrt(df / (df - 2))


def load_scores_from_json(json_path, metric='mAP'):
    """
    Load scores from a JSON file.
    
    Expected JSON format:
    {
        "runs": [
            {"mAP": 0.85, "rank1": 0.92, ...},
            {"mAP": 0.86, "rank1": 0.93, ...},
            ...
        ]
    }
    
    Or simple array format:
    [0.85, 0.86, 0.87, ...]
    
    Args:
        json_path: Path to JSON file
        metric: Metric name to extract (default: 'mAP')
    
    Returns:
        numpy array of scores
    """
    import json
    import os.path as osp
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        # Simple array format
        return np.array(data)
    elif isinstance(data, dict):
        if 'runs' in data:
            # Extract metric from each run
            scores = [run[metric] for run in data['runs'] if metric in run]
            return np.array(scores)
        elif metric in data:
            # Direct metric key
            return np.array(data[metric])
        else:
            raise ValueError(f"Metric '{metric}' not found in JSON file")
    else:
        raise ValueError(f"Unexpected JSON format in {json_path}")


def compare_results(baseline_path, aida_path, metric='mAP', alpha=0.05):
    """
    Compare results from two JSON files using paired t-test.
    
    Args:
        baseline_path: Path to baseline results JSON file
        aida_path: Path to AIDA results JSON file
        metric: Metric name to compare (default: 'mAP')
        alpha: Significance level (default: 0.05)
    
    Returns:
        dict: Results from paired_t_test
    """
    baseline_scores = load_scores_from_json(baseline_path, metric=metric)
    aida_scores = load_scores_from_json(aida_path, metric=metric)
    
    print(f"Loaded {len(baseline_scores)} baseline scores from {baseline_path}")
    print(f"Loaded {len(aida_scores)} AIDA scores from {aida_path}")
    
    return paired_t_test(baseline_scores, aida_scores, alpha=alpha, verbose=True)


