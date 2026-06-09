from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
import json
import sys

sys.path.append(".")
from idm.utils.stats_tests import paired_t_test, load_scores_from_json, compare_results
from idm.utils.serialization import read_json, write_json


def parse_log_file(log_path, metric='mAP'):
    """
    Parse a log file to extract metric values.
    
    Looks for lines like:
    - "Mean AP: 85.2%"
    - "mAP: 0.852"
    - "Rank-1: 92.3%"
    
    Args:
        log_path: Path to log file
        metric: Metric to extract ('mAP', 'rank1', 'rank5', 'rank10')
    
    Returns:
        list: List of metric values found in the log file
    """
    scores = []
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if metric.lower() == 'map':
                # Look for "Mean AP: XX.X%" or "mAP: 0.XXX"
                if 'Mean AP:' in line or 'mAP:' in line.lower():
                    # Extract number
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'AP' in part.upper() or 'map' in part.lower():
                            if i + 1 < len(parts):
                                value_str = parts[i + 1].rstrip('%').rstrip(',')
                                try:
                                    value = float(value_str)
                                    # Convert percentage to decimal if needed
                                    if value > 1.0:
                                        value = value / 100.0
                                    scores.append(value)
                                    break
                                except ValueError:
                                    pass
            
            elif metric.lower() == 'rank1':
                # Look for "Rank-1: XX.X%" or "rank1: 0.XXX"
                if 'rank-1' in line.lower() or 'rank1' in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'rank-1' in part.lower() or 'rank1' in part.lower():
                            if i + 1 < len(parts):
                                value_str = parts[i + 1].rstrip('%').rstrip(',')
                                try:
                                    value = float(value_str)
                                    if value > 1.0:
                                        value = value / 100.0
                                    scores.append(value)
                                    break
                                except ValueError:
                                    pass
    
    return scores


def extract_results_from_logs(log_dir, metric='mAP'):
    """
    Extract results from multiple log files in a directory.
    
    Args:
        log_dir: Directory containing log files
        metric: Metric to extract
    
    Returns:
        list: List of metric values
    """
    scores = []
    
    if not osp.isdir(log_dir):
        print(f"Warning: {log_dir} is not a directory")
        return scores
    
    # Look for log files
    log_files = []
    for fname in os.listdir(log_dir):
        if fname.endswith('.txt') and ('log' in fname.lower() or 'test' in fname.lower()):
            log_files.append(osp.join(log_dir, fname))
    
    for log_file in sorted(log_files):
        file_scores = parse_log_file(log_file, metric=metric)
        if file_scores:
            # Take the last value (final result)
            scores.append(file_scores[-1])
    
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline (IDM++) and AIDA results using paired t-test"
    )
    
    # Input options
    parser.add_argument('--baseline', type=str, required=True,
                        help='Path to baseline results (JSON file or directory with log files)')
    parser.add_argument('--aida', type=str, required=True,
                        help='Path to AIDA results (JSON file or directory with log files)')
    parser.add_argument('--metric', type=str, default='mAP',
                        choices=['mAP', 'rank1', 'rank5', 'rank10'],
                        help='Metric to compare (default: mAP)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level (default: 0.05)')
    
    # Output options
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save comparison results as JSON (optional)')
    parser.add_argument('--baseline-is-dir', action='store_true',
                        help='Treat baseline path as directory of log files')
    parser.add_argument('--aida-is-dir', action='store_true',
                        help='Treat AIDA path as directory of log files')
    
    args = parser.parse_args()
    
    # Load baseline scores
    print(f"\nLoading baseline scores from: {args.baseline}")
    if args.baseline_is_dir or osp.isdir(args.baseline):
        baseline_scores = extract_results_from_logs(args.baseline, metric=args.metric)
        if not baseline_scores:
            print(f"Warning: No scores found in {args.baseline}")
            return
    else:
        try:
            baseline_scores = load_scores_from_json(args.baseline, metric=args.metric)
        except Exception as e:
            print(f"Error loading baseline scores: {e}")
            return
    
    # Load AIDA scores
    print(f"Loading AIDA scores from: {args.aida}")
    if args.aida_is_dir or osp.isdir(args.aida):
        aida_scores = extract_results_from_logs(args.aida, metric=args.metric)
        if not aida_scores:
            print(f"Warning: No scores found in {args.aida}")
            return
    else:
        try:
            aida_scores = load_scores_from_json(args.aida, metric=args.metric)
        except Exception as e:
            print(f"Error loading AIDA scores: {e}")
            return
    
    # Perform paired t-test
    print(f"\nComparing {args.metric} scores:")
    print(f"  Baseline: {len(baseline_scores)} runs")
    print(f"  AIDA:     {len(aida_scores)} runs")
    
    results = paired_t_test(
        baseline_scores, 
        aida_scores, 
        alpha=args.alpha, 
        verbose=True
    )
    
    # Save results if requested
    if args.output:
        output_data = {
            'metric': args.metric,
            'baseline_path': args.baseline,
            'aida_path': args.aida,
            'baseline_scores': baseline_scores.tolist() if hasattr(baseline_scores, 'tolist') else list(baseline_scores),
            'aida_scores': aida_scores.tolist() if hasattr(aida_scores, 'tolist') else list(aida_scores),
            'test_results': results
        }
        write_json(output_data, args.output)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    import os
    main()

