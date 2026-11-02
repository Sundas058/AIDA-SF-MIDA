"""Aggregate measured retrieval metrics and compare paired runs."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--baseline", nargs="+")
    parser.add_argument("--output", default="runs/summary.json")
    args = parser.parse_args()
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.inputs]
    summary = {"runs": len(runs), "inputs": args.inputs, "metrics": {}}
    for metric in ("rank1", "rank5", "rank10", "mAP"):
        scores = np.asarray([run[metric] for run in runs], dtype=float)
        if not np.isfinite(scores).all():
            parser.error(f"{metric} contains non-finite values")
        summary["metrics"][metric] = {"mean": float(scores.mean()),
            "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0}
    if args.baseline:
        if len(args.baseline) != len(runs) or len(runs) < 2:
            parser.error("A paired comparison requires at least two matched run files per method.")
        baseline = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.baseline]
        if any(a.get("seed") != b.get("seed") or a.get("target") != b.get("target")
               for a, b in zip(runs, baseline)):
            parser.error("Paired run files must have matching seeds and targets in the supplied order.")
        summary["paired_tests"] = {}
        for metric in summary["metrics"]:
            a = np.asarray([r[metric] for r in runs])
            b = np.asarray([r[metric] for r in baseline])
            if np.array_equal(a, b):
                statistic, pvalue = 0.0, 1.0
            else:
                result = ttest_rel(a, b)
                statistic, pvalue = float(result.statistic), float(result.pvalue)
            summary["paired_tests"][metric] = {
                "mean_difference": float((a - b).mean()),
                "t_statistic": statistic if np.isfinite(statistic) else None,
                "p_value": pvalue if np.isfinite(pvalue) else None}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2, allow_nan=False)
    Path(args.output).write_text(text, encoding="utf-8")
    print(text)

if __name__ == "__main__":
    main()

