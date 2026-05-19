#!/usr/bin/env python
"""
scripts/multi_seed_eval.py
============================
Run the full evaluation pipeline for multiple seeds and compute mean±std,
confidence intervals, and Wilcoxon significance tests vs each baseline.

Usage:
    python scripts/multi_seed_eval.py \
        --seeds 42 0 1 \
        --output results/ablation/multi_seed_results.json
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import subprocess
import numpy as np
from scipy.stats import wilcoxon, norm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_one_seed(seed: int, extra_args: list) -> dict:
    """Run evaluate_checkpoint.py for one seed and return parsed JSON."""
    out_path = f"results/seeds/results_seed{seed}.json"
    import os
    if os.path.exists(out_path):
        logger.info(f"Seed {seed} results already exist at {out_path}; loading directly.")
        with open(out_path) as f:
            return json.load(f)

    cmd = [
        sys.executable, "-m", "scripts.evaluate_checkpoint",
        "--seed", str(seed),
        "--output", out_path,
    ] + extra_args

    logger.info(f"Running seed={seed}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    with open(out_path) as f:
        return json.load(f)


def aggregate_seeds(seed_results: list[dict]) -> dict:
    """Compute mean ± std ± 95% CI for each metric across seeds."""
    metrics = ["ERR", "CCS", "CSS", "MAE", "tau"]
    variants = list(seed_results[0]["results"].keys())
    aggregated = {}

    for variant in variants:
        aggregated[variant] = {}
        for metric in metrics:
            values = [r["results"][variant].get(metric) for r in seed_results
                      if r["results"].get(variant, {}).get(metric) is not None]
            if not values:
                aggregated[variant][metric] = {"mean": None, "std": None, "ci95": None, "n": 0}
                continue
            arr = np.array(values, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            ci95 = float(norm.ppf(0.975) * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            aggregated[variant][metric] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "ci95": round(ci95, 4),
                "n": len(arr),
                "values": [round(v, 4) for v in arr.tolist()],
            }

    return aggregated


def significance_tests(aggregated: dict, reference: str = "EFADT (Full)") -> dict:
    """
    Wilcoxon signed-rank test between reference variant and each other variant.
    Reports p-value and Cohen's d effect size.
    """
    from evaluation.metrics import significance_test
    tests = {}
    if reference not in aggregated:
        return tests

    ref_mae_values = aggregated[reference].get("MAE", {}).get("values", [])
    if not ref_mae_values:
        return tests

    for variant, metrics in aggregated.items():
        if variant == reference:
            continue
        other_values = metrics.get("MAE", {}).get("values", [])
        if len(other_values) < 2 or len(ref_mae_values) < 2:
            continue
        if len(ref_mae_values) != len(other_values):
            logger.warning(f"Unequal sample sizes for {reference} vs {variant}")
            continue

        res = significance_test(ref_mae_values, other_values)
        tests[f"{reference} vs {variant}"] = {
            "metric": "MAE",
            "wilcoxon_stat": round(res["wilcoxon_stat"], 4),
            "p_value": round(res["p_value"], 6),
            "cohens_d": round(res["cohens_d"], 4),
            "significant_p05": res["significant_p05"],
        }
    return tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1])
    parser.add_argument("--output", default="results/ablation/multi_seed_results.json")
    parser.add_argument("--checkpoint-dir", default="models/lstm/checkpoints")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--test-months", nargs="+", type=int, default=[10, 11, 12])
    parser.add_argument("--centralized-epochs", type=int, default=1)
    args = parser.parse_args()

    Path("results/seeds").mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    extra = [
        "--checkpoint-dir", args.checkpoint_dir,
        "--data-dir", args.data_dir,
        "--centralized-epochs", str(args.centralized_epochs),
        "--test-months",
    ] + [str(m) for m in args.test_months]

    seed_results = [run_one_seed(s, extra) for s in args.seeds]
    aggregated = aggregate_seeds(seed_results)
    sig_tests = significance_tests(aggregated)

    output = {
        "seeds": args.seeds,
        "n_seeds": len(args.seeds),
        "aggregated": aggregated,
        "significance_tests": sig_tests,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Variant':<28} {'ERR%':>12} {'CCS':>10} {'CSS':>10} {'MAE':>14}")
    print("-" * 90)
    for variant, metrics in aggregated.items():
        def fmt(m):
            v = m.get("mean"); s = m.get("std")
            if v is None: return "      —"
            return f"{v:.3f}+/-{s:.3f}"
        print(f"{variant:<28} {fmt(metrics['ERR']):>12} {fmt(metrics['CCS']):>10} "
              f"{fmt(metrics['CSS']):>10} {fmt(metrics['MAE']):>14}")
    print("=" * 90)

    print("\nSignificance Tests (Wilcoxon, MAE):")
    for test, result in sig_tests.items():
        sig = "[SIG]" if result["significant_p05"] else "[NS]"
        print(f"  {sig} {test}: p={result['p_value']:.4f}, d={result['cohens_d']:.3f}")

    logger.info(f"Multi-seed results written to {args.output}")


if __name__ == "__main__":
    main()
