#!/usr/bin/env python
"""
scripts/run_ablations.py
=========================
Runs all ablation variants for a given seed and produces
results/ablation/ablation_seed{seed}.json.

Variants:
  EFADT (Full)           — Full EFADT with DP, XAI, DT
  -XAI                   — Trust score not displayed (τ = 0)
  -DT-WIF (DT-only)      — Naive persistence forecast, rule-based HVAC
  -DP                    — FL without Differential Privacy
  -MOO (energy-only)     — Utility weights override: lambda_e=1, lambda_c=0, lambda_d=0
  -FL (centralized)      — Centralized LSTM baseline on pooled dataset
  Rule-Based             — Baseline thermostat with naive persistence forecast

Usage:
    python scripts/run_ablations.py --seed 42
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import subprocess
import sys
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def retrain_no_dp(n_buildings: int, seed: int, config_path: str, data_dir: str, building_config: str, n_rounds: int = None) -> str:
    """Retrain FL without DP and save checkpoints to a separate directory."""
    ckpt_dir = f"models/lstm/checkpoints_no_dp_seed{seed}"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "federated.simulation",
        "--n-buildings", str(n_buildings),
        "--seed", str(seed),
        "--no-dp",
        "--config", config_path,
        "--building-config", building_config,
        "--data-dir", data_dir,
    ]
    if n_rounds is not None:
        cmd += ["--n-rounds", str(n_rounds)]

    logger.info(f"Retraining FL without DP... Command: {' '.join(cmd)}")
    env = {**os.environ, "CHECKPOINT_DIR": ckpt_dir}
    subprocess.run(cmd, check=True, env=env)
    return ckpt_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="configs/hyperparams.yaml")
    parser.add_argument("--building-config", default="configs/building_params.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--checkpoint-dir", default="models/lstm/checkpoints")
    parser.add_argument("--test-months", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", default="results/ablation")
    parser.add_argument("--n-rounds", type=int, default=None, help="Override rounds during no-DP retraining")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    n_buildings = config["data"]["n_buildings"]

    test_months = args.test_months if args.test_months is not None else config["data"].get("test_months", [10, 11, 12])

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    def eval_variant(ckpt_dir, output_file, extra_flags=None):
        cmd = [
            sys.executable, "scripts/evaluate_checkpoint.py",
            "--checkpoint-dir", ckpt_dir,
            "--data-dir", args.data_dir,
            "--config", args.config,
            "--building-config", args.building_config,
            "--seed", str(args.seed),
            "--output", output_file,
        ]
        if test_months:
            cmd += ["--test-months"] + [str(m) for m in test_months]
        if extra_flags:
            cmd += extra_flags
        
        logger.info(f"Evaluating checkpoints... Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        with open(output_file) as f:
            return json.load(f)

    # 1. Run standard evaluation on DP checkpoints (runs all variants except real -DP)
    logger.info("Evaluating default checkpoints (with DP)...")
    dp_output = f"{args.output_dir}/full_dp_eval_seed{args.seed}.json"
    dp_results = eval_variant(args.checkpoint_dir, dp_output)
    
    results = dp_results.get("results", {})

    # 2. Retrain and evaluate without DP to get true -DP ablation metrics
    logger.info("Retraining without DP for true -DP ablation metrics...")
    ckpt_no_dp = retrain_no_dp(
        n_buildings=n_buildings,
        seed=args.seed,
        config_path=args.config,
        data_dir=args.data_dir,
        building_config=args.building_config,
        n_rounds=args.n_rounds,
    )
    
    no_dp_output = f"{args.output_dir}/no_dp_eval_seed{args.seed}.json"
    no_dp_results = eval_variant(
        ckpt_no_dp, 
        no_dp_output, 
        extra_flags=["--variant", "EFADT (Full)"]
    )
    
    # Overwrite the -DP fallback with the actual evaluated no-DP metrics
    results["-DP"] = no_dp_results.get("results", {}).get("EFADT (Full)", {})

    # Write the consolidated ablation result
    output_path = f"{args.output_dir}/ablation_seed{args.seed}.json"
    with open(output_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "test_months": test_months,
            "results": results
        }, f, indent=2)
        
    logger.info(f"Consolidated ablation results written successfully to {output_path}")


if __name__ == "__main__":
    main()
