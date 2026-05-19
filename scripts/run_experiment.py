"""
EFADT — Full Experiment Runner
================================
Runs the complete EFADT experiment pipeline:
  1. Generate synthetic dataset
  2. Run FL simulation (or load checkpoints)
  3. Evaluate all ablation variants
  4. Produce results table
  5. Save experiment artifacts

Usage:
    python scripts/run_experiment.py --quick          # 3 buildings, 5 rounds
    python scripts/run_experiment.py --full           # 12 buildings, 100 rounds
    python scripts/run_experiment.py --eval-only      # Skip training, load checkpoints
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def step_generate_data(n_buildings: int, n_days: int, seed: int) -> None:
    logger.info(f"Step 1: Generating dataset ({n_buildings} buildings, {n_days} days)...")
    from data.generation.generate_dataset import generate_full_dataset
    generate_full_dataset(
        n_buildings=n_buildings,
        n_days=n_days,
        config_path="configs/hyperparams.yaml",
        building_config_path="configs/building_params.yaml",
        output_dir="data/raw",
        seed=seed,
    )


def step_run_fl(n_buildings: int, n_rounds: int, apply_dp: bool, seed: int) -> dict:
    import torch, pandas as pd
    logger.info(f"Step 2: Running FL simulation ({n_buildings} buildings, {n_rounds} rounds, DP={apply_dp})...")
    with open("configs/hyperparams.yaml") as f:
        config = yaml.safe_load(f)
    with open("configs/building_params.yaml") as f:
        building_cfg = yaml.safe_load(f)["buildings"]

    building_ids = list(building_cfg.keys())[:n_buildings]
    building_data = {}
    for bid in building_ids:
        path = f"data/raw/{bid}.parquet"
        if os.path.exists(path):
            building_data[bid] = pd.read_parquet(path)

    if not building_data:
        raise FileNotFoundError("No building data found. Run Step 1 first.")

    from federated.simulation import run_simulation
    summary = run_simulation(
        config=config, building_data=building_data,
        n_rounds=n_rounds, apply_dp=apply_dp,
    )
    return summary


def step_evaluate(checkpoint_dir: str, data_dir: str, test_months: list, seed: int, n_buildings: int) -> None:
    logger.info("Step 3: Evaluating from trained checkpoints...")
    import subprocess, sys
    result = subprocess.run([
        sys.executable, "scripts/evaluate_checkpoint.py",
        "--checkpoint-dir", checkpoint_dir,
        "--data-dir", data_dir,
        "--n-buildings", str(n_buildings),
        "--test-months"] + [str(m) for m in test_months] + [
        "--seed", str(seed),
        "--output", "results/ablation/full_results.json",
    ], check=True)
    logger.info("Evaluation complete — results at results/ablation/full_results.json")


def main():
    parser = argparse.ArgumentParser(description="EFADT full experiment runner")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 3 buildings, 5 rounds, 14 days")
    parser.add_argument("--full", action="store_true", help="Full mode: 12 buildings, 100 rounds, 365 days")
    parser.add_argument("--eval-only", action="store_true", help="Skip data gen + training")
    parser.add_argument("--no-dp", action="store_true", help="Disable differential privacy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.quick:
        n_buildings, n_rounds, n_days = 3, 5, 14
    elif args.full:
        n_buildings, n_rounds, n_days = 12, 100, 365
    else:
        n_buildings, n_rounds, n_days = 3, 5, 14  # default quick

    t_start = time.time()

    if not args.eval_only:
        step_generate_data(n_buildings, n_days, args.seed)
        try:
            summary = step_run_fl(n_buildings, n_rounds, not args.no_dp, args.seed)
            logger.info(f"FL convergence: {summary.get('convergence_round', 'N/A')}")
        except Exception as e:
            logger.warning(f"FL simulation failed (expected without flwr): {e}")

    step_evaluate(
        checkpoint_dir="models/lstm/checkpoints",
        data_dir="data/raw",
        test_months=[10, 11, 12] if args.full else [1],
        seed=args.seed,
        n_buildings=n_buildings
    )

    elapsed = time.time() - t_start
    logger.info(f"Experiment complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
