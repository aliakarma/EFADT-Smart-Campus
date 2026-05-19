"""
EFADT — Flower Simulation Runner
==================================
Runs the full federated learning simulation using Flower's simulation API.
No network connections required — all clients run in-process.

Usage:
    python -m federated.simulation --config configs/hyperparams.yaml
    python -m federated.simulation --n-rounds 10 --n-buildings 3  # quick test
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import flwr as fl
import mlflow
import numpy as np
import pandas as pd
import torch
import yaml

from federated.client import create_client_fn
from federated.server import build_server_strategy, weighted_average
from models.lstm.architecture import build_model
from federated.dp_mechanism import compute_sigma, estimate_total_privacy_budget

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_building_data(
    data_dir: str,
    n_buildings: int,
    building_config_path: str,
) -> dict:
    """
    Load per-building parquet files into a dict.

    Returns
    -------
    dict : {building_id: pd.DataFrame}
    """
    with open(building_config_path) as f:
        import yaml
        building_cfg = yaml.safe_load(f)["buildings"]

    building_ids = list(building_cfg.keys())[:n_buildings]
    data = {}

    for bid in building_ids:
        parquet_path = os.path.join(data_dir, f"{bid}.parquet")
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            data[bid] = df
            logger.info(f"Loaded {bid}: {len(df):,} records")
        else:
            logger.warning(f"No data for {bid} at {parquet_path}; skipping")

    if not data:
        raise FileNotFoundError(
            f"No building data found in {data_dir}. "
            "Run: python -m data.generation.generate_dataset"
        )
    return data


def run_simulation(
    config: dict,
    building_data: dict,
    n_rounds: Optional[int] = None,
    apply_dp: bool = True,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> dict:
    """
    Run the EFADT federated simulation.

    Parameters
    ----------
    config : dict
        Full hyperparameter config.
    building_data : dict
        {building_id: pd.DataFrame}
    n_rounds : int, optional
        Override number of FL rounds from config.
    apply_dp : bool
        Whether to apply DP (set False for -DP ablation).
    device : torch.device, optional
    seed : int

    Returns
    -------
    dict : Convergence summary metrics.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if device is None:
        device = torch.device("cpu")  # Simulation runs on CPU by default

    fl_cfg = config["federated"]
    n_rounds = n_rounds or fl_cfg["num_rounds"]
    n_clients = len(building_data)

    logger.info(f"Starting EFADT FL simulation | Rounds={n_rounds} | Buildings={n_clients} | DP={apply_dp}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "efadt-campus")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"fl_seed{seed}_rounds{n_rounds}_dp{apply_dp}"):
        # Log hyperparameters
        sigma_val = compute_sigma(config["dp"]["epsilon"], config["dp"]["delta"])
        mlflow.log_params({
            "seed": seed,
            "n_rounds": n_rounds,
            "n_buildings": n_clients,
            "apply_dp": apply_dp,
            "epsilon": config["dp"]["epsilon"],
            "sigma": sigma_val,
            "hidden_size": config["lstm"]["hidden_size"],
            "local_epochs": config["lstm"]["local_epochs"],
            "lambda_e": config["agent"]["lambda_e"],
            "lambda_c": config["agent"]["lambda_c"],
            "lambda_d": config["agent"]["lambda_d"],
        })

        checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "models/lstm/checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Build strategy
        strategy = build_server_strategy(config, checkpoint_dir=checkpoint_dir, n_clients=n_clients)

        # Override evaluate_metrics_aggregation_fn for weighted MAE
        strategy.evaluate_metrics_aggregation_fn = weighted_average

        # Build client factory
        client_fn = create_client_fn(
            building_data=building_data,
            config=config,
            apply_dp=apply_dp,
            device=device,
            seed=seed,
        )

        # Run Flower simulation
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=n_clients,
            config=fl.server.ServerConfig(num_rounds=n_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )

        # Extract convergence summary
        summary = strategy.get_convergence_summary()
        summary["history"] = history
        summary["n_rounds"] = n_rounds
        summary["n_buildings"] = n_clients
        summary["apply_dp"] = apply_dp

        # Compute and log DP budget
        sigma = compute_sigma(config["dp"]["epsilon"], config["dp"]["delta"])
        total_eps = estimate_total_privacy_budget(
            epsilon_per_round=config["dp"]["epsilon"],
            n_rounds=n_rounds,
            delta=config["dp"]["delta"],
            composition="renyi",
        )
        logger.info(f"DP accounting: per-round ε={config['dp']['epsilon']}, σ={sigma:.4f}, ε_total={total_eps:.3f}")

        # Log per-round MAE from strategy
        for round_info in strategy.round_metrics:
            mlflow.log_metric("global_mae", round_info["global_mae"], step=round_info["round"])

        if summary:
            mlflow.log_metrics({
                "convergence_round": summary.get("convergence_round") or -1,
                "final_mae": summary.get("final_mae", -1),
                "best_mae": summary.get("best_mae", -1),
            })

        if apply_dp:
            mlflow.log_metrics({
                "dp_sigma": sigma,
                "dp_epsilon_per_round": config["dp"]["epsilon"],
                "dp_epsilon_total_renyi": total_eps,
            })

        # Unpack best round global parameters to Bxx_best.pt for evaluation compatibility
        best_round = None
        if strategy.round_metrics:
            best_round_info = min(strategy.round_metrics, key=lambda x: x["global_mae"])
            best_round = best_round_info["round"]
        else:
            best_round = n_rounds

        best_pkl_path = os.path.join(checkpoint_dir, f"global_round_{best_round:04d}.pkl")
        if os.path.exists(best_pkl_path):
            import pickle
            logger.info(f"Unpacking best global model checkpoint from {best_pkl_path} for all buildings...")
            with open(best_pkl_path, "rb") as f:
                ckpt_data = pickle.load(f)
            ndarrays = ckpt_data["parameters"]

            from models.lstm.architecture import build_model
            dummy_model = build_model(config, device=torch.device("cpu"))
            params_dict = zip(dummy_model.state_dict().keys(), ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}

            for bid in building_data.keys():
                pt_path = os.path.join(checkpoint_dir, f"{bid}_best.pt")
                torch.save({
                    "epoch": best_round,
                    "model_state_dict": state_dict,
                    "val_mae": summary.get("best_mae", -1),
                    "building_id": bid,
                }, pt_path)
                logger.info(f"Saved FL model checkpoint for {bid} to {pt_path}")
        else:
            logger.warning(f"Could not find best round global pickle at {best_pkl_path} to unpack.")

        # Log checkpoint directory as artifact
        if os.path.exists(checkpoint_dir):
            mlflow.log_artifacts(checkpoint_dir, artifact_path="checkpoints")

    logger.info(f"Simulation complete. Summary: {summary}")
    return summary


def main():
    from typing import Optional

    parser = argparse.ArgumentParser(description="Run EFADT FL simulation")
    parser.add_argument("--config", default="configs/hyperparams.yaml")
    parser.add_argument("--building-config", default="configs/building_params.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--n-rounds", type=int, default=None)
    parser.add_argument("--n-buildings", type=int, default=12)
    parser.add_argument("--no-dp", action="store_true", help="Disable differential privacy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    building_data = load_building_data(
        args.data_dir, args.n_buildings, args.building_config
    )

    summary = run_simulation(
        config=config,
        building_data=building_data,
        n_rounds=args.n_rounds,
        apply_dp=not args.no_dp,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("EFADT FL Simulation Complete")
    print("=" * 60)
    print(f"Convergence round:  {summary.get('convergence_round', 'Not reached')}")
    print(f"Final global MAE:   {summary.get('final_mae', 'N/A'):.3f} persons")
    print(f"Best global MAE:    {summary.get('best_mae', 'N/A'):.3f} persons")
    print("=" * 60)


if __name__ == "__main__":
    main()
