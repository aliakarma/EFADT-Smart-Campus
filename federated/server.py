"""
EFADT — Federated Aggregation Server
======================================
Implements the FedAvg aggregation server with differential privacy.

Responsibilities:
  - Broadcast global model θ^(r) to all building clients
  - Collect DP-noised updates Δθ̃_b
  - Compute weighted average: θ^(r+1) = Σ_b (n_b/n) * θ_b^(r+1)
  - Track convergence via global validation MAE
  - Save model checkpoints per round
"""

from __future__ import annotations

import logging
import mlflow
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import Metrics, Parameters, Scalar
from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)


class EFADTFedAvgStrategy(FedAvg):
    """
    Custom FedAvg strategy with:
      - Convergence monitoring (MAE threshold)
      - Per-round checkpoint saving
      - Aggregated metric logging

    Parameters
    ----------
    convergence_mae_threshold : float
        Stop and mark convergence when global MAE drops below this value.
    checkpoint_dir : str
        Directory to save round checkpoints.
    **kwargs
        Passed to FedAvg.
    """

    def __init__(
        self,
        convergence_mae_threshold: float = 3.5,
        checkpoint_dir: str = "models/lstm/checkpoints",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.convergence_mae_threshold = convergence_mae_threshold
        self.checkpoint_dir = checkpoint_dir
        self.round_metrics: list[dict] = []
        self.convergence_round: Optional[int] = None
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate fit results and log DP metrics."""
        if not results:
            return None, {}

        # Call parent FedAvg aggregation
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # Collect per-client metrics
        train_losses = [r.metrics.get("train_loss", 0.0) for _, r in results]
        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        n_clipped = sum(int(r.metrics.get("was_clipped", 0)) for _, r in results)

        logger.info(
            f"Round {server_round:3d} | "
            f"Clients: {len(results)} | "
            f"Avg train loss: {avg_train_loss:.4f} | "
            f"Clipped: {n_clipped}/{len(results)}"
        )

        # Save round checkpoint
        if aggregated_parameters is not None:
            self._save_checkpoint(server_round, aggregated_parameters)

        # Log to MLflow
        try:
            mlflow.log_metric("avg_train_loss", avg_train_loss, step=server_round)
        except Exception as e:
            logger.debug(f"Failed to log avg_train_loss to MLflow: {e}")

        return aggregated_parameters, {"avg_train_loss": avg_train_loss}

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate evaluation results and check convergence."""
        if not results:
            return None, {}

        # Weighted MAE aggregation
        total_examples = sum(r.num_examples for _, r in results)
        weighted_mae = sum(
            r.num_examples * r.metrics.get("mae", 0.0)
            for _, r in results
        ) / max(total_examples, 1)

        self.round_metrics.append({
            "round": server_round,
            "global_mae": weighted_mae,
            "n_clients": len(results),
        })

        # Log to MLflow in real-time
        try:
            mlflow.log_metric("global_mae", weighted_mae, step=server_round)
        except Exception as e:
            logger.debug(f"Failed to log global_mae to MLflow in real-time: {e}")

        # Check convergence
        if (
            self.convergence_round is None
            and weighted_mae < self.convergence_mae_threshold
        ):
            self.convergence_round = server_round
            logger.info(
                f"★ Convergence achieved at round {server_round}! "
                f"MAE = {weighted_mae:.3f} < threshold {self.convergence_mae_threshold}"
            )

        logger.info(f"Round {server_round:3d} | Global MAE = {weighted_mae:.3f} persons")

        return weighted_mae, {"global_mae": weighted_mae, "n_clients": len(results)}

    def _save_checkpoint(self, round_num: int, parameters) -> None:
        """Save aggregated parameters as numpy arrays for round `round_num`."""
        import pickle
        ckpt_path = os.path.join(self.checkpoint_dir, f"global_round_{round_num:04d}.pkl")
        # Convert Flower Parameters to list of numpy arrays
        ndarrays = fl.common.parameters_to_ndarrays(parameters)
        with open(ckpt_path, "wb") as f:
            pickle.dump({"round": round_num, "parameters": ndarrays}, f)
        if round_num % 10 == 0:
            logger.info(f"Checkpoint saved: {ckpt_path}")

    def get_convergence_summary(self) -> dict:
        """Return a summary of training convergence."""
        if not self.round_metrics:
            return {}
        final_mae = self.round_metrics[-1]["global_mae"]
        best_mae = min(m["global_mae"] for m in self.round_metrics)
        return {
            "convergence_round": self.convergence_round,
            "final_mae": final_mae,
            "best_mae": best_mae,
            "total_rounds": len(self.round_metrics),
        }


def build_server_strategy(
    config: dict,
    checkpoint_dir: str = "models/lstm/checkpoints",
    n_clients: int = None
) -> EFADTFedAvgStrategy:
    """
    Build the EFADT FedAvg strategy from config.

    Parameters
    ----------
    config : dict
        Full hyperparameter configuration.
    checkpoint_dir : str
        Directory to save checkpoints.
    n_clients : int, optional
        Active client count to cap min_fit/min_evaluate/min_available.

    Returns
    -------
    EFADTFedAvgStrategy
    """
    fl_cfg = config["federated"]
    min_fit = fl_cfg["min_fit_clients"]
    min_eval = fl_cfg["min_evaluate_clients"]
    min_avail = fl_cfg["min_available_clients"]

    if n_clients is not None:
        min_fit = min(min_fit, n_clients)
        min_eval = min(min_eval, n_clients)
        min_avail = min(min_avail, n_clients)

    strategy = EFADTFedAvgStrategy(
        convergence_mae_threshold=fl_cfg["convergence_mae_threshold"],
        checkpoint_dir=checkpoint_dir,
        fraction_fit=fl_cfg["fraction_fit"],
        fraction_evaluate=fl_cfg["fraction_evaluate"],
        min_fit_clients=min_fit,
        min_evaluate_clients=min_eval,
        min_available_clients=min_avail,
    )
    return strategy


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Flower metric aggregation function for FedAvg evaluate.
    Computes weighted average of MAE across clients.
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_mae = sum(
        num_examples * m.get("mae", 0.0)
        for num_examples, m in metrics
    ) / max(total_examples, 1)
    return {"mae": weighted_mae}
