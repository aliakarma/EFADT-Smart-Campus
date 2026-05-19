"""
EFADT — Flower Federated Learning Client
==========================================
Implements the Flower NumPyClient interface for each building node.

Each client:
  1. Receives global model parameters from the FL server
  2. Runs E=5 local training epochs on building-private data
  3. Applies DP gradient perturbation to the model update
  4. Returns noised update + local metrics to the server

Architecture note: Raw sensor data NEVER leaves the building node.
Only DP-noised gradient updates are transmitted.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.lstm.architecture import OccupancyLSTM, build_model
from models.lstm.train_local import prepare_data, train_local, evaluate_local
from federated.dp_mechanism import privatize_model_update

logger = logging.getLogger(__name__)

# Type aliases for Flower
Parameters = List[np.ndarray]
Metrics = Dict[str, float]
FitResult = Tuple[Parameters, int, Metrics]
EvalResult = Tuple[float, int, Metrics]


class EFADTClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for a single building node.

    Parameters
    ----------
    building_id : str
        Identifier (e.g., 'B01').
    df : pd.DataFrame
        Local building sensor data (stays on-device).
    config : dict
        System hyperparameter configuration.
    device : torch.device
    apply_dp : bool
        Whether to apply differential privacy to gradient updates.
    """

    def __init__(
        self,
        building_id: str,
        df,
        config: dict,
        device: Optional[torch.device] = None,
        apply_dp: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.building_id = building_id
        self.config = config
        self.apply_dp = apply_dp
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._base_seed = seed
        self._round_counter = 0

        lstm_cfg = config["lstm"]
        dp_cfg = config["dp"]
        self.local_epochs = lstm_cfg["local_epochs"]
        self.batch_size = lstm_cfg["batch_size"]
        self.lr = lstm_cfg["learning_rate"]
        self.lookback = lstm_cfg["lookback_steps"]
        self.epsilon = dp_cfg["epsilon"]
        self.delta = dp_cfg["delta"]
        self.clip_norm = dp_cfg["max_grad_norm"]

        # Prepare local data (stays on-device; only used for local training)
        data_cfg = config.get("data", {})
        train_ds, val_ds, self.scaler = prepare_data(
            df,
            lookback=self.lookback,
            train_months=data_cfg.get("train_months", [1,2,3,4,5,6]),
            val_months=data_cfg.get("val_months", [7,8,9]),
        )
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size * 2, shuffle=False)
        self.n_train_examples = len(train_ds)

        # Build local model (initially random; will be overwritten by server params)
        self.model = build_model(config, device=self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr,
            weight_decay=config["lstm"]["weight_decay"]
        )

        logger.info(
            f"Client {building_id} initialized | "
            f"train={len(train_ds)}, val={len(val_ds)} | "
            f"DP={'ON' if apply_dp else 'OFF'}"
        )

    # ── Flower Interface ────────────────────────────────────────────────────

    def get_parameters(self, config: dict) -> Parameters:
        """Return current local model parameters as a list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: Parameters) -> None:
        """Load global model parameters into local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: Parameters, config: dict) -> FitResult:
        """
        Flower fit round:
          1. Load global model
          2. Record pre-training params (for computing Δθ)
          3. Run E local epochs
          4. Apply DP to gradient update
          5. Return noised params + metrics
        """
        # Step 1: Load global model
        self.set_parameters(parameters)
        global_params = [p.copy() for p in parameters]

        # Step 2: Local training
        train_loss = train_local(
            self.model, self.train_loader, self.optimizer,
            self.device, n_epochs=self.local_epochs,
        )

        # Step 3: Compute local model params after training
        local_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        self._round_counter += 1
        # Step 4: Apply DP (optional — disabled for -DP ablation)
        if self.apply_dp:
            rng = np.random.default_rng(self._base_seed + self._round_counter * 1000)
            noised_updates, dp_info = privatize_model_update(
                local_params, global_params,
                epsilon=self.epsilon, delta=self.delta,
                clip_norm=self.clip_norm, rng=rng,
            )
            # Reconstruct noised parameters: θ^(r) + Δθ̃_b
            final_params = [gp + nu for gp, nu in zip(global_params, noised_updates)]
        else:
            final_params = local_params
            dp_info = {"sigma": 0.0, "was_clipped": False}

        metrics: Metrics = {
            "train_loss": float(train_loss),
            "dp_sigma": float(dp_info.get("sigma", 0.0)),
            "was_clipped": float(dp_info.get("was_clipped", False)),
        }

        logger.debug(
            f"  {self.building_id} | round fit | "
            f"loss={train_loss:.4f} | clipped={dp_info.get('was_clipped')}"
        )

        return final_params, self.n_train_examples, metrics

    def evaluate(self, parameters: Parameters, config: dict) -> EvalResult:
        """
        Flower evaluate round: Compute validation MAE on local data.
        """
        self.set_parameters(parameters)
        mae = evaluate_local(self.model, self.val_loader, self.device)

        metrics: Metrics = {"mae": float(mae)}
        logger.debug(f"  {self.building_id} | eval | MAE={mae:.3f}")

        return float(mae), self.n_train_examples, metrics


def create_client_fn(
    building_data: dict,
    config: dict,
    apply_dp: bool = True,
    device: Optional[torch.device] = None,
    seed: int = 42,
):
    """
    Factory for Flower simulation: returns a function that creates clients by ID.

    Parameters
    ----------
    building_data : dict
        Mapping from building_id (str) to pd.DataFrame.
    config : dict
        Hyperparameter config.
    apply_dp : bool
        Whether to apply DP.
    device : torch.device, optional
    seed : int

    Returns
    -------
    Callable[[str], EFADTClient]
    """
    def client_fn(cid: str) -> EFADTClient:
        building_id = list(building_data.keys())[int(cid)]
        df = building_data[building_id]
        return EFADTClient(
            building_id=building_id,
            df=df,
            config=config,
            device=device,
            apply_dp=apply_dp,
            seed=seed + int(cid),
        )

    return client_fn
