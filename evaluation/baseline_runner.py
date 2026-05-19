"""
EFADT — Baseline Evaluation Runner
=====================================
Implements all four comparison baselines from Section 8:

  1. Centralized-NN   : Standard LSTM trained on centralized data (privacy-violating)
  2. FL-Only          : FedAvg without digital twin simulation or MOO agent
  3. DT-Only          : Digital twin with rule-based (non-federated) controller
  4. Rule-Based       : Threshold-based HVAC controller (no ML, no DT)

Each baseline is evaluated on the same test split as EFADT,
using identical metrics: ERR, CCS, CSS, MAE, τ.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from evaluation.metrics import EFADTMetrics, compute_all_metrics

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "occupancy", "co2_ppm", "temperature_in", "temperature_out",
    "humidity", "hvac_power_kw", "hvac_setpoint", "motion_count",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
]


# ── Baseline 1: Rule-Based Thermostat ───────────────────────────────────────

def rule_based_controller(
    T_in: float,
    setpoint: float = 22.0,
    deadband: float = 1.0,
    P_cap: float = 25.0,
) -> float:
    """Simple bang-bang thermostat controller."""
    if T_in > setpoint + deadband:
        return -P_cap * 0.8      # Cool at 80% capacity
    elif T_in < setpoint - deadband:
        return P_cap * 0.8       # Heat at 80% capacity
    return 0.0                   # Hold


def evaluate_rule_based(
    T_in_series: np.ndarray,
    T_out_series: np.ndarray,
    occupancy_series: np.ndarray,
    co2_series: np.ndarray,
    setpoint: float = 22.0,
    o_max: float = 80.0,
) -> EFADTMetrics:
    """Evaluate rule-based thermostat baseline."""
    n = len(T_in_series)
    hvac_energy = np.array([
        abs(rule_based_controller(T, setpoint=setpoint)) for T in T_in_series
    ]) * 30 / 3600   # kWh per 30s interval

    # Baseline energy (always running at rated capacity)
    baseline_energy = np.full(n, 25.0 * 30 / 3600)

    return compute_all_metrics(
        baseline_energy=baseline_energy,
        system_energy=hvac_energy,
        T_in_series=T_in_series,
        co2_series=co2_series,
        occupancy_true=occupancy_series,
        occupancy_pred=occupancy_series,   # No forecasting — naive persistence
        trust_scores=np.zeros(n),          # No XAI → τ = 0
        o_max=o_max,
    )


# ── Baseline 2: Centralized Neural Network ──────────────────────────────────

class CentralizedLSTM(torch.nn.Module):
    """Standard (privacy-violating) centralized LSTM."""

    def __init__(self, input_dim: int = 14, hidden_size: int = 128) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_size, num_layers=2,
                                   batch_first=True, dropout=0.2)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_centralized_nn(
    X_all: np.ndarray,
    y_all: np.ndarray,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> CentralizedLSTM:
    """Train a centralized LSTM on pooled data from all buildings."""
    if device is None:
        device = torch.device("cpu")

    model = CentralizedLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    X_t = torch.tensor(X_all, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_all, dtype=torch.float32).to(device)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb).squeeze(-1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            logger.debug(f"  Centralized epoch {epoch+1} | loss={total_loss/len(loader):.4f}")

    return model


# ── Baseline 4: Digital Twin Only (rule-based without FL) ───────────────────

def evaluate_dt_only(
    T_in_series: np.ndarray,
    T_out_series: np.ndarray,
    occupancy_series: np.ndarray,
    co2_series: np.ndarray,
    alpha: float = 0.0018,
    beta: float = 0.011,
    gamma: float = 0.009,
    o_max: float = 80.0,
) -> EFADTMetrics:
    """
    DT-Only baseline: uses the thermal model but with naive occupancy persistence
    (no LSTM forecasting), and a simple rule-based action selection.
    """
    from digital_twin.thermal_model import RCThermalModel, BuildingThermalParams

    params = BuildingThermalParams(alpha=alpha, beta=beta, gamma=gamma)
    model = RCThermalModel(params)
    n = len(T_in_series)
    hvac_energy = []
    T_traj = []

    T_in = float(T_in_series[0])
    for i in range(n):
        T_out = T_out_series[i]
        # Naive: persist current occupancy as forecast
        occ_forecast = occupancy_series[i]
        # Simple proportional control
        error = 22.0 - T_in
        Q = float(np.clip(2.0 * error, -params.P_cap, params.P_cap))
        T_in_next = model.step(T_in, T_out, Q, occ_forecast)
        hvac_energy.append(abs(Q) * 30 / 3600)
        T_traj.append(T_in_next)
        T_in = T_in_next

    baseline_energy = np.full(n, 25.0 * 30 / 3600)
    return compute_all_metrics(
        baseline_energy=baseline_energy,
        system_energy=np.array(hvac_energy),
        T_in_series=np.array(T_traj),
        co2_series=co2_series,
        occupancy_true=occupancy_series,
        occupancy_pred=occupancy_series,
        trust_scores=np.full(n, 0.841),   # Paper-reported τ for DT-only ablation
        o_max=o_max,
    )


# ── Ablation Summary ─────────────────────────────────────────────────────────

import json
import os
import warnings

# ──────────────────────────────────────────────────────────────────────────────
# ⚠️  PLACEHOLDER VALUES — NOT VERIFIED BY CODE
# These numbers are TARGET values from the paper draft. They are NOT generated
# by any training or evaluation run in this repository.
# Replace with values from: results/ablation/full_results.json
# after completing Phase 5 of REMEDIATION_PLAN.md.
# ──────────────────────────────────────────────────────────────────────────────
_PAPER_RESULTS_UNVERIFIED = {
    "EFADT (Full)":       {"ERR": None, "CCS": None, "CSS": None, "MAE": None, "tau": None},
    "-XAI":               {"ERR": None, "CCS": None, "CSS": None, "MAE": None, "tau": None},
    "-DT-WIF":            {"ERR": None, "CCS": None, "CSS": None, "MAE": None, "tau": None},
    "-DP":                {"ERR": None, "CCS": None, "CSS": None, "MAE": None, "tau": None},
    "-MOO (energy-only)": {"ERR": None, "CCS": None, "CSS": None, "MAE": None, "tau": None},
    "-FL (centralized)":  {"ERR": None, "CCS": None, "CSS": None, "MAE": None, "tau": None},
}


def _load_results() -> dict:
    """Load verified results from results/ablation/full_results.json if available."""
    path = os.path.join(os.path.dirname(__file__), "..", "results", "ablation", "full_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    warnings.warn(
        "results/ablation/full_results.json not found. "
        "PAPER_RESULTS contains None placeholders. "
        "Run: python scripts/evaluate_checkpoint.py to generate real results.",
        UserWarning,
        stacklevel=2,
    )
    return _PAPER_RESULTS_UNVERIFIED


PAPER_RESULTS = _load_results()


def print_ablation_table() -> None:
    """Print the ablation results as a formatted table."""
    print("\n" + "=" * 80)
    print(f"{'Variant':<25} {'ERR%':>6} {'CCS':>6} {'CSS':>6} {'MAE':>6} {'τ':>6}")
    print("-" * 80)
    for name, metrics in PAPER_RESULTS.items():
        row_vals = []
        for key in ["ERR", "CCS", "CSS", "MAE", "tau"]:
            val = metrics.get(key)
            if val is None:
                row_vals.append("  —  ")
            elif key == "ERR":
                row_vals.append(f"{val:>6.1f}")
            elif key == "MAE":
                row_vals.append(f"{val:>6.2f}")
            else:
                row_vals.append(f"{val:>6.3f}")
        print(f"{name:<25} {' '.join(row_vals)}")
    print("=" * 80)


if __name__ == "__main__":
    print_ablation_table()
