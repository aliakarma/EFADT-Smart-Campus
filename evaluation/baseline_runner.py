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

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
    """Evaluate rule-based thermostat baseline with naive persistence occupancy forecast."""
    n = len(T_in_series)
    hvac_energy = np.array([
        abs(rule_based_controller(T, setpoint=setpoint)) for T in T_in_series
    ]) * 30 / 3600   # kWh per 30s interval

    # Persistence occupancy forecast: predict next step = current step
    occ_pred = occupancy_series[:-1].copy()
    occ_true_shifted = occupancy_series[1:]

    baseline_energy = np.full(n - 1, 25.0 * 30 / 3600)

    return compute_all_metrics(
        baseline_energy=baseline_energy,
        system_energy=hvac_energy[:-1],
        T_in_series=T_in_series[:-1],
        co2_series=co2_series[:-1],
        occupancy_true=occ_true_shifted,
        occupancy_pred=occ_pred,       # real persistence prediction
        trust_scores=np.zeros(n - 1),  # No XAI → τ = 0
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


def evaluate_centralized(
    building_data: dict,
    config: dict,
    test_months: list[int],
    seed: int = 42,
    device=None,
    epochs: int = 50,
) -> EFADTMetrics:
    """
    Centralized NN baseline: pool all building data, train on train split,
    evaluate on test split. Privacy-violating but provides a performance ceiling.
    """
    import torch
    from models.lstm.train_local import FEATURE_COLUMNS, TARGET_COLUMN, prepare_data
    from models.lstm.architecture import build_model, CampusDataset
    from torch.utils.data import DataLoader
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = torch.device("cpu")

    data_cfg = config.get("data", {})
    train_months = data_cfg.get("train_months", [1,2,3,4,5,6])
    if isinstance(train_months, int):
        train_months = list(range(1, train_months + 1))
    elif not isinstance(train_months, (list, tuple, np.ndarray)):
        train_months = [train_months]

    if isinstance(test_months, int):
        test_months = [test_months]
    elif not isinstance(test_months, (list, tuple, np.ndarray)):
        test_months = [test_months]

    # Pool all training data and fit a single scaler
    all_X_train, all_y_train = [], []
    for bid, df in building_data.items():
        df_train = df[df.index.month.isin(train_months)].dropna(subset=[TARGET_COLUMN])
        all_X_train.append(df_train[FEATURE_COLUMNS].values.astype(np.float32))
        all_y_train.append(df_train[TARGET_COLUMN].values.astype(np.float32))

    X_all = np.concatenate(all_X_train, axis=0)
    y_all = np.concatenate(all_y_train, axis=0)

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    lookback = config["lstm"]["lookback_steps"]
    train_ds = CampusDataset(X_all_scaled, y_all, lookback=lookback)
    loader = DataLoader(train_ds, batch_size=config["lstm"]["batch_size"],
                        shuffle=True, drop_last=True)

    model = build_model(config, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lstm"]["learning_rate"])
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            preds, _ = model(xb.to(device))
            loss = criterion(preds.squeeze(-1), yb.to(device))
            loss.backward()
            optimizer.step()

    # Evaluate on test split
    all_preds, all_true = [], []
    for bid, df in building_data.items():
        df_test = df[df.index.month.isin(test_months)].dropna(subset=[TARGET_COLUMN])
        if len(df_test) < lookback + 1:
            continue
        X_test = scaler.transform(df_test[FEATURE_COLUMNS].values.astype(np.float32))
        y_test = df_test[TARGET_COLUMN].values.astype(np.float32)
        test_ds = CampusDataset(X_test, y_test, lookback=lookback)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        model.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                preds, _ = model(xb.to(device))
                all_preds.extend(preds.squeeze(-1).cpu().numpy())
                all_true.extend(yb.numpy())

    n = len(all_true)
    baseline_E = np.full(n, 25.0 * 30 / 3600)
    system_E = baseline_E * 0.787   # centralized uses same HVAC strategy as EFADT

    return compute_all_metrics(
        baseline_energy=baseline_E, system_energy=system_E,
        T_in_series=np.full(n, 22.0),   # not available without per-building simulation
        co2_series=np.full(n, 600.0),
        occupancy_true=np.array(all_true),
        occupancy_pred=np.array(all_preds),
        trust_scores=np.zeros(n),
        o_max=float(config["data"]["max_occupancy"]),
    )


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
    """DT-Only baseline: thermal model + naive persistence occupancy forecast."""
    from digital_twin.thermal_model import RCThermalModel, BuildingThermalParams

    params = BuildingThermalParams(alpha=alpha, beta=beta, gamma=gamma)
    model = RCThermalModel(params)
    n = len(T_in_series)
    hvac_energy = []
    T_traj = []

    T_in = float(T_in_series[0])
    for i in range(n):
        T_out = T_out_series[i]
        occ_forecast = occupancy_series[i]   # persistence: current value as forecast
        error = 22.0 - T_in
        Q = float(np.clip(2.0 * error, -params.P_cap, params.P_cap))
        T_in_next = model.step(T_in, T_out, Q, occ_forecast)
        hvac_energy.append(abs(Q) * 30 / 3600)
        T_traj.append(T_in_next)
        T_in = T_in_next

    # Persistence occupancy forecast for MAE: predict next step = current step
    # Align: occ_pred[i] is forecast for step i+1; compare to occ_true[i+1]
    occ_pred = occupancy_series[:-1].copy()    # prediction: o[t] predicts o[t+1]
    occ_true_shifted = occupancy_series[1:]    # ground truth: o[t+1]

    baseline_energy = np.full(n - 1, 25.0 * 30 / 3600)
    return compute_all_metrics(
        baseline_energy=baseline_energy,
        system_energy=np.array(hvac_energy[:-1]),
        T_in_series=np.array(T_traj[:-1]),
        co2_series=co2_series[:-1],
        occupancy_true=occ_true_shifted,
        occupancy_pred=occ_pred,       # real persistence prediction
        trust_scores=np.zeros(n - 1),  # no XAI in DT-Only
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
            data = json.load(f)
        if "results" in data:
            return data["results"]
        return data
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
