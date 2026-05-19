"""
EFADT — Local LSTM Training (Single Building Node)
====================================================
Trains the OccupancyLSTM on a single building's data.
Used both for:
  1. Standalone local training (baseline comparison)
  2. Local epochs within each FL round (called by federated/client.py)

Includes:
  - StandardScaler feature normalization
  - MSE loss training with Adam optimizer
  - Validation MAE reporting per epoch
  - Checkpoint saving
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import yaml

from models.lstm.architecture import OccupancyLSTM, CampusDataset, build_model

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "occupancy", "co2_ppm", "temperature_in", "temperature_out",
    "humidity", "hvac_power_kw", "hvac_setpoint", "motion_count",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
]
TARGET_COLUMN = "occupancy_next"


def prepare_data(
    df: pd.DataFrame,
    lookback: int = 12,
    train_months: list[int] = None,
    val_months: list[int] = None,
    scaler: Optional[StandardScaler] = None,
) -> tuple:
    """
    Calendar-based train/val split. Scaler is fit ONLY on training months.

    Parameters
    ----------
    df : pd.DataFrame with DatetimeIndex
    lookback : int  — LSTM sequence length
    train_months : list[int]  — months (1–12) used for training
    val_months : list[int]    — months used for validation
    scaler : StandardScaler, optional  — pre-fitted scaler (for inference reuse)

    Returns
    -------
    (train_dataset, val_dataset, scaler)
    """
    import pandas as pd

    if train_months is None:
        train_months = [1, 2, 3, 4, 5, 6]
    if val_months is None:
        val_months = [7, 8, 9]

    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    # Calendar split — strictly temporal, no random sampling
    train_mask = df.index.month.isin(train_months)
    val_mask = df.index.month.isin(val_months)

    df_train = df[train_mask]
    df_val = df[val_mask]

    if len(df_train) == 0:
        raise ValueError(f"No training data for months {train_months}. Check dataset date range.")
    if len(df_val) == 0:
        raise ValueError(f"No validation data for months {val_months}. Check dataset date range.")

    X_train = df_train[FEATURE_COLUMNS].values.astype(np.float32)
    y_train = df_train[TARGET_COLUMN].values.astype(np.float32)
    X_val = df_val[FEATURE_COLUMNS].values.astype(np.float32)
    y_val = df_val[TARGET_COLUMN].values.astype(np.float32)

    # Fit scaler on training data ONLY
    if scaler is None:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
    else:
        X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = CampusDataset(X_train, y_train, lookback=lookback)
    val_ds = CampusDataset(X_val, y_val, lookback=lookback)

    logger.info(
        f"Split: train={len(df_train):,} ({train_months}), "
        f"val={len(df_val):,} ({val_months})"
    )
    return train_ds, val_ds, scaler


def train_local(
    model: OccupancyLSTM,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int = 5,
    grad_clip: float = 1.0,
) -> float:
    """
    Run local training for `n_epochs` epochs.

    Returns
    -------
    float : Mean training MSE loss over last epoch.
    """
    criterion = nn.MSELoss()
    model.train()
    last_loss = 0.0

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds, _ = model(X_batch)
            loss = criterion(preds.squeeze(-1), y_batch)
            loss.backward()

            # Gradient clipping (also done by DP mechanism in FL)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        last_loss = total_loss / max(n_batches, 1)
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.log_metric("train_loss", last_loss, step=epoch)
        except Exception:
            pass

    return last_loss


def evaluate_local(
    model: OccupancyLSTM,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute Mean Absolute Error on validation set.

    Returns
    -------
    float : MAE in persons.
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds, _ = model(X_batch)
            all_preds.append(preds.squeeze(-1).cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds_arr = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)
    mae = np.mean(np.abs(preds_arr - targets_arr))
    return float(mae)


def run_standalone_training(
    building_id: str,
    data_path: str = "data/raw",
    config_path: str = "configs/hyperparams.yaml",
    checkpoint_dir: str = "models/lstm/checkpoints",
    seed: int = 42,
    epochs: int = 50,
) -> dict:
    """
    Train OccupancyLSTM for a single building in standalone (non-FL) mode.
    Returns training summary dict.
    """
    import pandas as pd

    torch.manual_seed(seed)
    np.random.seed(seed)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_cfg = cfg["lstm"]
    data_cfg = cfg["data"]

    # Load building data
    parquet_path = os.path.join(data_path, f"{building_id}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"No data found at {parquet_path}. Run generate_dataset.py first.")

    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df):,} records for building {building_id}")

    # Prepare datasets
    train_ds, val_ds, scaler = prepare_data(
        df,
        lookback=lstm_cfg["lookback_steps"],
        train_months=data_cfg.get("train_months", [1,2,3,4,5,6]),
        val_months=data_cfg.get("val_months", [7,8,9]),
    )
    train_loader = DataLoader(train_ds, batch_size=lstm_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=lstm_cfg["batch_size"] * 2, shuffle=False)

    # Build model and optimizer
    model = build_model(cfg, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lstm_cfg["learning_rate"],
        weight_decay=lstm_cfg["weight_decay"],
    )

    logger.info(f"Training OccupancyLSTM on {device} | Params: {model.get_parameter_count():,}")

    # Full training loop (for standalone; FL uses local_epochs only)
    n_full_epochs = epochs
    best_mae = float("inf")
    history = []

    for epoch in range(n_full_epochs):
        train_loss = train_local(model, train_loader, optimizer, device, n_epochs=1)
        val_mae = evaluate_local(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_mae": val_mae})

        if val_mae < best_mae:
            best_mae = val_mae
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"{building_id}_best.pt")
            import pickle
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": val_mae,
                "building_id": building_id,
                "feature_columns": FEATURE_COLUMNS,
            }, ckpt_path)

            scaler_path = ckpt_path.replace("_best.pt", "_best_scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1:3d} | train_loss={train_loss:.4f} | val_mae={val_mae:.3f}")

    logger.info(f"Best MAE for {building_id}: {best_mae:.3f} persons")
    return {"building_id": building_id, "best_mae": best_mae, "history": history}


def load_checkpoint(
    checkpoint_path: str,
    config: dict,
    device: Optional[torch.device] = None,
):
    """Load model and scaler from a training checkpoint."""
    import pickle
    from sklearn.preprocessing import StandardScaler

    if device is None:
        device = torch.device("cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(config, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler_path = checkpoint_path.replace("_best.pt", "_best_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. "
            "Retrain the model with the updated train_local.py."
        )
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, ckpt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--building-id", default="B01")
    parser.add_argument("--data-path", default="data/raw")
    parser.add_argument("--config", default="configs/hyperparams.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = run_standalone_training(args.building_id, args.data_path, args.config, epochs=args.epochs)
    print(f"\nResult: {result}")
