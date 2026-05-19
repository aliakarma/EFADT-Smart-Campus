"""
EFADT — Main Dataset Generation Script
========================================
Generates the synthetic smart campus dataset:
  - 12 buildings × 12 months × 30s = ~12.6M records
  - 14 features per record
  - Three scenario splits: Normal, Peak, Failure
  - Saved as Parquet files for efficient loading

Usage:
    python -m data.generation.generate_dataset --config configs/hyperparams.yaml
    python -m data.generation.generate_dataset --n_buildings 2 --n_days 30  # quick test
"""

from __future__ import annotations

import argparse
import logging
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from data.generation.occupancy_model import (
    generate_occupancy_series,
    generate_co2_from_occupancy,
)
from data.generation.thermal_simulator import (
    simulate_building_thermal_series,
    generate_hvac_power_series,
)
from data.generation.sensor_fault_injector import inject_sensor_faults

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Feature Names ────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "occupancy",          # persons
    "co2_ppm",            # parts per million
    "temperature_in",     # °C
    "temperature_out",    # °C
    "humidity",           # % RH
    "hvac_power_kw",      # kW
    "hvac_setpoint",      # °C
    "motion_count",       # motion sensor triggers
    "hour_sin",           # cyclical time encoding
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]

TARGET_COLUMN = "occupancy_next"   # target for 1-step-ahead forecasting


def load_building_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def cyclical_encode(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Encode a periodic feature as (sin, cos) pair."""
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def generate_building_dataset(
    building_id: str,
    params: dict,
    timestamps: pd.DatetimeIndex,
    global_params: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate the full time-series for one building."""
    n = len(timestamps)
    bp = params  # building params

    base_rates = np.array([
        bp.get("max_occupancy", 80) * 0.05,   # low
        bp.get("max_occupancy", 80) * 0.40,   # medium
        bp.get("max_occupancy", 80) * 0.85,   # high
    ])

    # ── Occupancy & CO₂ ────────────────────────────────────────────────────
    occupancy = generate_occupancy_series(
        n, base_rates, timestamps,
        max_occupancy=bp.get("max_occupancy", 80),
        noise_std=2.0, rng=rng,
    )
    co2 = generate_co2_from_occupancy(occupancy, rng=rng)

    # ── HVAC ───────────────────────────────────────────────────────────────
    setpoint = 22.0 + rng.normal(0, 0.5)   # slightly different per building
    hvac_power = generate_hvac_power_series(
        n, occupancy, bp.get("initial_temp", 22.0), setpoint,
        bp.get("hvac_capacity_kw", 20.0), rng=rng,
    )

    # ── Thermal ────────────────────────────────────────────────────────────
    T_in, T_out = simulate_building_thermal_series(
        n, occupancy, hvac_power, timestamps,
        alpha=bp["alpha"], beta=bp["beta"], gamma=bp["gamma"],
        initial_temp=bp.get("initial_temp", 22.0),
        outdoor_offset=bp.get("outdoor_temp_offset", 0.0),
        rng=rng,
    )

    # ── Humidity (synthetic, correlated with occupancy) ────────────────────
    humidity = 40.0 + 0.3 * occupancy + rng.normal(0, 3.0, n)
    humidity = np.clip(humidity, 20.0, 90.0)

    # ── Motion sensors ─────────────────────────────────────────────────────
    motion = (occupancy * rng.uniform(0.5, 1.5, n)).astype(int)

    # ── Cyclical time features ─────────────────────────────────────────────
    hours = timestamps.hour.to_numpy().astype(float)
    dows = timestamps.dayofweek.to_numpy().astype(float)
    months = (timestamps.month.to_numpy() - 1).astype(float)

    hour_sin, hour_cos = cyclical_encode(hours, 24.0)
    dow_sin, dow_cos = cyclical_encode(dows, 7.0)
    month_sin, month_cos = cyclical_encode(months, 12.0)

    # ── Assemble DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame({
        "timestamp": timestamps,
        "building_id": building_id,
        "occupancy": occupancy,
        "co2_ppm": co2,
        "temperature_in": T_in,
        "temperature_out": T_out,
        "humidity": humidity,
        "hvac_power_kw": hvac_power,
        "hvac_setpoint": np.full(n, setpoint),
        "motion_count": motion,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_of_week_sin": dow_sin,
        "day_of_week_cos": dow_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
    })

    # Target: next-step occupancy (shifted by 1 for supervised learning)
    df[TARGET_COLUMN] = df["occupancy"].shift(-1)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df.set_index("timestamp", inplace=True)
    return df


def generate_full_dataset(
    n_buildings: int = 12,
    start_date: str = "2024-01-01",
    n_days: int = 365,
    config_path: str = "configs/hyperparams.yaml",
    building_config_path: str = "configs/building_params.yaml",
    output_dir: str = "data/raw",
    seed: int = 42,
) -> None:
    """Main entry point for dataset generation."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path("data/scenarios").mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        global_cfg = yaml.safe_load(f)
    with open(building_config_path) as f:
        building_cfg = yaml.safe_load(f)["buildings"]

    building_ids = list(building_cfg.keys())[:n_buildings]
    sampling_interval = global_cfg["data"]["sampling_interval_s"]
    freq = f"{sampling_interval}s"

    timestamps = pd.date_range(start_date, periods=n_days * 24 * 3600 // sampling_interval, freq=freq)
    logger.info(f"Generating {len(timestamps):,} timesteps for {n_buildings} buildings")

    all_dfs = []
    start = time.time()

    for i, bid in enumerate(tqdm(building_ids, desc="Buildings")):
        rng = np.random.default_rng(seed + i)
        params = building_cfg[bid]
        df = generate_building_dataset(bid, params, timestamps, global_cfg, rng)
        out_path = os.path.join(output_dir, f"{bid}.parquet")
        df.to_parquet(out_path, compression="snappy")
        all_dfs.append(df)
        logger.info(f"  {bid}: {len(df):,} records → {out_path}")

    # Save combined dataset
    combined = pd.concat(all_dfs)
    combined.to_parquet(os.path.join(output_dir, "campus_timeseries.parquet"), compression="snappy")
    elapsed = time.time() - start
    logger.info(f"Total records: {len(combined):,} | Elapsed: {elapsed:.1f}s")

    def _hash_parquet(path: str) -> str:
        """SHA-256 of a Parquet file for reproducibility verification."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    # Write dataset manifest
    dataset_manifest = {
        "seed": seed,
        "n_buildings": n_buildings,
        "n_days": n_days,
        "start_date": start_date,
        "files": {}
    }
    for bid in building_ids:
        out_path = os.path.join(output_dir, f"{bid}.parquet")
        dataset_manifest["files"][bid] = {
            "path": out_path,
            "sha256": _hash_parquet(out_path),
        }

    manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(dataset_manifest, f, indent=2)
    logger.info(f"Dataset manifest written to {manifest_path}")

    # Create scenario splits
    _create_scenario_splits(combined, global_cfg)


def _create_scenario_splits(df: pd.DataFrame, cfg: dict) -> None:
    """Split dataset into Normal, Peak, Failure scenarios."""
    out_dir = "data/scenarios"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Normal: all data
    df.to_parquet(os.path.join(out_dir, "normal.parquet"), compression="snappy")

    # Peak: exam months only
    exam_months = [4, 5, 11, 12]
    peak = df[df.index.month.isin(exam_months)]
    peak.to_parquet(os.path.join(out_dir, "peak.parquet"), compression="snappy")

    # Failure: inject 20% sensor faults
    rng = np.random.default_rng(999)
    failure = inject_sensor_faults(df.copy(), fault_rate=0.20, rng=rng)
    failure.to_parquet(os.path.join(out_dir, "failure.parquet"), compression="snappy")

    logger.info(f"Scenarios: Normal={len(df):,}, Peak={len(peak):,}, Failure={len(failure):,}")


def main():
    parser = argparse.ArgumentParser(description="Generate EFADT synthetic dataset")
    parser.add_argument("--config", default="configs/hyperparams.yaml")
    parser.add_argument("--building-config", default="configs/building_params.yaml")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--n-buildings", type=int, default=12)
    parser.add_argument("--n-days", type=int, default=365)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_full_dataset(
        n_buildings=args.n_buildings,
        start_date=args.start_date,
        n_days=args.n_days,
        config_path=args.config,
        building_config_path=args.building_config,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
