"""
EFADT — Sensor Fault Injector
==============================
Simulates the 'Failure' operational scenario by randomly dropping 20% of sensor
nodes and filling missing values with the last known good reading (hold-last-value).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def inject_sensor_faults(
    df: pd.DataFrame,
    fault_rate: float = 0.20,
    fault_duration_steps: int = 20,
    columns_to_affect: Optional[list[str]] = None,
    rng: Optional[np.random.Generator] = None,
    fill_strategy: str = "hold",
) -> pd.DataFrame:
    """
    Inject random sensor faults into a building time series DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full building sensor DataFrame (index = timestamps).
    fault_rate : float
        Fraction of timesteps per sensor to fault (0.0–1.0).
    fault_duration_steps : int
        Duration of each fault event in timesteps (contiguous NaN block).
    columns_to_affect : list[str], optional
        Sensor columns to affect. If None, uses all numeric columns.
    rng : np.random.Generator, optional
    fill_strategy : str
        How to fill faults: 'hold' (last known), 'zero', 'nan' (leave as NaN).

    Returns
    -------
    pd.DataFrame
        DataFrame with injected faults, with a boolean 'sensor_fault' column added.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    df_out = df.copy()
    n = len(df_out)

    if columns_to_affect is None:
        columns_to_affect = df_out.select_dtypes(include=[np.number]).columns.tolist()

    # Track fault mask
    fault_mask = np.zeros(n, dtype=bool)

    for col in columns_to_affect:
        # Determine fault start positions
        n_faults = max(1, int(fault_rate * n / fault_duration_steps))
        fault_starts = rng.choice(n - fault_duration_steps, size=n_faults, replace=False)

        for start in fault_starts:
            end = min(start + fault_duration_steps, n)
            df_out.loc[df_out.index[start:end], col] = np.nan
            fault_mask[start:end] = True

    df_out["sensor_fault"] = fault_mask

    # Fill faults according to strategy
    if fill_strategy == "hold":
        # Forward-fill (hold last value), then backward-fill for leading NaNs
        df_out[columns_to_affect] = (
            df_out[columns_to_affect].ffill().bfill()
        )
    elif fill_strategy == "zero":
        df_out[columns_to_affect] = df_out[columns_to_affect].fillna(0.0)
    elif fill_strategy == "nan":
        pass  # leave as-is

    return df_out


def create_failure_scenario(
    df: pd.DataFrame,
    n_buildings_to_affect: int = 2,
    total_buildings: int = 12,
    fault_rate: float = 0.20,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Create a 'Failure' scenario by selecting 20% of buildings to inject faults into.

    Parameters
    ----------
    df : pd.DataFrame
        Combined DataFrame with 'building_id' column.
    n_buildings_to_affect : int
        Number of buildings with faults (default: 2/12 ≈ 20%).
    total_buildings : int
        Total number of buildings.
    fault_rate : float
        Per-timestep fault probability for affected buildings.
    seed : int

    Returns
    -------
    dict mapping building_id -> faulty DataFrame
    """
    rng = np.random.default_rng(seed)
    building_ids = df["building_id"].unique()
    affected = rng.choice(building_ids, size=n_buildings_to_affect, replace=False)

    result = {}
    for bid in building_ids:
        sub = df[df["building_id"] == bid].copy()
        if bid in affected:
            sub = inject_sensor_faults(sub, fault_rate=fault_rate, rng=rng)
        else:
            sub["sensor_fault"] = False
        result[bid] = sub

    return result


if __name__ == "__main__":
    # Quick demonstration
    n = 2880
    timestamps = pd.date_range("2024-01-01", periods=n, freq="30s")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "occupancy": np.random.randint(0, 50, n),
        "temperature": np.random.normal(22, 2, n),
        "co2": np.random.normal(600, 100, n),
    }).set_index("timestamp")

    df_faulty = inject_sensor_faults(df, fault_rate=0.20, rng=np.random.default_rng(0))
    n_faults = df_faulty["sensor_fault"].sum()
    print(f"Faults injected: {n_faults}/{n} timesteps ({100*n_faults/n:.1f}%)")
    print(f"NaN remaining: {df_faulty.isna().sum().sum()}")
