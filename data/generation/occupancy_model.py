"""
EFADT — Occupancy Generation Model
===================================
Implements a Poisson mixture model for synthetic campus occupancy generation.
Accounts for:
  - Weekly lecture timetables (weekday vs weekend)
  - Examination period surges
  - Holiday zero-occupancy
  - Gaussian stochastic residual variation
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson
from typing import Optional
import pandas as pd


# ── Academic Calendar Constants ─────────────────────────────────────────────

EXAM_MONTHS = [4, 5, 11, 12]          # April, May, November, December
HOLIDAY_MONTHS = [7, 8]               # July, August (summer break)
PEAK_HOURS_WEEKDAY = list(range(8, 19))  # 08:00–18:59


def is_exam_period(month: int) -> bool:
    return month in EXAM_MONTHS


def is_holiday(month: int, day_of_week: int) -> bool:
    """True if the timestep falls in a holiday or weekend."""
    if month in HOLIDAY_MONTHS:
        return True
    return day_of_week >= 5  # Saturday=5, Sunday=6


def get_schedule_weights(hour: int, day_of_week: int, month: int) -> np.ndarray:
    """
    Return mixing weights for [K=3] Poisson components:
      - Component 0: low occupancy (off-hours, weekends)
      - Component 1: medium occupancy (regular classes)
      - Component 2: high occupancy (peak/exam)
    """
    if is_holiday(month, day_of_week):
        return np.array([0.95, 0.04, 0.01])

    if hour in PEAK_HOURS_WEEKDAY:
        if is_exam_period(month):
            return np.array([0.05, 0.30, 0.65])
        else:
            return np.array([0.10, 0.65, 0.25])
    else:
        return np.array([0.70, 0.25, 0.05])


def generate_occupancy_series(
    n_steps: int,
    base_rates: np.ndarray,
    timestamps: pd.DatetimeIndex,
    max_occupancy: int = 80,
    noise_std: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a synthetic occupancy time series via Poisson mixture model.

    Parameters
    ----------
    n_steps : int
        Number of 30-second timesteps to generate.
    base_rates : np.ndarray, shape (3,)
        Mean Poisson rates for [low, medium, high] occupancy components.
    timestamps : pd.DatetimeIndex
        Corresponding timestamps for schedule-aware generation.
    max_occupancy : int
        Hard cap on occupancy (room capacity).
    noise_std : float
        Std-dev of additive Gaussian residual (sensor jitter simulation).
    rng : np.random.Generator, optional
        Random number generator (for reproducibility).

    Returns
    -------
    np.ndarray, shape (n_steps,)
        Integer occupancy counts in [0, max_occupancy].
    """
    if rng is None:
        rng = np.random.default_rng(42)

    occupancy = np.zeros(n_steps, dtype=float)

    for i in range(n_steps):
        ts = timestamps[i]
        weights = get_schedule_weights(ts.hour, ts.dayofweek, ts.month)
        component = rng.choice(len(base_rates), p=weights)
        rate = base_rates[component]
        occ = poisson.rvs(mu=max(rate, 0.1), random_state=rng.integers(0, 2**31))
        # Add Gaussian residual (sensor/behavioural noise)
        noise = rng.normal(0, noise_std)
        occupancy[i] = occ + noise

    # Clip to valid range and return as integers
    occupancy = np.clip(np.round(occupancy), 0, max_occupancy).astype(int)
    return occupancy


def generate_co2_from_occupancy(
    occupancy: np.ndarray,
    base_co2: float = 420.0,
    co2_per_person: float = 15.0,
    noise_std: float = 10.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Estimate CO₂ concentration (ppm) from occupancy count.
    Uses a simple linear model: CO2 = base_co2 + co2_per_person * occupancy + noise

    Parameters
    ----------
    occupancy : np.ndarray
        Occupancy counts.
    base_co2 : float
        Outdoor/background CO₂ level (ppm).
    co2_per_person : float
        CO₂ contribution per occupant per 30-second interval (ppm).
    noise_std : float
        Measurement noise standard deviation.
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray
        CO₂ concentration in ppm, clipped to [400, 2000].
    """
    if rng is None:
        rng = np.random.default_rng(42)
    co2 = base_co2 + co2_per_person * occupancy + rng.normal(0, noise_std, size=len(occupancy))
    return np.clip(co2, 400.0, 2000.0)


if __name__ == "__main__":
    # Quick sanity check
    import pandas as pd

    n_steps = 2880  # 1 day at 30s
    timestamps = pd.date_range("2024-01-15 00:00", periods=n_steps, freq="30s")
    base_rates = np.array([3.0, 25.0, 65.0])  # low / medium / high
    rng = np.random.default_rng(42)

    occ = generate_occupancy_series(n_steps, base_rates, timestamps, max_occupancy=80, rng=rng)
    co2 = generate_co2_from_occupancy(occ, rng=rng)

    print(f"Occupancy — min: {occ.min()}, max: {occ.max()}, mean: {occ.mean():.1f}")
    print(f"CO₂       — min: {co2.min():.0f}, max: {co2.max():.0f}, mean: {co2.mean():.0f} ppm")
