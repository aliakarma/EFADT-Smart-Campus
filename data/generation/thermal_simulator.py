"""
EFADT — RC Thermal Simulator for Data Generation
==================================================
Implements the first-order RC (Resistance-Capacitance) thermal model:

    dT_in/dt = alpha*(T_out - T_in) + beta*Q_HVAC + gamma*kappa*occ

Used both for:
  1. Synthetic dataset generation (forward simulation)
  2. Digital twin what-if simulation during inference
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# Saudi Arabian outdoor temperature profile (monthly mean °C, based on Medina climate)
SAUDI_MONTHLY_TEMPS = {
    1: 14.0,  # January
    2: 17.0,  # February
    3: 22.0,  # March
    4: 28.0,  # April
    5: 34.0,  # May
    6: 38.0,  # June
    7: 40.0,  # July
    8: 39.0,  # August
    9: 35.0,  # September
    10: 28.0, # October
    11: 21.0, # November
    12: 15.0, # December
}

DIURNAL_AMPLITUDE = 7.0   # °C peak-to-trough daily swing


def outdoor_temperature(month: int, hour: int, noise_std: float = 1.0,
                         rng: Optional[np.random.Generator] = None) -> float:
    """
    Estimate outdoor temperature from monthly mean + diurnal cycle + noise.

    Parameters
    ----------
    month : int (1–12)
    hour : int (0–23)
    noise_std : float
    rng : np.random.Generator, optional

    Returns
    -------
    float : Temperature in °C
    """
    if rng is None:
        rng = np.random.default_rng(42)
    base = SAUDI_MONTHLY_TEMPS[month]
    # Diurnal: peak at 14:00, trough at 04:00 → cosine shifted by 14h
    diurnal = DIURNAL_AMPLITUDE * np.cos(2 * np.pi * (hour - 14) / 24)
    return base + diurnal + rng.normal(0, noise_std)


def euler_step(
    T_in: float,
    T_out: float,
    Q_hvac: float,
    occ: float,
    alpha: float,
    beta: float,
    gamma: float,
    kappa: float = 0.1,
    dt: float = 30.0,
) -> float:
    """
    Single Euler integration step of the RC thermal model.

    Parameters
    ----------
    T_in : float
        Current indoor temperature (°C).
    T_out : float
        Current outdoor temperature (°C).
    Q_hvac : float
        HVAC power (kW, positive = heating, negative = cooling).
    occ : float
        Occupancy count (persons).
    alpha : float
        Thermal conductance coefficient [1/s].
    beta : float
        HVAC efficiency coefficient [°C/kW/s].
    gamma : float
        Occupancy heat coupling coefficient.
    kappa : float
        Metabolic heat per person [kW/person].
    dt : float
        Timestep in seconds.

    Returns
    -------
    float : Updated indoor temperature (°C).
    """
    dT_dt = (
        alpha * (T_out - T_in)
        + beta * Q_hvac
        + gamma * kappa * occ
    )
    return T_in + dt * dT_dt


def simulate_building_thermal_series(
    n_steps: int,
    occupancy: np.ndarray,
    hvac_power: np.ndarray,
    timestamps,
    alpha: float,
    beta: float,
    gamma: float,
    kappa: float = 0.1,
    initial_temp: float = 22.0,
    dt: float = 30.0,
    noise_std: float = 0.1,
    outdoor_offset: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the full indoor temperature and outdoor temperature series.

    Returns
    -------
    (T_in_series, T_out_series) : both shape (n_steps,)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    months = timestamps.month.to_numpy()
    hours = timestamps.hour.to_numpy()
    base_temps = np.array([SAUDI_MONTHLY_TEMPS[m] for m in months])
    diurnal = DIURNAL_AMPLITUDE * np.cos(2 * np.pi * (hours - 14) / 24)
    noise = rng.normal(0, noise_std, n_steps)
    T_out_series = base_temps + diurnal + noise + outdoor_offset

    T_in_series = np.zeros(n_steps)
    T_in = initial_temp
    
    measurement_noise = rng.normal(0, 0.05, n_steps)
    for i in range(n_steps):
        T_out = T_out_series[i]
        T_in = euler_step(
            T_in, T_out, hvac_power[i], occupancy[i],
            alpha, beta, gamma, kappa, dt
        )
        T_in_series[i] = T_in + measurement_noise[i]

    return T_in_series, T_out_series


def generate_hvac_power_series(
    n_steps: int,
    occupancy: np.ndarray,
    T_in: float,
    setpoint: float,
    capacity_kw: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a simple rule-based HVAC power series based on temperature setpoint.
    Uses proportional control: Q = Kp * (setpoint - T_in).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    Kp = 2.0  # proportional gain
    hvac_power = np.zeros(n_steps)
    T_current = T_in

    for i in range(n_steps):
        # Proportional control towards setpoint
        error = setpoint - T_current
        Q = np.clip(Kp * error, -capacity_kw, capacity_kw)
        # Stochastic on/off cycling (HVAC compressor noise)
        if abs(error) < 0.5:
            Q *= rng.uniform(0.3, 0.7)
        hvac_power[i] = Q
        # Simple forward estimate for next T_current
        T_current += 0.001 * Q  # rough

    return hvac_power


if __name__ == "__main__":
    import pandas as pd

    n_steps = 2880
    timestamps = pd.date_range("2024-06-15 00:00", periods=n_steps, freq="30s")
    occ = np.random.randint(0, 50, size=n_steps)
    hvac = np.full(n_steps, 5.0)  # constant 5 kW

    T_in, T_out = simulate_building_thermal_series(
        n_steps, occ, hvac, timestamps,
        alpha=0.0018, beta=0.011, gamma=0.009,
        initial_temp=23.0
    )
    print(f"T_in — min: {T_in.min():.1f}°C, max: {T_in.max():.1f}°C, mean: {T_in.mean():.1f}°C")
    print(f"T_out — min: {T_out.min():.1f}°C, max: {T_out.max():.1f}°C, mean: {T_out.mean():.1f}°C")
