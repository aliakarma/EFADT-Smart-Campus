"""
EFADT — Digital Twin RC Thermal Model
=======================================
Implements the first-order RC thermal model for per-building
digital twin simulation used during agent decision-making:

    dT_in/dt = α(T_out - T_in) + β·Q_HVAC + γ·κ·ô

Euler discretized at Δt = 30s (one decision cycle).

Used for:
  - Pre-execution what-if simulation: evaluate all candidate HVAC actions
    before any physical actuation
  - Energy estimation: compute E(u) from simulated HVAC trajectory
  - Comfort scoring: compute C(u) from simulated temperature trajectory
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ThermalState:
    """Snapshot of digital twin state at a single timestep."""
    T_in: float            # Indoor temperature (°C)
    T_out: float           # Outdoor temperature (°C)
    Q_hvac: float          # HVAC power (kW)
    occupancy: float       # Occupancy count (persons)
    timestamp: Optional[str] = None


@dataclass
class BuildingThermalParams:
    """
    RC thermal parameters for one building.
    Fitted via OLS regression on historical data.
    """
    alpha: float = 0.0018     # Thermal conductance [1/s]
    beta: float = 0.011       # HVAC efficiency [°C/kW/s]
    gamma: float = 0.009      # Occupancy heat coupling
    kappa: float = 0.1        # Metabolic heat per person [kW/person]
    dt: float = 30.0          # Euler timestep [s]
    T_min: float = 20.0       # Comfort lower bound [°C]
    T_max: float = 26.0       # Comfort upper bound [°C]
    P_cap: float = 25.0       # HVAC capacity [kW]


class RCThermalModel:
    """
    Resistance-Capacitance (RC) first-order thermal model for one building.

    This is the core physics model inside the digital twin.
    One RCThermalModel instance per building node.

    Parameters
    ----------
    params : BuildingThermalParams
        Building-specific thermal coefficients.
    """

    def __init__(self, params: BuildingThermalParams) -> None:
        self.params = params

    def step(
        self,
        T_in: float,
        T_out: float,
        Q_hvac: float,
        occ: float,
    ) -> float:
        """
        Single Euler step of the RC thermal ODE.

        Parameters
        ----------
        T_in : float
            Current indoor temperature (°C).
        T_out : float
            Current outdoor temperature (°C).
        Q_hvac : float
            HVAC power (kW). Positive = heating, negative = cooling.
        occ : float
            Occupancy (persons).

        Returns
        -------
        float : Updated indoor temperature after dt seconds.
        """
        p = self.params
        dT_dt = (
            p.alpha * (T_out - T_in)
            + p.beta * Q_hvac
            + p.gamma * p.kappa * occ
        )
        return T_in + p.dt * dT_dt

    def simulate_horizon(
        self,
        T_in_0: float,
        T_out: float,
        Q_hvac: float,
        occ_forecast: np.ndarray,
        H: int = 6,
    ) -> np.ndarray:
        """
        Simulate temperature trajectory over H steps for a given HVAC action.

        Parameters
        ----------
        T_in_0 : float
            Initial indoor temperature (°C).
        T_out : float
            Assumed constant outdoor temperature (°C) over horizon.
        Q_hvac : float
            Candidate HVAC power (kW) to evaluate.
        occ_forecast : np.ndarray, shape (H,)
            Occupancy forecast over the H-step horizon.
        H : int
            Horizon length (default: 6 steps = 3 minutes).

        Returns
        -------
        np.ndarray, shape (H,) : Simulated indoor temperature trajectory.
        """
        traj = np.zeros(H)
        T = T_in_0
        for h in range(H):
            occ_h = occ_forecast[h] if h < len(occ_forecast) else occ_forecast[-1]
            T = self.step(T, T_out, Q_hvac, occ_h)
            traj[h] = T
        return traj

    def is_comfort_compliant(self, T_trajectory: np.ndarray) -> bool:
        """
        Check if a temperature trajectory stays within the comfort band
        [T_min, T_max] for all timesteps.
        """
        return bool(
            np.all(T_trajectory >= self.params.T_min) and
            np.all(T_trajectory <= self.params.T_max)
        )

    def compute_energy_cost(self, Q_hvac: float, H: int = 6) -> float:
        """
        Compute normalized energy cost E(u) ∈ [0, 1] for an HVAC action.

        E(u) = (|Q_hvac| * H * dt) / P_cap / H / dt
             = |Q_hvac| / P_cap   (normalized by capacity)

        Higher E → more energy consumed.
        """
        return min(abs(Q_hvac) / max(self.params.P_cap, 1e-6), 1.0)

    def compute_comfort_score(self, T_trajectory: np.ndarray) -> float:
        """
        Compute comfort compliance score C(u) ∈ [0, 1].

        C(u) = fraction of steps within comfort band [T_min, T_max].
        Higher C → better comfort.
        """
        T_min, T_max = self.params.T_min, self.params.T_max
        in_band = ((T_trajectory >= T_min) & (T_trajectory <= T_max)).sum()
        return float(in_band) / max(len(T_trajectory), 1)

    def fit_parameters(
        self,
        T_in_series: np.ndarray,
        T_out_series: np.ndarray,
        Q_hvac_series: np.ndarray,
        occ_series: np.ndarray,
    ) -> BuildingThermalParams:
        """
        Fit thermal parameters (α, β, γ) via OLS regression.

        Regresses:
            ΔT_in = α*(T_out - T_in)*dt + β*Q_hvac*dt + γ*κ*occ*dt

        Parameters
        ----------
        T_in_series, T_out_series, Q_hvac_series, occ_series : np.ndarray
            Historical time-series arrays of same length n.

        Returns
        -------
        BuildingThermalParams : Updated params with fitted α, β, γ.
        """
        from sklearn.linear_model import LinearRegression

        dt = self.params.dt
        kappa = self.params.kappa

        n = len(T_in_series) - 1

        # Compute target: ΔT_in per step
        delta_T = np.diff(T_in_series)

        # Design matrix: [α_component, β_component, γ_component]
        X_reg = np.stack([
            (T_out_series[:n] - T_in_series[:n]) * dt,     # α feature
            Q_hvac_series[:n] * dt,                         # β feature
            kappa * occ_series[:n] * dt,                    # γ feature
        ], axis=1)

        reg = LinearRegression(fit_intercept=False, positive=True)
        reg.fit(X_reg, delta_T)

        alpha_fit, beta_fit, gamma_fit = reg.coef_
        fitted_params = BuildingThermalParams(
            alpha=max(alpha_fit, 1e-6),
            beta=max(beta_fit, 1e-6),
            gamma=max(gamma_fit, 1e-6),
            kappa=self.params.kappa,
            dt=self.params.dt,
            T_min=self.params.T_min,
            T_max=self.params.T_max,
            P_cap=self.params.P_cap,
        )

        # Update self
        self.params = fitted_params
        return fitted_params


if __name__ == "__main__":
    # Quick sanity check
    params = BuildingThermalParams(alpha=0.0018, beta=0.011, gamma=0.009)
    model = RCThermalModel(params)

    # Simulate 6-step horizon for HVAC action Q=5kW
    occ_forecast = np.array([30, 32, 35, 38, 40, 38])
    T_traj = model.simulate_horizon(T_in_0=25.0, T_out=38.0, Q_hvac=-8.0, occ_forecast=occ_forecast)
    print(f"Temperature trajectory: {T_traj.round(2)}")
    print(f"Comfort compliant: {model.is_comfort_compliant(T_traj)}")
    print(f"Energy cost E(u): {model.compute_energy_cost(-8.0):.3f}")
    print(f"Comfort score C(u): {model.compute_comfort_score(T_traj):.3f}")
