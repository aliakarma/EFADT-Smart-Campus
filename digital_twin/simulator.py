"""
EFADT — Digital Twin H-Step What-If Simulator
===============================================
Wraps the RCThermalModel for pre-execution simulation of all candidate actions.

Core loop (per decision cycle):
  For each candidate action u ∈ U_b:
    1. Sync DT state from live sensors
    2. Simulate H=6 steps via RC thermal model with occ forecast
    3. Compute E(u), C(u), D(u) scores
    4. Check hard constraints
    5. Return feasible action scores to agent

Decision latency budget: ~27ms for simulation phase.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from digital_twin.thermal_model import RCThermalModel, BuildingThermalParams, ThermalState


@dataclass
class ActionScore:
    """Scores for a single candidate action."""
    action_id: int
    hvac_power_kw: float
    E: float                  # Normalized energy cost [0, 1]
    C: float                  # Comfort compliance score [0, 1]
    D: float                  # Crowd density risk [0, 1]
    T_trajectory: np.ndarray  # Simulated temperature trajectory
    feasible: bool            # Whether all hard constraints are satisfied
    violation_reason: str = ""


class DigitalTwinSimulator:
    """
    Digital twin simulator for one building node.

    Maintains current building state and provides what-if simulation
    for any candidate HVAC setpoint action before physical execution.

    Parameters
    ----------
    building_id : str
    params : BuildingThermalParams
        RC thermal model parameters.
    H : int
        Simulation horizon (number of 30-second steps, default: 6).
    o_max : int
        Maximum occupancy capacity (persons, default: 80).
    P_cap : float
        HVAC power capacity (kW, default: 25.0).
    """

    def __init__(
        self,
        building_id: str,
        params: Optional[BuildingThermalParams] = None,
        H: int = 6,
        o_max: int = 80,
        P_cap: float = 25.0,
    ) -> None:
        self.building_id = building_id
        self.H = H
        self.o_max = o_max
        self.P_cap = P_cap
        self.params = params or BuildingThermalParams()
        self.thermal_model = RCThermalModel(self.params)

        # Current state (synchronized from live sensors every 30s)
        self.current_state: Optional[ThermalState] = None

    def sync_state(self, state: ThermalState) -> None:
        """
        Update digital twin state from live sensor reading.
        Called at the start of every 30-second decision cycle.

        Parameters
        ----------
        state : ThermalState
            Current sensor readings.
        """
        self.current_state = state

    def evaluate_action(
        self,
        hvac_power_kw: float,
        occ_forecast: np.ndarray,
        action_id: int = 0,
    ) -> ActionScore:
        """
        Evaluate a single candidate HVAC action via simulation.

        Parameters
        ----------
        hvac_power_kw : float
            Candidate HVAC power [kW]. Positive = heating, negative = cooling.
        occ_forecast : np.ndarray, shape (H,)
            Occupancy forecast over the H-step horizon.
        action_id : int
            Candidate identifier for tracking.

        Returns
        -------
        ActionScore
            E, C, D scores + feasibility flag.
        """
        if self.current_state is None:
            raise RuntimeError("Digital twin state not synchronized. Call sync_state() first.")

        T_in_0 = self.current_state.T_in
        T_out = self.current_state.T_out

        # Simulate H-step temperature trajectory
        T_traj = self.thermal_model.simulate_horizon(
            T_in_0=T_in_0,
            T_out=T_out,
            Q_hvac=hvac_power_kw,
            occ_forecast=occ_forecast,
            H=self.H,
        )

        # Compute scores
        E = self.thermal_model.compute_energy_cost(hvac_power_kw, self.H)
        C = self.thermal_model.compute_comfort_score(T_traj)

        # Crowd density risk D(u) = max forecast occupancy / o_max
        max_forecast_occ = float(np.max(occ_forecast[:self.H]))
        D = min(max_forecast_occ / max(self.o_max, 1), 1.0)

        # Hard constraint checks
        feasible = True
        violation_reason = ""

        # 1. Comfort constraint: all simulated temps must be in [T_min, T_max]
        if not self.thermal_model.is_comfort_compliant(T_traj):
            feasible = False
            violation_reason = f"Comfort violation: T ∉ [{self.params.T_min}, {self.params.T_max}]°C"

        # 2. Crowd constraint: forecast occupancy must not exceed o_max
        elif max_forecast_occ > self.o_max:
            feasible = False
            violation_reason = f"Crowd violation: occ={max_forecast_occ:.0f} > o_max={self.o_max}"

        # 3. Power constraint: HVAC power must not exceed capacity
        elif abs(hvac_power_kw) > self.P_cap:
            feasible = False
            violation_reason = f"Power violation: |Q|={abs(hvac_power_kw):.1f} > P_cap={self.P_cap}"

        return ActionScore(
            action_id=action_id,
            hvac_power_kw=hvac_power_kw,
            E=E, C=C, D=D,
            T_trajectory=T_traj,
            feasible=feasible,
            violation_reason=violation_reason,
        )

    def evaluate_all_actions(
        self,
        candidate_powers: np.ndarray,
        occ_forecast: np.ndarray,
    ) -> list[ActionScore]:
        """
        Evaluate all candidate HVAC actions in the action space.

        Parameters
        ----------
        candidate_powers : np.ndarray
            Array of HVAC power values [kW] to evaluate.
        occ_forecast : np.ndarray, shape (H,)
            Occupancy forecast over the horizon.

        Returns
        -------
        list[ActionScore]
            Scores for each candidate action.
        """
        scores = []
        for i, q in enumerate(candidate_powers):
            score = self.evaluate_action(q, occ_forecast, action_id=i)
            scores.append(score)
        return scores

    def update_thermal_params(
        self,
        T_in_series: np.ndarray,
        T_out_series: np.ndarray,
        Q_hvac_series: np.ndarray,
        occ_series: np.ndarray,
    ) -> None:
        """
        Refit thermal parameters from historical data.
        Should be called periodically (e.g., weekly).
        """
        fitted = self.thermal_model.fit_parameters(
            T_in_series, T_out_series, Q_hvac_series, occ_series
        )
        self.params = fitted
        self.thermal_model = RCThermalModel(fitted)


if __name__ == "__main__":
    # Quick validation
    params = BuildingThermalParams(alpha=0.0018, beta=0.011, gamma=0.009)
    sim = DigitalTwinSimulator("B01", params=params)

    # Sync state
    state = ThermalState(T_in=25.0, T_out=38.0, Q_hvac=0.0, occupancy=30)
    sim.sync_state(state)

    # Evaluate 5 candidate actions
    candidates = np.array([-20.0, -10.0, -5.0, 0.0, 5.0])  # kW
    occ_forecast = np.array([30, 32, 35, 38, 35, 30])

    t0 = time.time()
    scores = sim.evaluate_all_actions(candidates, occ_forecast)
    elapsed_ms = (time.time() - t0) * 1000

    print(f"Evaluated {len(scores)} actions in {elapsed_ms:.1f}ms")
    for s in scores:
        status = "✓" if s.feasible else "✗"
        print(f"  {status} Q={s.hvac_power_kw:+.0f}kW | E={s.E:.3f} C={s.C:.3f} D={s.D:.3f} "
              f"| {s.violation_reason or 'OK'}")
