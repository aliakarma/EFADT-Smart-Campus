"""
EFADT — Agent Optimizer
=========================
Orchestrates the full agent decision procedure for one building node
per 30-second decision cycle (Algorithm 1, Steps 5–6).

Workflow:
  1. Build action space U_b
  2. Evaluate all candidates via digital twin simulator
  3. Apply hard constraints (comfort, crowd, power)
  4. Select u* = argmin utility over feasible set
  5. Return selected action + all scores for XAI
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from agent.action_space import BuildingAction, build_action_space, get_hvac_powers
from agent.utility_function import UtilityWeights, select_optimal_action, utility_summary
from digital_twin.simulator import DigitalTwinSimulator, ActionScore
from digital_twin.thermal_model import ThermalState

logger = logging.getLogger(__name__)


class EFADTAgent:
    """
    Utility-driven multi-objective agent for one building node.

    For each 30-second decision cycle:
      - Receives an occupancy forecast from the FL-LSTM
      - Evaluates all candidate actions through the digital twin
      - Returns the optimal HVAC setpoint u*

    Parameters
    ----------
    building_id : str
    simulator : DigitalTwinSimulator
        Pre-initialized digital twin for this building.
    weights : UtilityWeights
        Multi-objective balancing coefficients (λ_e, λ_c, λ_d).
    action_space : list[BuildingAction], optional
        Pre-built action set. If None, built from defaults.
    """

    def __init__(
        self,
        building_id: str,
        simulator: DigitalTwinSimulator,
        weights: Optional[UtilityWeights] = None,
        action_space: Optional[list[BuildingAction]] = None,
    ) -> None:
        self.building_id = building_id
        self.simulator = simulator
        self.weights = weights or UtilityWeights()
        self.action_space = action_space or build_action_space(
            hvac_min_kw=-simulator.params.P_cap,
            hvac_max_kw=simulator.params.P_cap,
        )
        self.hvac_powers = get_hvac_powers(self.action_space)
        self._decision_history: list[dict] = []

    def decide(
        self,
        current_state: ThermalState,
        occ_forecast: np.ndarray,
    ) -> tuple[BuildingAction, list[ActionScore], float]:
        """
        Main decision procedure. Algorithm 1, Steps 5–6.

        Parameters
        ----------
        current_state : ThermalState
            Live sensor readings (synced to digital twin).
        occ_forecast : np.ndarray, shape (H,)
            Occupancy forecast from FL-LSTM over H steps.

        Returns
        -------
        (best_action, all_scores, best_utility)
        """
        t0 = time.time()

        # Sync digital twin state
        self.simulator.sync_state(current_state)

        # Evaluate all candidate actions
        all_scores = self.simulator.evaluate_all_actions(self.hvac_powers, occ_forecast)

        # Select optimal feasible action
        best_score, best_utility = select_optimal_action(all_scores, self.weights)

        # Map ActionScore back to BuildingAction
        best_action = self.action_space[best_score.action_id]

        elapsed_ms = (time.time() - t0) * 1000
        n_feasible = sum(1 for s in all_scores if s.feasible)

        logger.debug(
            f"{self.building_id} | decide | "
            f"Feasible: {n_feasible}/{len(all_scores)} | "
            f"u*: {best_action} | utility={best_utility:.4f} | "
            f"{elapsed_ms:.1f}ms"
        )

        # Record decision for monitoring
        self._decision_history.append({
            "T_in": current_state.T_in,
            "T_out": current_state.T_out,
            "occ_forecast_max": float(np.max(occ_forecast)),
            "best_action": str(best_action),
            "best_utility": best_utility,
            "n_feasible": n_feasible,
            "elapsed_ms": elapsed_ms,
        })

        return best_action, all_scores, best_utility

    def get_decision_summary(self, n_recent: int = 10) -> list[dict]:
        """Return the most recent `n_recent` decision records."""
        return self._decision_history[-n_recent:]

    @classmethod
    def from_config(
        cls,
        building_id: str,
        simulator: DigitalTwinSimulator,
        config: dict,
    ) -> "EFADTAgent":
        """Build agent from hyperparameter config dict."""
        import yaml
        weights_cfg = config.get("agent", {})
        weights = UtilityWeights(
            lambda_e=weights_cfg.get("lambda_e", 0.5),
            lambda_c=weights_cfg.get("lambda_c", 0.35),
            lambda_d=weights_cfg.get("lambda_d", 0.15),
        )
        return cls(building_id=building_id, simulator=simulator, weights=weights)


if __name__ == "__main__":
    from digital_twin.thermal_model import BuildingThermalParams

    logging.basicConfig(level=logging.DEBUG)

    # Set up simulator
    params = BuildingThermalParams(alpha=0.0018, beta=0.011, gamma=0.009)
    sim = DigitalTwinSimulator("B01", params=params, H=6, o_max=80, P_cap=25.0)
    agent = EFADTAgent("B01", sim)

    # Simulate a hot day with high occupancy
    state = ThermalState(T_in=27.0, T_out=40.0, Q_hvac=0.0, occupancy=60)
    occ_forecast = np.array([60, 65, 70, 68, 65, 60])

    best_action, all_scores, utility = agent.decide(state, occ_forecast)
    print(f"\nSelected action: {best_action}")
    print(f"Utility: {utility:.4f}")
    print(f"\nTop 5 actions by utility:")
    for row in utility_summary(all_scores, agent.weights)[:5]:
        print(f"  {row}")
