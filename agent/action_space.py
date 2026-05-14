"""
EFADT — Action Space Definition
=================================
Defines the discrete action set U_b for each building node.

Actions include:
  1. HVAC setpoint adjustments (continuous, discretized)
  2. Classroom reassignment signals
  3. Crowd redistribution alerts

The physical action space is discretized into a manageable set
of candidates to allow exhaustive evaluation through the digital twin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BuildingAction:
    """
    Represents a single candidate action for a building node.

    Parameters
    ----------
    action_id : int
        Index in the candidate action list.
    hvac_power_kw : float
        HVAC power level [kW]. Negative = cooling, positive = heating.
    hvac_setpoint_c : float
        Target indoor temperature setpoint [°C].
    classroom_reassignment : bool
        Whether to signal a classroom reassignment recommendation.
    crowd_alert : bool
        Whether to trigger a crowd redistribution alert.
    alert_level : str
        'none', 'advisory', 'mandatory'
    """
    action_id: int
    hvac_power_kw: float
    hvac_setpoint_c: float
    classroom_reassignment: bool = False
    crowd_alert: bool = False
    alert_level: str = "none"

    def __str__(self) -> str:
        parts = [f"Q={self.hvac_power_kw:+.1f}kW(T={self.hvac_setpoint_c:.1f}°C)"]
        if self.classroom_reassignment:
            parts.append("REASSIGN")
        if self.crowd_alert:
            parts.append(f"ALERT:{self.alert_level}")
        return " | ".join(parts)


def build_action_space(
    hvac_min_kw: float = -25.0,
    hvac_max_kw: float = 25.0,
    hvac_step_kw: float = 2.5,
    setpoint_min: float = 18.0,
    setpoint_max: float = 28.0,
    setpoint_step: float = 0.5,
    include_crowd_actions: bool = True,
) -> list[BuildingAction]:
    """
    Build the candidate action set U_b for a building node.

    Actions are combinations of:
      - HVAC power levels (discretized)
      - Crowd management signals (when occupancy is high)

    Parameters
    ----------
    hvac_min_kw, hvac_max_kw, hvac_step_kw : float
        Range and step for HVAC power [kW].
    setpoint_min, setpoint_max, setpoint_step : float
        Range and step for HVAC temperature setpoint [°C].
    include_crowd_actions : bool
        Whether to include crowd alert actions.

    Returns
    -------
    list[BuildingAction]
    """
    actions = []
    action_id = 0

    # Generate HVAC power candidates
    hvac_powers = np.arange(hvac_min_kw, hvac_max_kw + hvac_step_kw, hvac_step_kw)
    setpoints = np.arange(setpoint_min, setpoint_max + setpoint_step, setpoint_step)

    # Simple mapping: power → setpoint (linear interpolation)
    for Q in hvac_powers:
        # Estimate setpoint from power (rough heuristic for action representation)
        T_set = 23.0 + (Q / max(abs(hvac_max_kw), 1.0)) * (setpoint_max - setpoint_min) / 2
        T_set = float(np.clip(T_set, setpoint_min, setpoint_max))

        actions.append(BuildingAction(
            action_id=action_id,
            hvac_power_kw=float(Q),
            hvac_setpoint_c=T_set,
            classroom_reassignment=False,
            crowd_alert=False,
        ))
        action_id += 1

    if include_crowd_actions:
        # Add crowd alert variants for high-occupancy scenarios
        for Q in [0.0, -5.0, -10.0]:
            T_set = 22.0 if Q == 0.0 else 21.0
            actions.append(BuildingAction(
                action_id=action_id,
                hvac_power_kw=float(Q),
                hvac_setpoint_c=T_set,
                classroom_reassignment=True,
                crowd_alert=True,
                alert_level="advisory",
            ))
            action_id += 1

        # Mandatory alert (extreme crowd)
        actions.append(BuildingAction(
            action_id=action_id,
            hvac_power_kw=-15.0,
            hvac_setpoint_c=20.5,
            classroom_reassignment=True,
            crowd_alert=True,
            alert_level="mandatory",
        ))
        action_id += 1

    return actions


def get_hvac_powers(action_space: list[BuildingAction]) -> np.ndarray:
    """Extract HVAC power values from action space for vectorized simulation."""
    return np.array([a.hvac_power_kw for a in action_space])
