"""
EFADT — Multi-Objective Utility Function
==========================================
Implements the agent's decision criterion (EFADT Eq. Section 5.5):

    u* = argmin_{u ∈ U_b^feas} [λ_e·E(u) - λ_c·C(u) + λ_d·D(u)]

where:
    E(u) ∈ [0,1]  normalized energy cost (minimize)
    C(u) ∈ [0,1]  comfort compliance (maximize → negative sign)
    D(u) ∈ [0,1]  crowd density risk (minimize)

    λ_e = 0.5, λ_c = 0.35, λ_d = 0.15  (from grid search)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from digital_twin.simulator import ActionScore


@dataclass
class UtilityWeights:
    """Multi-objective balancing coefficients."""
    lambda_e: float = 0.5    # Energy weight
    lambda_c: float = 0.35   # Comfort weight
    lambda_d: float = 0.15   # Crowd density weight

    def validate(self) -> None:
        assert self.lambda_e >= 0 and self.lambda_c >= 0 and self.lambda_d >= 0, \
            "All weights must be non-negative"
        total = self.lambda_e + self.lambda_c + self.lambda_d
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"

    @classmethod
    def from_config(cls, cfg: dict) -> "UtilityWeights":
        w = cfg.get("agent_weights", cfg.get("agent", {}))
        return cls(
            lambda_e=w.get("lambda_e", 0.5),
            lambda_c=w.get("lambda_c", 0.35),
            lambda_d=w.get("lambda_d", 0.15),
        )

    @classmethod
    def energy_only(cls) -> "UtilityWeights":
        """Ablation variant: single-objective energy minimization."""
        return cls(lambda_e=1.0, lambda_c=0.0, lambda_d=0.0)


def compute_utility(
    score: ActionScore,
    weights: UtilityWeights,
) -> float:
    """
    Compute scalar utility for a single action score.

    Lower utility = better action (minimization problem).

    Parameters
    ----------
    score : ActionScore
        E, C, D scores from digital twin simulation.
    weights : UtilityWeights

    Returns
    -------
    float : Utility score (lower is better).
    """
    return (
        weights.lambda_e * score.E
        - weights.lambda_c * score.C
        + weights.lambda_d * score.D
    )


def select_optimal_action(
    action_scores: list[ActionScore],
    weights: UtilityWeights,
    fallback_action_id: int = 0,
) -> tuple[Optional[ActionScore], float]:
    """
    Select the optimal action from all feasible candidates.

    u* = argmin_{u ∈ U_b^feas} utility(u)

    Parameters
    ----------
    action_scores : list[ActionScore]
        All evaluated candidate actions.
    weights : UtilityWeights
        Multi-objective weights.
    fallback_action_id : int
        Index to use if no feasible action found (emergency fallback).

    Returns
    -------
    (best_score, best_utility)
        best_score : ActionScore of selected action (or None if no feasibles).
        best_utility : float.
    """
    feasible = [s for s in action_scores if s.feasible]

    if not feasible:
        # Emergency fallback: return do-nothing (Q≈0) action
        # even if constraints are soft-violated
        fallback = action_scores[min(fallback_action_id, len(action_scores) - 1)]
        return fallback, float("inf")

    utilities = [compute_utility(s, weights) for s in feasible]
    best_idx = int(np.argmin(utilities))
    return feasible[best_idx], utilities[best_idx]


def utility_summary(
    action_scores: list[ActionScore],
    weights: UtilityWeights,
) -> list[dict]:
    """
    Compute utility for all actions and return sorted summary.

    Useful for debugging and logging decision rationale.
    """
    results = []
    for score in action_scores:
        u = compute_utility(score, weights) if score.feasible else float("inf")
        results.append({
            "action_id": score.action_id,
            "hvac_kw": score.hvac_power_kw,
            "E": round(score.E, 4),
            "C": round(score.C, 4),
            "D": round(score.D, 4),
            "utility": round(u, 4),
            "feasible": score.feasible,
            "violation": score.violation_reason,
        })
    return sorted(results, key=lambda x: x["utility"])
