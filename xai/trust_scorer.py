"""
EFADT — Trust Score Computation
=================================
Implements EFADT Eq. (5) — Quantitative Trust Score:

    τ(u*) = η₁ · (Σ_{φⱼ>0} φⱼ / Σ|φⱼ|)  ← SHAP coherence
           + η₂ · C(u*)                    ← comfort alignment
           + η₃ · (1 - D(u*))              ← safety alignment

where η₁ = 0.5, η₂ = 0.3, η₃ = 0.2 (sum to 1.0)

τ ∈ [0, 1], higher = more trustworthy decision.
EFADT achieves τ = 0.887 on average across 12 months.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TrustWeights:
    """Trust score weighting coefficients."""
    eta1: float = 0.5    # SHAP coherence weight
    eta2: float = 0.3    # Comfort alignment weight
    eta3: float = 0.2    # Crowd safety alignment weight

    def validate(self) -> None:
        total = self.eta1 + self.eta2 + self.eta3
        assert abs(total - 1.0) < 1e-6, f"eta weights must sum to 1.0, got {total}"

    @classmethod
    def from_config(cls, cfg: dict) -> "TrustWeights":
        xai_cfg = cfg.get("xai", cfg.get("trust_weights", {}))
        return cls(
            eta1=xai_cfg.get("eta1", 0.5),
            eta2=xai_cfg.get("eta2", 0.3),
            eta3=xai_cfg.get("eta3", 0.2),
        )


@dataclass
class TrustScoreResult:
    """Full trust score with component breakdown."""
    tau: float                 # Composite trust score ∈ [0, 1]
    shap_coherence: float      # η₁ term
    comfort_alignment: float   # η₂ term
    safety_alignment: float    # η₃ term
    below_threshold: bool      # Flag if τ < alert threshold

    def __str__(self) -> str:
        return (
            f"τ={self.tau:.3f} ["
            f"SHAP={self.shap_coherence:.3f}, "
            f"Comfort={self.comfort_alignment:.3f}, "
            f"Safety={self.safety_alignment:.3f}]"
            f"{' ⚠ LOW TRUST' if self.below_threshold else ''}"
        )


def compute_shap_coherence(shap_values: np.ndarray) -> float:
    """
    Compute the SHAP coherence term:
        Σ_{φⱼ>0} φⱼ / Σ|φⱼ|

    This measures the fraction of explanatory mass that is positively
    directional (features pushing the forecast in the "expected" direction).

    Parameters
    ----------
    shap_values : np.ndarray, shape (n_features,)
        SHAP φⱼ values.

    Returns
    -------
    float ∈ [0, 1] : Higher = more coherent SHAP explanation.
    """
    positive_mass = float(np.sum(shap_values[shap_values > 0]))
    total_mass = float(np.sum(np.abs(shap_values)))
    if total_mass < 1e-10:
        return 0.0
    return positive_mass / total_mass


def compute_trust_score(
    shap_values: np.ndarray,
    C_u_star: float,
    D_u_star: float,
    weights: Optional[TrustWeights] = None,
    alert_threshold: float = 0.7,
) -> TrustScoreResult:
    """
    Compute the quantitative trust score τ(u*) per EFADT Eq. (5).

    Parameters
    ----------
    shap_values : np.ndarray, shape (n_features,)
        SHAP values for the current decision instance.
    C_u_star : float
        Comfort compliance score of the selected action ∈ [0, 1].
    D_u_star : float
        Crowd density risk of the selected action ∈ [0, 1].
    weights : TrustWeights, optional
        η₁, η₂, η₃ coefficients. Defaults to EFADT paper values.
    alert_threshold : float
        τ below this triggers a human review flag.

    Returns
    -------
    TrustScoreResult
    """
    if weights is None:
        weights = TrustWeights()

    shap_coherence = compute_shap_coherence(shap_values)
    comfort_alignment = float(np.clip(C_u_star, 0.0, 1.0))
    safety_alignment = float(np.clip(1.0 - D_u_star, 0.0, 1.0))

    tau = (
        weights.eta1 * shap_coherence
        + weights.eta2 * comfort_alignment
        + weights.eta3 * safety_alignment
    )
    tau = float(np.clip(tau, 0.0, 1.0))

    return TrustScoreResult(
        tau=tau,
        shap_coherence=shap_coherence * weights.eta1,
        comfort_alignment=comfort_alignment * weights.eta2,
        safety_alignment=safety_alignment * weights.eta3,
        below_threshold=tau < alert_threshold,
    )


def aggregate_trust_scores(trust_scores: list[float]) -> dict:
    """
    Compute summary statistics over a batch of trust scores.
    Used for monitoring and SDG reporting.
    """
    arr = np.array(trust_scores)
    return {
        "mean_tau": float(np.mean(arr)),
        "std_tau": float(np.std(arr)),
        "min_tau": float(np.min(arr)),
        "max_tau": float(np.max(arr)),
        "fraction_high_trust": float(np.mean(arr >= 0.85)),
        "fraction_low_trust": float(np.mean(arr < 0.7)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Simulate EFADT typical case
    shap_vals = rng.normal(0, 0.3, 14)
    C = 0.92   # High comfort compliance
    D = 0.40   # Moderate crowd density

    result = compute_trust_score(shap_vals, C, D)
    print(f"Trust score: {result}")

    # Simulate batch statistics
    scores = [
        compute_trust_score(rng.normal(0, 0.3, 14), rng.uniform(0.8, 1.0), rng.uniform(0.2, 0.5)).tau
        for _ in range(1000)
    ]
    stats = aggregate_trust_scores(scores)
    print(f"\nBatch statistics (n=1000): {stats}")
