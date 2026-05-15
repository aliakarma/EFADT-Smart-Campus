"""
EFADT — Differential Privacy Mechanism
=========================================
Implements (ε, δ)-DP gradient perturbation per EFADT Section 5.3:

    Δθ̃_b = clip(Δθ_b, C) + N(0, σ²C²I)

Parameters:
    ε = 1.0   (privacy budget — strong guarantee)
    δ = 1e-5  (failure probability)
    C = 1.0   (ℓ₂ clipping norm)
    σ ≈ 1.47  (noise multiplier from Gaussian mechanism formula)
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_sigma(epsilon: float, delta: float) -> float:
    """
    Compute the Gaussian noise multiplier σ for (ε, δ)-DP
    using the analytical formula:

        σ = sqrt(2 * ln(1.25 / δ)) / ε

    Parameters
    ----------
    epsilon : float
        Privacy budget ε > 0.
    delta : float
        Failure probability δ > 0.

    Returns
    -------
    float : Noise multiplier σ.
    """
    sigma = math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    logger.debug(f"DP noise multiplier σ = {sigma:.4f} for (ε={epsilon}, δ={delta})")
    return sigma


def clip_gradient(
    gradient: np.ndarray,
    clip_norm: float = 1.0,
) -> tuple[np.ndarray, float]:
    """
    Clip gradient to maximum ℓ₂-norm (per-sample sensitivity bounding).

    Parameters
    ----------
    gradient : np.ndarray
        Flat gradient vector Δθ_b.
    clip_norm : float
        Maximum ℓ₂-norm C.

    Returns
    -------
    (clipped_gradient, actual_norm)
    """
    actual_norm = float(np.linalg.norm(gradient))
    if actual_norm > clip_norm:
        gradient = gradient * (clip_norm / actual_norm)
    return gradient, actual_norm


def add_gaussian_noise(
    gradient: np.ndarray,
    sigma: float,
    clip_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add calibrated Gaussian noise N(0, σ²C²I) to clipped gradient.

    Parameters
    ----------
    gradient : np.ndarray
        Clipped gradient (flat vector).
    sigma : float
        Noise multiplier.
    clip_norm : float
        Clipping norm C (sensitivity).
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray : Noised gradient Δθ̃_b.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise_std = sigma * clip_norm
    noise = rng.normal(0.0, noise_std, size=gradient.shape)
    return gradient + noise


def privatize_gradient(
    gradient: np.ndarray,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, dict]:
    """
    Full DP gradient privatization: clip then add Gaussian noise.

    Parameters
    ----------
    gradient : np.ndarray
        Raw gradient update Δθ_b (flattened).
    epsilon : float
        Privacy budget.
    delta : float
        Failure probability.
    clip_norm : float
        ℓ₂ clipping bound C.
    rng : np.random.Generator, optional

    Returns
    -------
    (privatized_gradient, info_dict)
        info_dict contains: sigma, original_norm, clipped, epsilon, delta
    """
    sigma = compute_sigma(epsilon, delta)
    clipped, original_norm = clip_gradient(gradient, clip_norm)
    privatized = add_gaussian_noise(clipped, sigma, clip_norm, rng)

    info = {
        "sigma": sigma,
        "original_norm": original_norm,
        "was_clipped": original_norm > clip_norm,
        "epsilon": epsilon,
        "delta": delta,
        "clip_norm": clip_norm,
    }
    return privatized, info


def privatize_model_update(
    local_params: list[np.ndarray],
    global_params: list[np.ndarray],
    epsilon: float = 1.0,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[np.ndarray], dict]:
    """
    Compute DP-noised model update Δθ̃_b from local and global parameters.

    Parameters
    ----------
    local_params : list[np.ndarray]
        Local model parameter arrays (after local training).
    global_params : list[np.ndarray]
        Global model parameter arrays (before local training).
    epsilon, delta, clip_norm, rng : see privatize_gradient.

    Returns
    -------
    (noised_update_list, aggregated_info)
        noised_update_list: list of noised delta arrays (same shapes as params)
    """
    # Compute raw updates
    updates = [lp - gp for lp, gp in zip(local_params, global_params)]

    # Flatten, privatize, unflatten
    flat = np.concatenate([u.flatten() for u in updates])
    noised_flat, info = privatize_gradient(flat, epsilon, delta, clip_norm, rng)

    # Reconstruct per-layer updates
    noised_updates = []
    offset = 0
    for u in updates:
        size = u.size
        noised_updates.append(noised_flat[offset : offset + size].reshape(u.shape))
        offset += size

    return noised_updates, info


def estimate_total_privacy_budget(
    epsilon_per_round: float,
    n_rounds: int,
    delta: float = 1e-5,
    composition: str = "basic",
) -> float:
    """
    Estimate total privacy budget after R rounds.

    IMPORTANT: Basic composition gives ε_total = R × ε_per_round, which is
    an overestimate. Use Rényi DP accounting (via Opacus) for tighter bounds.

    Parameters
    ----------
    epsilon_per_round : float
        Per-round epsilon.
    n_rounds : int
        Total FL rounds.
    delta : float
        Failure probability.
    composition : str
        'basic' (loose upper bound) or 'renyi' (tight, requires opacus).

    Returns
    -------
    float : Estimated total epsilon.
    """
    if composition == "basic":
        total_eps = epsilon_per_round * n_rounds
        logger.warning(
            f"Basic composition: ε_total = {total_eps:.1f} after {n_rounds} rounds. "
            "Use Rényi DP for tighter bound."
        )
        return total_eps
    elif composition == "renyi":
        try:
            # Opacus provides tight Rényi DP accounting
            from opacus.accountants import RDPAccountant
            accountant = RDPAccountant()
            sigma = compute_sigma(epsilon_per_round, delta)
            for _ in range(n_rounds):
                accountant.step(noise_multiplier=sigma, sample_rate=1.0)
            eps, _ = accountant.get_privacy_spent(delta=delta)
            logger.info(f"Rényi DP: ε_total = {eps:.3f} after {n_rounds} rounds")
            return eps
        except ImportError:
            logger.warning("Opacus not available; falling back to basic composition.")
            return epsilon_per_round * n_rounds
    else:
        raise ValueError(f"Unknown composition: {composition}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: compute sigma for EFADT's DP setting
    eps, dlt = 1.0, 1e-5
    sigma = compute_sigma(eps, dlt)
    print(f"DP parameters: ε={eps}, δ={dlt} → σ={sigma:.4f}")

    # Test gradient privatization
    rng = np.random.default_rng(42)
    fake_grad = rng.normal(0, 0.5, size=100_000)   # ~typical LSTM gradient size
    privatized, info = privatize_gradient(fake_grad, epsilon=eps, delta=dlt, rng=rng)
    print(f"Original norm: {info['original_norm']:.4f} | Clipped: {info['was_clipped']}")
    print(f"Privatized gradient norm: {np.linalg.norm(privatized):.4f}")

    # Total privacy budget
    total = estimate_total_privacy_budget(eps, n_rounds=100, composition="basic")
    print(f"Total ε (basic, 100 rounds): {total:.1f}")
