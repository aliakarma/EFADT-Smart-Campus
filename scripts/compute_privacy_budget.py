#!/usr/bin/env python
"""
scripts/compute_privacy_budget.py
===================================
Audits the total differential privacy cost for a given FL configuration.
Must be run after training to produce results/dp_audit.json.

Usage:
    python scripts/compute_privacy_budget.py \
        --config configs/hyperparams.yaml \
        --n-rounds 100
"""
import argparse
import json
import os
import sys
import yaml

# Ensure project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.dp_mechanism import compute_sigma, estimate_total_privacy_budget

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hyperparams.yaml")
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--output", default="results/dp_audit.json")
    args = parser.parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)["dp"]

    epsilon = config["epsilon"]
    delta = config["delta"]
    sigma = compute_sigma(epsilon, delta)

    # Basic composition (upper bound)
    eps_basic = estimate_total_privacy_budget(epsilon, args.n_rounds, delta, "basic")

    # Rényi DP (tight, requires opacus)
    eps_renyi = estimate_total_privacy_budget(epsilon, args.n_rounds, delta, "renyi")

    report = {
        "per_round": {
            "epsilon": epsilon,
            "delta": delta,
            "sigma": round(sigma, 4),
        },
        "total_after_rounds": {
            "n_rounds": args.n_rounds,
            "epsilon_basic_composition": round(eps_basic, 3),
            "epsilon_renyi_dp": round(eps_renyi, 3),
            "delta": delta,
        },
        "note": (
            "The system satisfies ({eps_renyi:.2f}, {delta})-DP total after {n_rounds} rounds "
            "under Rényi DP composition. The per-round budget is ε={epsilon}."
        ).format(eps_renyi=eps_renyi, delta=delta, n_rounds=args.n_rounds, epsilon=epsilon),
    }

    print(json.dumps(report, indent=2))
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
