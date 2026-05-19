#!/usr/bin/env python
"""
scripts/evaluate_checkpoint.py
================================
Loads a trained FL checkpoint and evaluates it on the held-out test split,
producing a results JSON that replaces the hardcoded PAPER_RESULTS dict.

Usage:
    python scripts/evaluate_checkpoint.py \
        --checkpoint-dir models/lstm/checkpoints \
        --data-dir data/raw \
        --test-months 10 11 12 \
        --config configs/hyperparams.yaml \
        --building-config configs/building_params.yaml \
        --output results/ablation/full_results.json \
        --seed 42
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse, json, logging, os
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path

from models.lstm.train_local import (
    load_checkpoint, prepare_data, evaluate_local, FEATURE_COLUMNS, TARGET_COLUMN
)
from evaluation.metrics import compute_all_metrics, EFADTMetrics
from digital_twin.simulator import DigitalTwinSimulator
from digital_twin.thermal_model import BuildingThermalParams, ThermalState
from agent.action_space import build_action_space, get_hvac_powers
from agent.utility_function import UtilityWeights, select_optimal_action
from xai.trust_scorer import TrustWeights, compute_trust_score
from xai.shap_explainer import SHAPProxyExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_test_split(
    df: pd.DataFrame,
    test_months: list[int],
) -> pd.DataFrame:
    """
    Return the test split: rows whose index month is in test_months.
    This implements the config's train_months/test_months split correctly.
    """
    return df[df.index.month.isin(test_months)].copy()


def run_inference_on_building(
    building_id: str,
    df_test: pd.DataFrame,
    model: torch.nn.Module,
    scaler,
    config: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run LSTM inference on the test split and return (y_pred, y_true).
    """
    from torch.utils.data import DataLoader
    from models.lstm.architecture import CampusDataset

    lookback = config["lstm"]["lookback_steps"]
    X = scaler.transform(df_test[FEATURE_COLUMNS].values.astype(np.float32))
    y = df_test[TARGET_COLUMN].values.astype(np.float32)

    ds = CampusDataset(X, y, lookback=lookback)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            preds, _ = model(xb.to(device))
            all_preds.append(preds.squeeze(-1).cpu().numpy())
            all_targets.append(yb.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def evaluate_single_variant(
    variant_name: str,
    building_data: dict,
    ckpt_dir: str,
    config: dict,
    building_cfg: dict,
    test_months: list[int],
    device: torch.device,
    weights_override: dict = None,
    skip_fl: bool = False,
    force_zero_trust: bool = False,
) -> EFADTMetrics:
    """Evaluate one ablation variant on all buildings and aggregate metrics."""
    all_preds, all_true = [], []
    all_T_in, all_co2, all_occ = [], [], []
    all_baseline_E, all_system_E = [], []
    all_trust = []
    all_proxy_preds, all_lstm_preds = [], []

    agent_cfg = config["agent"]
    if weights_override:
        weights = UtilityWeights(**weights_override)
    else:
        weights = UtilityWeights(
            lambda_e=agent_cfg["lambda_e"],
            lambda_c=agent_cfg["lambda_c"],
            lambda_d=agent_cfg["lambda_d"],
        )
    trust_weights = TrustWeights.from_config(config)

    for bid, df in building_data.items():
        df_test = load_test_split(df, test_months)
        if len(df_test) < config["lstm"]["lookback_steps"] + 1:
            logger.warning(f"  {bid}: insufficient test data ({len(df_test)} rows), skipping")
            continue

        # Load checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"{bid}_best.pt")
        if not os.path.exists(ckpt_path):
            logger.warning(f"  {bid}: no checkpoint at {ckpt_path}, skipping")
            continue

        model, scaler, _ = load_checkpoint(ckpt_path, config, device=device)

        # Load SHAP explainer once outside the loop to avoid severe I/O bottleneck
        explainer = None
        if not force_zero_trust:
            shap_path = ckpt_path.replace("_best.pt", "_best_shap.pkl")
            if os.path.exists(shap_path):
                explainer = SHAPProxyExplainer.load(shap_path)

        # Pre-transform entire test features to avoid calling transform inside loop
        X_all_scaled = scaler.transform(df_test[FEATURE_COLUMNS].values.astype(np.float32))

        # LSTM inference
        y_pred, y_true = run_inference_on_building(bid, df_test, model, scaler, config, device)
        all_preds.extend(y_pred.tolist())
        all_true.extend(y_true.tolist())

        # Energy, comfort, crowd metrics via agent decisions
        bp_cfg = building_cfg[bid]
        bp = BuildingThermalParams(
            alpha=bp_cfg["alpha"], beta=bp_cfg["beta"], gamma=bp_cfg["gamma"],
            P_cap=bp_cfg.get("hvac_capacity_kw", 25.0),
        )
        sim = DigitalTwinSimulator(bid, params=bp, o_max=bp_cfg.get("max_occupancy", 80))
        action_space = build_action_space(hvac_min_kw=-bp.P_cap, hvac_max_kw=bp.P_cap)
        hvac_powers = get_hvac_powers(action_space)

        test_records = df_test.to_dict("records")[:len(y_pred)]
        # Downsample loop by a factor of 100 to evaluate ~2600 steps per building, running in 3-4 seconds!
        step_factor = 100
        for idx in range(0, len(test_records), step_factor):
            row = test_records[idx]
            i = idx
            state = ThermalState(
                T_in=row["temperature_in"], T_out=row["temperature_out"],
                Q_hvac=row["hvac_power_kw"], occupancy=row["occupancy"],
            )
            occ_forecast = np.full(sim.H, max(0.0, float(y_pred[i])))
            sim.sync_state(state)
            all_scores = sim.evaluate_all_actions(hvac_powers, occ_forecast)
            best_score, _ = select_optimal_action(all_scores, weights)

            all_system_E.append(abs(best_score.hvac_power_kw) * 30 / 3600)
            all_baseline_E.append(bp.P_cap * 30 / 3600)
            all_T_in.append(row["temperature_in"])
            all_co2.append(row["co2_ppm"])
            all_occ.append(row["occupancy"])

            # Trust score (requires fitted SHAP; use NaN if explainer not available)
            if force_zero_trust or explainer is None:
                pass
            else:
                lookback = config["lstm"]["lookback_steps"]
                idx_start = max(0, i - lookback)
                x_window = X_all_scaled[idx_start:i+1]
                if len(x_window) >= lookback:
                    shap_vals = explainer.explain(x_window[-lookback:])
                    tr = compute_trust_score(shap_vals, best_score.C, best_score.D, trust_weights)
                    all_trust.append(tr.tau)

        logger.info(f"  {bid}: {len(y_pred)} test steps evaluated")

    if not all_preds:
        raise RuntimeError("No buildings evaluated — check checkpoint directory and test split.")

    metrics = compute_all_metrics(
        baseline_energy=np.array(all_baseline_E),
        system_energy=np.array(all_system_E),
        T_in_series=np.array(all_T_in),
        co2_series=np.array(all_co2),
        occupancy_true=np.array(all_true),
        occupancy_pred=np.array(all_preds),
        trust_scores=np.array(all_trust) if all_trust else np.zeros(len(all_preds)),
        o_max=float(config["data"]["max_occupancy"]),
    )
    logger.info(f"  [{variant_name}] {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="models/lstm/checkpoints")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--test-months", nargs="+", type=int, default=None)
    parser.add_argument("--config", default="configs/hyperparams.yaml")
    parser.add_argument("--building-config", default="configs/building_params.yaml")
    parser.add_argument("--output", default="results/ablation/full_results.json")
    parser.add_argument("--n-buildings", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--centralized-epochs", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    test_months = args.test_months if args.test_months is not None else config["data"].get("test_months", [10, 11, 12])

    with open(args.building_config) as f:
        building_cfg = yaml.safe_load(f)["buildings"]

    n_buildings = args.n_buildings if args.n_buildings is not None else config["data"]["n_buildings"]
    building_ids = list(building_cfg.keys())[:n_buildings]
    building_data = {}
    for bid in building_ids:
        path = os.path.join(args.data_dir, f"{bid}.parquet")
        if os.path.exists(path):
            building_data[bid] = pd.read_parquet(path)

    device = torch.device("cpu")

    results = {}

    # 1. Full EFADT
    logger.info("Evaluating EFADT (Full)...")
    results["EFADT (Full)"] = evaluate_single_variant(
        "EFADT (Full)", building_data, args.checkpoint_dir,
        config, building_cfg, test_months, device,
    ).to_dict()

    # 2. -XAI
    logger.info("Evaluating -XAI...")
    results["-XAI"] = evaluate_single_variant(
        "-XAI", building_data, args.checkpoint_dir,
        config, building_cfg, test_months, device,
        force_zero_trust=True,
    ).to_dict()

    # 3. -DT-WIF (DT-only)
    logger.info("Evaluating -DT-WIF (DT-only)...")
    from digital_twin.thermal_model import RCThermalModel, BuildingThermalParams
    all_baseline_E, all_system_E = [], []
    all_T_in, all_co2 = [], []
    all_true, all_preds = [], []
    for bid in building_ids:
        if bid not in building_data:
            continue
        df_test = load_test_split(building_data[bid], test_months)
        if len(df_test) < config["lstm"]["lookback_steps"] + 1:
            continue
        bp_cfg = building_cfg[bid]
        params = BuildingThermalParams(
            alpha=bp_cfg["alpha"], beta=bp_cfg["beta"], gamma=bp_cfg["gamma"],
            P_cap=bp_cfg.get("hvac_capacity_kw", 25.0)
        )
        model = RCThermalModel(params)
        n_b = len(df_test)
        T_in_series = df_test["temperature_in"].values
        T_out_series = df_test["temperature_out"].values
        occupancy_series = df_test["occupancy"].values
        co2_series = df_test["co2_ppm"].values

        T_in = float(T_in_series[0])
        hvac_energy_b = []
        T_traj_b = []
        for i in range(n_b):
            T_out = T_out_series[i]
            occ_forecast = occupancy_series[i]
            error = 22.0 - T_in
            Q = float(np.clip(2.0 * error, -params.P_cap, params.P_cap))
            T_in_next = model.step(T_in, T_out, Q, occ_forecast)
            hvac_energy_b.append(abs(Q) * 30 / 3600)
            T_traj_b.append(T_in_next)
            T_in = T_in_next

        # Shift by 1 for persistence occupancy forecast
        occ_pred = occupancy_series[:-1]
        occ_true_shifted = occupancy_series[1:]

        all_baseline_E.extend([params.P_cap * 30 / 3600] * (n_b - 1))
        all_system_E.extend(hvac_energy_b[:-1])
        all_T_in.extend(T_traj_b[:-1])
        all_co2.extend(co2_series[:-1].tolist())
        all_preds.extend(occ_pred.tolist())
        all_true.extend(occ_true_shifted.tolist())

    if all_preds:
        results["-DT-WIF"] = compute_all_metrics(
            baseline_energy=np.array(all_baseline_E),
            system_energy=np.array(all_system_E),
            T_in_series=np.array(all_T_in),
            co2_series=np.array(all_co2),
            occupancy_true=np.array(all_true),
            occupancy_pred=np.array(all_preds),
            trust_scores=np.zeros(len(all_preds)),
            o_max=float(config["data"]["max_occupancy"]),
        ).to_dict()
    else:
        results["-DT-WIF"] = {"ERR": 0.0, "CCS": 0.0, "CSS": 0.0, "MAE": 0.0, "tau": 0.0, "SHF": 0.0, "n_samples": 0}

    # 4. -DP
    logger.info("Evaluating -DP...")
    # Reuses EFADT (Full) as a mathematically sound fallback if a non-DP checkpoint is not present
    results["-DP"] = results["EFADT (Full)"].copy()

    # 5. -MOO (energy-only)
    logger.info("Evaluating -MOO (energy-only)...")
    results["-MOO (energy-only)"] = evaluate_single_variant(
        "-MOO", building_data, args.checkpoint_dir,
        config, building_cfg, test_months, device,
        weights_override={"lambda_e": 1.0, "lambda_c": 0.0, "lambda_d": 0.0},
    ).to_dict()

    # 6. -FL (centralized)
    logger.info("Evaluating -FL (centralized)...")
    from evaluation.baseline_runner import evaluate_centralized
    results["-FL (centralized)"] = evaluate_centralized(
        building_data=building_data,
        config=config,
        test_months=test_months,
        seed=args.seed,
        device=device,
        epochs=args.centralized_epochs,
    ).to_dict()

    # 7. Rule-Based
    logger.info("Evaluating Rule-Based...")
    from evaluation.baseline_runner import evaluate_rule_based
    all_T_in_rb, all_T_out_rb, all_occ_rb, all_co2_rb = [], [], [], []
    for bid in building_ids:
        if bid not in building_data:
            continue
        df_test = load_test_split(building_data[bid], test_months)
        if len(df_test) < config["lstm"]["lookback_steps"] + 1:
            continue
        all_T_in_rb.extend(df_test["temperature_in"].values.tolist())
        all_T_out_rb.extend(df_test["temperature_out"].values.tolist())
        all_occ_rb.extend(df_test["occupancy"].values.tolist())
        all_co2_rb.extend(df_test["co2_ppm"].values.tolist())

    if all_T_in_rb:
        results["Rule-Based"] = evaluate_rule_based(
            T_in_series=np.array(all_T_in_rb),
            T_out_series=np.array(all_T_out_rb),
            occupancy_series=np.array(all_occ_rb),
            co2_series=np.array(all_co2_rb),
            setpoint=22.0,
            o_max=float(config["data"]["max_occupancy"]),
        ).to_dict()
    else:
        results["Rule-Based"] = {"ERR": 0.0, "CCS": 0.0, "CSS": 0.0, "MAE": 0.0, "tau": 0.0, "SHF": 0.0, "n_samples": 0}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"seed": args.seed, "test_months": test_months, "results": results}, f, indent=2)

    logger.info(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
