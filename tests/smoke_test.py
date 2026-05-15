"""
EFADT — Smoke Test Suite
==========================
Quick end-to-end validation of all critical imports, data flow,
and pipeline initialization. Runs in under 60 seconds.

Usage:
    python tests/smoke_test.py
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np
import torch


def run_test(name: str, fn):
    """Run a named test and return (passed, elapsed_ms, error_msg)."""
    t0 = time.time()
    try:
        fn()
        elapsed = (time.time() - t0) * 1000
        print(f"  ✓  {name:<55} ({elapsed:.0f}ms)")
        return True, elapsed, None
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        msg = f"{type(e).__name__}: {e}"
        print(f"  ✗  {name:<55} → {msg}")
        return False, elapsed, msg


# ── Test Functions ────────────────────────────────────────────────────────────

def test_imports():
    """Verify all core module imports succeed."""
    import data.generation.occupancy_model
    import data.generation.thermal_simulator
    import data.generation.sensor_fault_injector
    import models.lstm.architecture
    import federated.dp_mechanism
    import digital_twin.thermal_model
    import digital_twin.simulator
    import agent.action_space
    import agent.utility_function
    import agent.optimizer
    import xai.shap_explainer
    import xai.trust_scorer
    import xai.audit_logger
    import evaluation.metrics
    import evaluation.baseline_runner
    import pipeline.decision_cycle


def test_config_loading():
    """Verify YAML configs load correctly."""
    import yaml
    with open("configs/hyperparams.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "lstm" in cfg
    assert "federated" in cfg
    assert "dp" in cfg
    assert "agent" in cfg
    assert "xai" in cfg
    assert cfg["lstm"]["hidden_size"] == 128
    assert cfg["federated"]["num_rounds"] == 100


def test_occupancy_model():
    """Verify occupancy data generation pipeline."""
    import pandas as pd
    from data.generation.occupancy_model import (
        generate_occupancy_series, generate_co2_from_occupancy
    )
    ts = pd.date_range("2024-01-15 08:00", periods=288, freq="30s")
    rng = np.random.default_rng(42)
    occ = generate_occupancy_series(288, np.array([5.0, 30.0, 70.0]), ts, max_occupancy=80, rng=rng)
    co2 = generate_co2_from_occupancy(occ, rng=rng)
    assert occ.shape == (288,)
    assert co2.shape == (288,)
    assert occ.min() >= 0
    assert occ.max() <= 80


def test_lstm_forward():
    """Verify LSTM forward pass runs correctly."""
    from models.lstm.architecture import OccupancyLSTM
    model = OccupancyLSTM(input_dim=14, hidden_size=64, num_layers=2)
    x = torch.randn(4, 12, 14)
    preds, (h, c) = model(x)
    assert preds.shape == (4, 1)
    assert not torch.any(torch.isnan(preds))


def test_dp_mechanism():
    """Verify DP gradient privatization."""
    from federated.dp_mechanism import privatize_gradient, compute_sigma
    sigma = compute_sigma(1.0, 1e-5)
    assert 1.0 < sigma < 2.0
    rng = np.random.default_rng(42)
    grad = rng.normal(0, 0.5, 50000)
    noised, info = privatize_gradient(grad, epsilon=1.0, delta=1e-5, rng=rng)
    assert noised.shape == grad.shape


def test_digital_twin_simulation():
    """Verify DT simulation produces valid outputs."""
    from digital_twin.simulator import DigitalTwinSimulator
    from digital_twin.thermal_model import BuildingThermalParams, ThermalState
    params = BuildingThermalParams(alpha=0.0018, beta=0.011, gamma=0.009)
    sim = DigitalTwinSimulator("B01", params=params, H=6, o_max=80)
    state = ThermalState(T_in=26.0, T_out=38.0, Q_hvac=0.0, occupancy=50)
    sim.sync_state(state)
    occ_forecast = np.full(6, 50.0)
    score = sim.evaluate_action(-12.0, occ_forecast)
    assert 0.0 <= score.E <= 1.0
    assert 0.0 <= score.C <= 1.0
    assert score.T_trajectory.shape == (6,)


def test_agent_decision():
    """Verify agent decision cycle runs end-to-end."""
    from agent.optimizer import EFADTAgent
    from digital_twin.simulator import DigitalTwinSimulator
    from digital_twin.thermal_model import BuildingThermalParams, ThermalState
    params = BuildingThermalParams(alpha=0.0018, beta=0.011, gamma=0.009)
    sim = DigitalTwinSimulator("B01", params=params, H=6, o_max=80, P_cap=25.0)
    agent = EFADTAgent("B01", sim)
    state = ThermalState(T_in=27.0, T_out=40.0, Q_hvac=0.0, occupancy=60)
    occ_forecast = np.full(6, 60.0)
    best_action, all_scores, utility = agent.decide(state, occ_forecast)
    assert best_action is not None
    assert len(all_scores) > 0
    assert isinstance(utility, float)


def test_shap_explainer():
    """Verify SHAP proxy explainer fits and explains."""
    from xai.shap_explainer import SHAPProxyExplainer
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (200, 12, 14))
    preds = rng.normal(20, 5, 200)
    explainer = SHAPProxyExplainer(n_estimators=10, max_depth=3)
    explainer.fit(X, preds)
    x_test = rng.normal(0, 1, (12, 14))
    shap_vals = explainer.explain(x_test)
    assert shap_vals.shape == (14,)


def test_trust_scorer():
    """Verify trust score computation."""
    from xai.trust_scorer import compute_trust_score
    rng = np.random.default_rng(42)
    shap_vals = rng.normal(0, 0.3, 14)
    result = compute_trust_score(shap_vals, C_u_star=0.9, D_u_star=0.3)
    assert 0.0 <= result.tau <= 1.0


def test_audit_logger(tmp_dir="/tmp/efadt_smoke_audit"):
    """Verify tamper-evident audit log write and verify."""
    import os
    from xai.audit_logger import JSONLAuditLogger
    os.makedirs(tmp_dir, exist_ok=True)
    audit = JSONLAuditLogger(log_dir=tmp_dir)
    feature_names = [f"f{i}" for i in range(14)]
    for i in range(3):
        shap_vals = np.random.randn(14)
        audit.log("B01", f"Q={i}", shap_vals, feature_names, trust_score=0.88)
    assert audit.verify_chain("B01")


def test_evaluation_metrics():
    """Verify metric computations match expected ranges."""
    from evaluation.metrics import compute_all_metrics
    n = 1000
    rng = np.random.default_rng(42)
    baseline_E = np.full(n, 10.0)
    system_E = np.full(n, 6.53)
    T_in = np.full(n, 22.0)
    co2 = np.full(n, 600.0)
    occ = np.full(n, 40.0)
    occ_pred = occ + rng.normal(0, 3.0, n)
    trust = np.full(n, 0.887)
    proxy = rng.normal(0, 1, n)
    lstm = proxy + rng.normal(0, 0.1, n)
    metrics = compute_all_metrics(baseline_E, system_E, T_in, co2, occ, occ_pred, trust, proxy, lstm)
    assert 30.0 < metrics.ERR < 40.0
    assert metrics.CCS == 1.0
    assert metrics.n_samples == n


def test_api_startup():
    """Verify FastAPI app can start without errors."""
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
        resp2 = client.get("/buildings")
        assert resp2.status_code == 200
        assert len(resp2.json()["buildings"]) > 0


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    tests = [
        ("Core imports",              test_imports),
        ("Config loading",            test_config_loading),
        ("Occupancy model",           test_occupancy_model),
        ("LSTM forward pass",         test_lstm_forward),
        ("DP mechanism",              test_dp_mechanism),
        ("Digital twin simulation",   test_digital_twin_simulation),
        ("Agent decision cycle",      test_agent_decision),
        ("SHAP proxy explainer",      test_shap_explainer),
        ("Trust scorer",              test_trust_scorer),
        ("Audit logger + hash chain", test_audit_logger),
        ("Evaluation metrics",        test_evaluation_metrics),
        ("API startup",               test_api_startup),
    ]

    print("\n" + "=" * 70)
    print("  EFADT Smoke Test Suite")
    print("=" * 70)

    results = []
    t_start = time.time()
    for name, fn in tests:
        passed, elapsed, error = run_test(name, fn)
        results.append((name, passed, elapsed, error))

    total_elapsed = (time.time() - t_start) * 1000
    n_passed = sum(1 for _, p, _, _ in results if p)
    n_failed = len(results) - n_passed

    print("=" * 70)
    print(f"  Results: {n_passed}/{len(results)} passed | Total: {total_elapsed:.0f}ms")
    if n_failed > 0:
        print(f"\n  FAILURES ({n_failed}):")
        for name, passed, _, error in results:
            if not passed:
                print(f"    ✗ {name}: {error}")
    print("=" * 70 + "\n")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    # Need to run from repo root
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.exit(main())
