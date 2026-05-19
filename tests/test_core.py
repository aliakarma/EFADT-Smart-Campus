"""
EFADT — Core Test Suite
========================
Tests for:
  - Data generation (occupancy, thermal, faults)
  - LSTM architecture (forward pass, shapes)
  - DP mechanism (clipping, noise, sigma)
  - Digital twin simulator (RC step, horizon, scores)
  - Agent (action space, utility, optimizer)
  - XAI (SHAP, trust scorer, audit logger)
  - Evaluation metrics
"""

import hashlib
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def timestamps():
    return pd.date_range("2024-01-15 08:00", periods=100, freq="30s")


@pytest.fixture(scope="module")
def dummy_thermal_params():
    from digital_twin.thermal_model import BuildingThermalParams
    return BuildingThermalParams(alpha=0.0018, beta=0.011, gamma=0.009)


@pytest.fixture(scope="module")
def dummy_simulator(dummy_thermal_params):
    from digital_twin.simulator import DigitalTwinSimulator
    return DigitalTwinSimulator("B01", params=dummy_thermal_params, H=6, o_max=80)


@pytest.fixture(scope="module")
def synced_simulator(dummy_simulator):
    from digital_twin.thermal_model import ThermalState
    state = ThermalState(T_in=25.0, T_out=38.0, Q_hvac=0.0, occupancy=40)
    dummy_simulator.sync_state(state)
    return dummy_simulator


# ── Data Generation ───────────────────────────────────────────────────────────

class TestOccupancyModel:
    def test_output_shape(self, timestamps, rng):
        from data.generation.occupancy_model import generate_occupancy_series
        base_rates = np.array([3.0, 25.0, 65.0])
        occ = generate_occupancy_series(100, base_rates, timestamps, max_occupancy=80, rng=rng)
        assert occ.shape == (100,)

    def test_bounded_output(self, timestamps, rng):
        from data.generation.occupancy_model import generate_occupancy_series
        base_rates = np.array([3.0, 25.0, 65.0])
        occ = generate_occupancy_series(100, base_rates, timestamps, max_occupancy=80, rng=rng)
        assert occ.min() >= 0
        assert occ.max() <= 80

    def test_integer_output(self, timestamps, rng):
        from data.generation.occupancy_model import generate_occupancy_series
        base_rates = np.array([3.0, 25.0, 65.0])
        occ = generate_occupancy_series(100, base_rates, timestamps, max_occupancy=80, rng=rng)
        assert occ.dtype in (np.int32, np.int64)

    def test_co2_from_occupancy(self, rng):
        from data.generation.occupancy_model import generate_co2_from_occupancy
        occ = np.array([0, 10, 50, 80])
        co2 = generate_co2_from_occupancy(occ, rng=rng)
        assert co2.shape == (4,)
        assert co2.min() >= 400.0
        assert co2.max() <= 2000.0
        # CO2 should generally increase with occupancy (ignoring noise for large differences)
        assert float(co2[3]) > float(co2[0]) - 200  # with noise tolerance

    def test_thermal_series(self, timestamps, rng):
        from data.generation.occupancy_model import generate_occupancy_series
        from data.generation.thermal_simulator import simulate_building_thermal_series
        base_rates = np.array([3.0, 25.0, 65.0])
        occ = generate_occupancy_series(100, base_rates, timestamps, max_occupancy=80, rng=rng)
        hvac = np.full(100, 0.0)
        T_in, T_out = simulate_building_thermal_series(
            100, occ, hvac, timestamps, alpha=0.0018, beta=0.011, gamma=0.009
        )
        assert T_in.shape == (100,)
        assert T_out.shape == (100,)
        assert not np.any(np.isnan(T_in))

    def test_fault_injection(self, rng):
        from data.generation.sensor_fault_injector import inject_sensor_faults
        n = 500
        ts = pd.date_range("2024-01-01", periods=n, freq="30s")
        df = pd.DataFrame({
            "occupancy": np.random.randint(0, 50, n),
            "temperature": np.random.normal(22, 2, n),
        }, index=ts)
        df_faulty = inject_sensor_faults(df, fault_rate=0.20, rng=rng)
        assert "sensor_fault" in df_faulty.columns
        fault_rate = df_faulty["sensor_fault"].mean()
        # Should have injected faults (0.10–0.40 is a reasonable range given batching)
        assert 0.05 <= fault_rate <= 0.50


# ── LSTM Architecture ─────────────────────────────────────────────────────────

class TestOccupancyLSTM:
    def test_forward_shape(self):
        from models.lstm.architecture import OccupancyLSTM
        model = OccupancyLSTM(input_dim=14, hidden_size=64, num_layers=2)
        x = torch.randn(8, 12, 14)
        preds, (h, c) = model(x)
        assert preds.shape == (8, 1)
        assert h.shape == (2, 8, 64)

    def test_predict_nonnegative(self):
        from models.lstm.architecture import OccupancyLSTM
        model = OccupancyLSTM()
        x = torch.randn(4, 12, 14)
        preds = model.predict(x)
        assert torch.all(preds >= 0.0)

    def test_parameter_count(self):
        from models.lstm.architecture import OccupancyLSTM
        model = OccupancyLSTM(input_dim=14, hidden_size=128, num_layers=2)
        n_params = model.get_parameter_count()
        assert n_params > 0
        assert n_params < 2_000_000  # sanity upper bound

    def test_build_model_from_config(self):
        from models.lstm.architecture import build_model
        cfg = {"lstm": {"input_dim": 14, "hidden_size": 64, "num_layers": 2,
                         "dropout": 0.1, "output_dim": 1}}
        model = build_model(cfg, device=torch.device("cpu"))
        assert model is not None
        x = torch.randn(2, 12, 14)
        out, _ = model(x)
        assert out.shape == (2, 1)

    def test_campus_dataset(self):
        from models.lstm.architecture import CampusDataset
        X = np.random.randn(200, 14).astype(np.float32)
        y = np.random.randn(200).astype(np.float32)
        ds = CampusDataset(X, y, lookback=12)
        assert len(ds) == 200 - 12
        x_seq, y_val = ds[0]
        assert x_seq.shape == (12, 14)
        assert y_val.shape == ()


# ── Differential Privacy ─────────────────────────────────────────────────────

class TestDPMechanism:
    def test_sigma_computation(self):
        from federated.dp_mechanism import compute_sigma
        sigma = compute_sigma(epsilon=1.0, delta=1e-5)
        assert sigma > 1.0  # sigma must be positive and meaningful
        assert abs(sigma - 4.845) < 0.01  # ~4.845 for eps=1.0, delta=1e-5

    def test_gradient_clipping(self):
        from federated.dp_mechanism import clip_gradient
        grad = np.array([3.0, 4.0])   # norm = 5.0
        clipped, norm = clip_gradient(grad, clip_norm=1.0)
        assert abs(np.linalg.norm(clipped) - 1.0) < 1e-6
        assert abs(norm - 5.0) < 1e-6

    def test_no_clip_when_under(self):
        from federated.dp_mechanism import clip_gradient
        grad = np.array([0.3, 0.4])   # norm = 0.5 < 1.0
        clipped, norm = clip_gradient(grad, clip_norm=1.0)
        np.testing.assert_array_almost_equal(clipped, grad)

    def test_gaussian_noise_shape(self, rng):
        from federated.dp_mechanism import add_gaussian_noise
        grad = np.zeros(100)
        noised = add_gaussian_noise(grad, sigma=1.0, clip_norm=1.0, rng=rng)
        assert noised.shape == (100,)
        assert not np.allclose(noised, 0.0)  # Noise was added

    def test_privatize_gradient_roundtrip(self, rng):
        from federated.dp_mechanism import privatize_gradient
        grad = np.random.randn(1000)
        noised, info = privatize_gradient(grad, epsilon=1.0, delta=1e-5, rng=rng)
        assert noised.shape == grad.shape
        assert "sigma" in info
        assert "was_clipped" in info

    def test_privatize_model_update(self, rng):
        from federated.dp_mechanism import privatize_model_update
        local = [np.random.randn(10, 5), np.random.randn(5)]
        glob = [np.random.randn(10, 5), np.random.randn(5)]
        noised, info = privatize_model_update(local, glob, epsilon=1.0, delta=1e-5, rng=rng)
        assert len(noised) == len(local)
        assert noised[0].shape == local[0].shape


# ── Digital Twin ──────────────────────────────────────────────────────────────

class TestRCThermalModel:
    def test_euler_step_cooling(self, dummy_thermal_params):
        from digital_twin.thermal_model import RCThermalModel
        model = RCThermalModel(dummy_thermal_params)
        T_in = 28.0  # hot room
        T_out = 35.0
        Q_hvac = -20.0  # cooling
        T_new = model.step(T_in, T_out, Q_hvac, occ=30)
        # With aggressive cooling, temperature should decrease
        assert T_new < T_in

    def test_horizon_shape(self, dummy_thermal_params):
        from digital_twin.thermal_model import RCThermalModel
        model = RCThermalModel(dummy_thermal_params)
        occ_forecast = np.full(6, 30.0)
        traj = model.simulate_horizon(T_in_0=23.0, T_out=35.0, Q_hvac=-10.0,
                                       occ_forecast=occ_forecast, H=6)
        assert traj.shape == (6,)
        assert not np.any(np.isnan(traj))

    def test_comfort_check(self, dummy_thermal_params):
        from digital_twin.thermal_model import RCThermalModel
        model = RCThermalModel(dummy_thermal_params)
        T_ok = np.array([21.0, 22.0, 23.0, 24.0, 25.0, 25.5])
        T_bad = np.array([21.0, 22.0, 27.0, 28.0, 25.0, 24.0])  # exceeds T_max
        assert model.is_comfort_compliant(T_ok)
        assert not model.is_comfort_compliant(T_bad)

    def test_energy_cost_normalized(self, dummy_thermal_params):
        from digital_twin.thermal_model import RCThermalModel
        model = RCThermalModel(dummy_thermal_params)
        E = model.compute_energy_cost(Q_hvac=25.0)
        assert 0.0 <= E <= 1.0
        E_zero = model.compute_energy_cost(Q_hvac=0.0)
        assert E_zero == 0.0

    def test_comfort_score(self, dummy_thermal_params):
        from digital_twin.thermal_model import RCThermalModel
        model = RCThermalModel(dummy_thermal_params)
        T_all_ok = np.full(6, 22.0)
        T_none_ok = np.full(6, 30.0)  # above T_max
        assert model.compute_comfort_score(T_all_ok) == 1.0
        assert model.compute_comfort_score(T_none_ok) == 0.0


class TestDigitalTwinSimulator:
    def test_evaluate_action(self, synced_simulator):
        from digital_twin.simulator import ActionScore
        occ_forecast = np.full(6, 40.0)
        score = synced_simulator.evaluate_action(-10.0, occ_forecast, action_id=0)
        assert isinstance(score, ActionScore)
        assert 0.0 <= score.E <= 1.0
        assert 0.0 <= score.C <= 1.0
        assert 0.0 <= score.D <= 1.0
        assert isinstance(score.feasible, bool)

    def test_evaluate_all_actions(self, synced_simulator):
        candidates = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
        occ_forecast = np.full(6, 40.0)
        scores = synced_simulator.evaluate_all_actions(candidates, occ_forecast)
        assert len(scores) == 5
        for s in scores:
            assert 0.0 <= s.E <= 1.0

    def test_power_violation(self, synced_simulator):
        from digital_twin.thermal_model import ThermalState
        state = ThermalState(T_in=22.0, T_out=30.0, Q_hvac=0.0, occupancy=30)
        synced_simulator.sync_state(state)
        occ_forecast = np.full(6, 30.0)
        # Exceed capacity
        score = synced_simulator.evaluate_action(100.0, occ_forecast)
        assert not score.feasible
        assert not score.feasible  # infeasible due to extreme power/comfort violation


# ── Agent ─────────────────────────────────────────────────────────────────────

class TestActionSpace:
    def test_build_action_space(self):
        from agent.action_space import build_action_space
        actions = build_action_space()
        assert len(actions) > 0
        for a in actions:
            assert hasattr(a, "hvac_power_kw")
            assert hasattr(a, "hvac_setpoint_c")
            assert a.action_id >= 0

    def test_get_hvac_powers(self):
        from agent.action_space import build_action_space, get_hvac_powers
        actions = build_action_space(hvac_min_kw=-10.0, hvac_max_kw=10.0, hvac_step_kw=5.0)
        powers = get_hvac_powers(actions)
        assert isinstance(powers, np.ndarray)
        assert len(powers) == len(actions)


class TestUtilityFunction:
    def test_compute_utility(self):
        from agent.utility_function import UtilityWeights, compute_utility
        from digital_twin.simulator import ActionScore
        weights = UtilityWeights(lambda_e=0.5, lambda_c=0.35, lambda_d=0.15)
        score = ActionScore(
            action_id=0, hvac_power_kw=-10.0,
            E=0.4, C=0.9, D=0.3,
            T_trajectory=np.zeros(6), feasible=True
        )
        u = compute_utility(score, weights)
        expected = 0.5 * 0.4 - 0.35 * 0.9 + 0.15 * 0.3
        assert abs(u - expected) < 1e-6

    def test_select_optimal_action(self):
        from agent.utility_function import UtilityWeights, select_optimal_action
        from digital_twin.simulator import ActionScore
        weights = UtilityWeights()
        scores = [
            ActionScore(0, -10.0, E=0.3, C=0.9, D=0.2, T_trajectory=np.zeros(6), feasible=True),
            ActionScore(1, -5.0,  E=0.2, C=0.7, D=0.3, T_trajectory=np.zeros(6), feasible=True),
            ActionScore(2, 0.0,   E=0.0, C=0.5, D=0.1, T_trajectory=np.zeros(6), feasible=True),
        ]
        best, utility = select_optimal_action(scores, weights)
        assert best is not None
        assert isinstance(utility, float)

    def test_no_feasible_fallback(self):
        from agent.utility_function import UtilityWeights, select_optimal_action
        from digital_twin.simulator import ActionScore
        weights = UtilityWeights()
        scores = [
            ActionScore(0, -10.0, E=0.3, C=0.9, D=0.2, T_trajectory=np.zeros(6),
                        feasible=False, violation_reason="Comfort"),
        ]
        best, utility = select_optimal_action(scores, weights, fallback_action_id=0)
        assert best is not None  # fallback returned
        assert utility == float("inf")

    def test_energy_only_ablation(self):
        from agent.utility_function import UtilityWeights
        w = UtilityWeights.energy_only()
        assert w.lambda_e == 1.0
        assert w.lambda_c == 0.0
        assert w.lambda_d == 0.0


# ── XAI & Trust ───────────────────────────────────────────────────────────────

class TestSHAPExplainer:
    def test_fit_and_explain(self):
        from xai.shap_explainer import SHAPProxyExplainer
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(0, 1, (n, 12, 14))
        preds = rng.normal(20, 5, n)
        explainer = SHAPProxyExplainer(n_estimators=20, max_depth=3)
        explainer.fit(X, preds)
        x_test = rng.normal(0, 1, (12, 14))
        shap_vals = explainer.explain(x_test)
        assert shap_vals.shape == (14,)

    def test_top_features(self):
        from xai.shap_explainer import SHAPProxyExplainer
        explainer = SHAPProxyExplainer()
        shap_vals = np.array([0.1, -0.5, 0.3] + [0.0] * 11)
        top3 = explainer.get_top_features(shap_vals, k=3)
        assert len(top3) == 3
        assert top3[0][0] == explainer.feature_names[1]  # highest abs val = -0.5


class TestTrustScorer:
    def test_trust_score_range(self):
        from xai.trust_scorer import compute_trust_score
        rng = np.random.default_rng(42)
        shap_vals = rng.normal(0, 0.3, 14)
        result = compute_trust_score(shap_vals, C_u_star=0.9, D_u_star=0.3)
        assert 0.0 <= result.tau <= 1.0

    def test_high_trust_scenario(self):
        from xai.trust_scorer import compute_trust_score
        # All positive SHAP, high comfort, low crowd density → high trust
        shap_vals = np.ones(14) * 0.1  # all positive
        result = compute_trust_score(shap_vals, C_u_star=1.0, D_u_star=0.0)
        assert result.tau > 0.7

    def test_low_trust_flag(self):
        from xai.trust_scorer import compute_trust_score
        shap_vals = np.array([-0.5] * 14)  # all negative SHAP
        result = compute_trust_score(shap_vals, C_u_star=0.0, D_u_star=1.0, alert_threshold=0.7)
        assert result.tau < 0.7
        assert result.below_threshold

    def test_shap_coherence_zero_mass(self):
        from xai.trust_scorer import compute_shap_coherence
        # All zeros → 0.0
        result = compute_shap_coherence(np.zeros(14))
        assert result == 0.0


class TestAuditLogger:
    def test_log_and_retrieve(self, tmp_path):
        from xai.audit_logger import JSONLAuditLogger
        audit = JSONLAuditLogger(log_dir=str(tmp_path))
        feature_names = [f"f{i}" for i in range(14)]
        shap_vals = np.random.randn(14)

        rec = audit.log("B01", "Q=-10kW", shap_vals, feature_names, trust_score=0.88)
        assert len(rec.record_hash) == 64   # SHA-256 hex
        assert rec.trust_score == round(0.88, 4)

    def test_chain_integrity(self, tmp_path):
        from xai.audit_logger import JSONLAuditLogger
        audit = JSONLAuditLogger(log_dir=str(tmp_path))
        feature_names = [f"f{i}" for i in range(14)]
        for i in range(5):
            shap_vals = np.random.randn(14)
            audit.log("B02", f"Q={i}", shap_vals, feature_names, trust_score=0.85)

        assert audit.verify_chain("B02") is True

    def test_sqlite_logger(self, tmp_path):
        from xai.audit_logger import SQLiteAuditLogger
        db_path = str(tmp_path / "test_audit.db")
        audit = SQLiteAuditLogger(db_path=db_path)
        feature_names = [f"f{i}" for i in range(14)]
        shap_vals = np.random.randn(14)
        rec = audit.log("B03", "Q=-5kW", shap_vals, feature_names, trust_score=0.90)
        assert rec.record_hash is not None
        assert os.path.exists(db_path)


# ── Evaluation Metrics ────────────────────────────────────────────────────────

class TestEvaluationMetrics:
    def test_err_computation(self):
        from evaluation.metrics import energy_reduction_ratio
        baseline = np.full(100, 10.0)
        system = np.full(100, 6.53)   # ~34.7% reduction
        err = energy_reduction_ratio(baseline, system)
        assert 30.0 < err < 40.0

    def test_ccs_all_ok(self):
        from evaluation.metrics import comfort_compliance_score
        T_in = np.full(100, 22.0)
        co2 = np.full(100, 600.0)
        ccs = comfort_compliance_score(T_in, co2)
        assert ccs == 1.0

    def test_css_all_safe(self):
        from evaluation.metrics import crowd_safety_score
        occ = np.full(100, 50.0)
        css = crowd_safety_score(occ, o_max=80.0)
        assert css == 1.0

    def test_mae(self):
        from evaluation.metrics import mean_absolute_error
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([13.0, 17.0, 27.0])
        mae = mean_absolute_error(y_true, y_pred)
        assert abs(mae - 3.0) < 1e-6

    def test_shap_fidelity(self):
        from evaluation.metrics import shap_fidelity
        x = np.arange(100, dtype=float)
        # Perfect correlation
        shf = shap_fidelity(x, x)
        assert abs(shf - 1.0) < 1e-6
        # Anti-correlated (should clamp to 0)
        shf_neg = shap_fidelity(x, -x)
        assert shf_neg == 0.0

    def test_compute_all_metrics(self):
        from evaluation.metrics import compute_all_metrics
        n = 200
        rng = np.random.default_rng(42)
        baseline_E = np.full(n, 10.0)
        system_E = np.full(n, 6.5)
        T_in = np.full(n, 22.0)
        co2 = np.full(n, 600.0)
        occ = np.full(n, 40.0)
        occ_pred = occ + rng.normal(0, 3.0, n)
        trust = np.full(n, 0.887)
        proxy = rng.normal(0, 1, n)
        lstm = proxy + rng.normal(0, 0.1, n)

        metrics = compute_all_metrics(baseline_E, system_E, T_in, co2, occ, occ_pred,
                                       trust, proxy, lstm)
        assert isinstance(metrics.ERR, float)
        assert isinstance(metrics.CCS, float)
        assert metrics.n_samples == n

    def test_significance_test(self):
        from evaluation.metrics import significance_test
        ref = [17.32, 17.30, 17.34]
        other = [19.33, 19.30, 19.36]
        res = significance_test(ref, other)
        assert "p_value" in res
        assert "cohens_d" in res
        assert res["cohens_d"] != 0.0
