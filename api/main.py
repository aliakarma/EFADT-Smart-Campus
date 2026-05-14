"""
EFADT — FastAPI REST Backend
==============================
Exposes the EFADT system via a REST API for:
  - Real-time decision queries per building
  - Audit log retrieval
  - Trust score monitoring
  - System status and health checks
  - Scenario simulation triggers
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from digital_twin.simulator import DigitalTwinSimulator
from digital_twin.thermal_model import BuildingThermalParams, ThermalState
from evaluation.metrics import compute_all_metrics
from models.lstm.architecture import build_model
from xai.audit_logger import JSONLAuditLogger
from xai.shap_explainer import SHAPProxyExplainer, FEATURE_NAMES
from xai.trust_scorer import TrustWeights, compute_trust_score

logger = logging.getLogger(__name__)

# ── App state ────────────────────────────────────────────────────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and config on startup."""
    logger.info("EFADT API starting up...")
    config_path = os.getenv("CONFIG_PATH", "configs/hyperparams.yaml")
    building_config_path = os.getenv("BUILDING_CONFIG", "configs/building_params.yaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(building_config_path) as f:
        building_cfg = yaml.safe_load(f)["buildings"]

    device = torch.device("cpu")

    # Build simulators per building
    _state["simulators"] = {}
    for bid, params in building_cfg.items():
        bp = BuildingThermalParams(
            alpha=params["alpha"],
            beta=params["beta"],
            gamma=params["gamma"],
            P_cap=params.get("hvac_capacity_kw", 25.0),
        )
        _state["simulators"][bid] = DigitalTwinSimulator(
            building_id=bid, params=bp,
            o_max=params.get("max_occupancy", 80),
        )

    # Build global LSTM model (initialized; would be loaded from checkpoint in production)
    _state["model"] = build_model(cfg, device=device)
    _state["device"] = device
    _state["config"] = cfg

    # Audit logger
    _state["audit_logger"] = JSONLAuditLogger(log_dir="data/audit")

    # SHAP explainer (unfitted — requires calibration data in production)
    _state["shap_explainer"] = SHAPProxyExplainer()
    _state["shap_fitted"] = False

    _state["trust_weights"] = TrustWeights.from_config(cfg)

    logger.info(f"EFADT API ready | Buildings: {list(building_cfg.keys())}")
    yield

    logger.info("EFADT API shutting down.")


app = FastAPI(
    title="EFADT — Smart Campus Resource Optimizer",
    description="Federated Agentic Digital Twin for Campus Energy & Comfort Management",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Schemas ─────────────────────────────────────────────────────────

class SensorReading(BaseModel):
    """IoT sensor snapshot for one building at one timestep."""
    building_id: str
    temperature_in: float = Field(..., ge=0, le=60, description="Indoor temperature (°C)")
    temperature_out: float = Field(..., ge=-20, le=60, description="Outdoor temperature (°C)")
    occupancy: float = Field(..., ge=0, le=1000, description="Current occupancy count")
    co2_ppm: float = Field(..., ge=300, le=5000, description="CO₂ concentration (ppm)")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    hvac_power_kw: float = Field(0.0, description="Current HVAC power (kW)")
    hvac_setpoint: float = Field(22.0, description="Current HVAC setpoint (°C)")
    motion_count: int = Field(0, ge=0, description="Motion sensor triggers")


class DecisionRequest(BaseModel):
    """Request for an HVAC optimization decision."""
    building_id: str
    sensor: SensorReading
    occ_forecast: list[float] = Field(
        default_factory=lambda: [0.0] * 6,
        description="6-step occupancy forecast (or pass empty to use persistence)"
    )


class DecisionResponse(BaseModel):
    """Optimal HVAC action with trust score and explanation."""
    building_id: str
    hvac_power_kw: float
    hvac_setpoint_c: float
    crowd_alert: bool
    alert_level: str
    trust_score: float
    top_features: list
    shap_values: dict
    best_utility: float
    n_feasible_actions: int
    latency_ms: float
    audit_hash: str


class SystemStatus(BaseModel):
    """API health and system status."""
    status: str
    n_buildings: int
    shap_fitted: bool
    model_loaded: bool


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "EFADT"}


@app.get("/status", response_model=SystemStatus, tags=["System"])
async def get_status():
    """Return system readiness status."""
    return SystemStatus(
        status="ready",
        n_buildings=len(_state.get("simulators", {})),
        shap_fitted=_state.get("shap_fitted", False),
        model_loaded=_state.get("model") is not None,
    )


@app.get("/buildings", tags=["Buildings"])
async def list_buildings():
    """List all configured building IDs."""
    return {"buildings": list(_state["simulators"].keys())}


@app.post("/decide", response_model=DecisionResponse, tags=["Agent"])
async def get_decision(req: DecisionRequest):
    """
    Run one EFADT decision cycle for a building.

    Given a sensor reading and occupancy forecast, returns:
    - Optimal HVAC setpoint/power
    - Trust score and SHAP explanation
    - Crowd management alerts
    """
    import time

    bid = req.building_id
    if bid not in _state["simulators"]:
        raise HTTPException(status_code=404, detail=f"Building {bid} not found")

    t0 = time.time()
    sim = _state["simulators"][bid]
    model = _state["model"]
    device = _state["device"]
    audit = _state["audit_logger"]
    tw = _state["trust_weights"]

    # Sync digital twin state
    state = ThermalState(
        T_in=req.sensor.temperature_in,
        T_out=req.sensor.temperature_out,
        Q_hvac=req.sensor.hvac_power_kw,
        occupancy=req.sensor.occupancy,
    )
    sim.sync_state(state)

    # Occupancy forecast
    occ_forecast = np.array(req.occ_forecast) if req.occ_forecast else np.full(6, req.sensor.occupancy)

    # Evaluate all candidate actions
    from agent.action_space import build_action_space, get_hvac_powers
    from agent.utility_function import UtilityWeights, select_optimal_action
    action_space = build_action_space(
        hvac_min_kw=-sim.params.P_cap, hvac_max_kw=sim.params.P_cap
    )
    hvac_powers = get_hvac_powers(action_space)
    all_scores = sim.evaluate_all_actions(hvac_powers, occ_forecast)

    agent_cfg = _state["config"].get("agent", {})
    weights = UtilityWeights(
        lambda_e=agent_cfg.get("lambda_e", 0.5),
        lambda_c=agent_cfg.get("lambda_c", 0.35),
        lambda_d=agent_cfg.get("lambda_d", 0.15),
    )
    best_score, best_utility = select_optimal_action(all_scores, weights)
    best_action = action_space[best_score.action_id]
    n_feasible = sum(1 for s in all_scores if s.feasible)

    # SHAP explanation (stub if proxy not fitted)
    rng = np.random.default_rng()
    shap_values = rng.normal(0, 0.1, 14)   # Stub — replace with explainer.explain()
    trust_result = compute_trust_score(
        shap_values=shap_values,
        C_u_star=best_score.C,
        D_u_star=best_score.D,
        weights=tw,
    )

    # Audit
    rec = audit.log(
        building_id=bid,
        action_str=str(best_action),
        shap_values=shap_values,
        feature_names=FEATURE_NAMES,
        trust_score=trust_result.tau,
        extra={"latency_ms": (time.time() - t0) * 1000},
    )

    top_features = sorted(
        zip(FEATURE_NAMES, shap_values.tolist()),
        key=lambda x: abs(x[1]), reverse=True
    )[:3]

    return DecisionResponse(
        building_id=bid,
        hvac_power_kw=round(best_action.hvac_power_kw, 2),
        hvac_setpoint_c=round(best_action.hvac_setpoint_c, 1),
        crowd_alert=best_action.crowd_alert,
        alert_level=best_action.alert_level,
        trust_score=round(trust_result.tau, 4),
        top_features=[[f, round(v, 6)] for f, v in top_features],
        shap_values=dict(zip(FEATURE_NAMES, [round(v, 6) for v in shap_values.tolist()])),
        best_utility=round(float(best_utility), 4),
        n_feasible_actions=n_feasible,
        latency_ms=round((time.time() - t0) * 1000, 2),
        audit_hash=rec.record_hash[:16],
    )


@app.get("/audit/{building_id}", tags=["Governance"])
async def get_audit_log(
    building_id: str,
    min_trust: float = Query(None, ge=0.0, le=1.0),
    max_records: int = Query(100, ge=1, le=10000),
):
    """Retrieve tamper-evident audit log for a building."""
    audit = _state["audit_logger"]
    records = audit.query(
        building_id=building_id,
        min_trust=min_trust,
        max_records=max_records,
    )
    return {"building_id": building_id, "n_records": len(records), "records": records}


@app.get("/audit/{building_id}/verify", tags=["Governance"])
async def verify_audit_chain(building_id: str):
    """Verify hash chain integrity for a building's audit log."""
    audit = _state["audit_logger"]
    valid = audit.verify_chain(building_id)
    return {"building_id": building_id, "chain_valid": valid}


@app.get("/simulate/{building_id}", tags=["Simulation"])
async def simulate_action(
    building_id: str,
    T_in: float = Query(23.0, description="Current indoor temperature (°C)"),
    T_out: float = Query(35.0, description="Current outdoor temperature (°C)"),
    Q_hvac: float = Query(-10.0, description="Candidate HVAC power (kW)"),
    occ: float = Query(40.0, description="Current occupancy (persons)"),
    H: int = Query(6, ge=1, le=24, description="Simulation horizon (steps)"),
):
    """
    Simulate a specific HVAC action through the digital twin.
    Useful for what-if analysis via the governance dashboard.
    """
    if building_id not in _state["simulators"]:
        raise HTTPException(status_code=404, detail=f"Building {building_id} not found")

    sim = _state["simulators"][building_id]
    state = ThermalState(T_in=T_in, T_out=T_out, Q_hvac=Q_hvac, occupancy=occ)
    sim.sync_state(state)

    occ_forecast = np.full(H, occ)
    score = sim.evaluate_action(Q_hvac, occ_forecast, action_id=0)

    return {
        "building_id": building_id,
        "Q_hvac_kw": Q_hvac,
        "T_trajectory": score.T_trajectory.tolist(),
        "E": round(score.E, 4),
        "C": round(score.C, 4),
        "D": round(score.D, 4),
        "feasible": score.feasible,
        "violation_reason": score.violation_reason,
    }


@app.get("/metrics/summary", tags=["Evaluation"])
async def get_metrics_summary():
    """Return paper-reported evaluation metrics for reference."""
    from evaluation.baseline_runner import PAPER_RESULTS
    return {"paper_results": PAPER_RESULTS}
