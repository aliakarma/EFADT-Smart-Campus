"""
EFADT — Full 30-Second Decision Cycle Orchestrator
====================================================
Implements Algorithm 1 in its entirety for one building node.

Pipeline stages and latency budget:
  Stage 1: LSTM inference (occupancy forecast)   ~67ms
  Stage 2: DT simulation (all candidate actions) ~27ms
  Stage 3: SHAP + trust scoring                  ~23ms
  ─────────────────────────────────────────────────────
  Total:                                         ~117ms (budget: 150ms)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from agent.action_space import BuildingAction
from agent.optimizer import EFADTAgent
from digital_twin.simulator import DigitalTwinSimulator, ActionScore
from digital_twin.thermal_model import ThermalState
from models.lstm.architecture import OccupancyLSTM
from xai.audit_logger import JSONLAuditLogger, SQLiteAuditLogger
from xai.shap_explainer import SHAPProxyExplainer, FEATURE_NAMES
from xai.trust_scorer import TrustWeights, compute_trust_score, TrustScoreResult

logger = logging.getLogger(__name__)


@dataclass
class DecisionCycleOutput:
    """Complete output of one 30-second decision cycle."""
    building_id: str
    timestamp: str
    selected_action: BuildingAction
    occ_forecast: np.ndarray
    shap_values: np.ndarray
    trust_result: TrustScoreResult
    all_scores: list[ActionScore]
    best_utility: float
    latency_ms: float
    latency_breakdown: dict = field(default_factory=dict)
    audit_hash: str = ""


class DecisionCycle:
    """
    Full EFADT per-building decision cycle.

    Parameters
    ----------
    building_id : str
    lstm_model : OccupancyLSTM
    simulator : DigitalTwinSimulator
    agent : EFADTAgent
    shap_explainer : SHAPProxyExplainer
    trust_weights : TrustWeights
    audit_logger : JSONLAuditLogger or SQLiteAuditLogger
    device : torch.device
    scaler : StandardScaler, optional
        Feature normalizer fitted during LSTM training.
    lookback : int
        LSTM input sequence length.
    latency_budget_ms : float
        Alert if pipeline exceeds this threshold.
    """

    def __init__(
        self,
        building_id: str,
        lstm_model: OccupancyLSTM,
        simulator: DigitalTwinSimulator,
        agent: EFADTAgent,
        shap_explainer: SHAPProxyExplainer,
        trust_weights: Optional[TrustWeights] = None,
        audit_logger=None,
        device: Optional[torch.device] = None,
        scaler=None,
        lookback: int = 12,
        latency_budget_ms: float = 150.0,
    ) -> None:
        self.building_id = building_id
        self.lstm_model = lstm_model
        self.simulator = simulator
        self.agent = agent
        self.shap_explainer = shap_explainer
        self.trust_weights = trust_weights or TrustWeights()
        self.audit_logger = audit_logger
        self.device = device or torch.device("cpu")
        assert scaler is not None, "StandardScaler must be provided for feature normalization."
        self.scaler = scaler
        self.lookback = lookback
        self.latency_budget_ms = latency_budget_ms
        self._sensor_buffer: list[np.ndarray] = []   # rolling window

    def push_sensor_reading(self, sensor_vector: np.ndarray) -> None:
        """
        Append a new 14-feature sensor vector to the rolling buffer.
        Call once per 30-second interval.
        """
        if self.scaler is not None:
            sensor_vector = self.scaler.transform(sensor_vector.reshape(1, -1))[0]
        self._sensor_buffer.append(sensor_vector.astype(np.float32))
        if len(self._sensor_buffer) > self.lookback:
            self._sensor_buffer.pop(0)

    def _get_lstm_input(self) -> Optional[np.ndarray]:
        """Return (lookback, n_features) array if buffer is full."""
        if len(self._sensor_buffer) < self.lookback:
            return None
        return np.stack(self._sensor_buffer, axis=0)   # (lookback, 14)

    # ── Main pipeline ────────────────────────────────────────────────────────

    def run(
        self,
        current_state: ThermalState,
        sensor_vector: np.ndarray,
    ) -> Optional[DecisionCycleOutput]:
        """
        Execute one complete 30-second decision cycle.

        Parameters
        ----------
        current_state : ThermalState
            Live sensor readings for DT synchronization.
        sensor_vector : np.ndarray, shape (14,)
            Raw feature vector from IoT sensors.

        Returns
        -------
        DecisionCycleOutput, or None if buffer not yet full.
        """
        from datetime import datetime, timezone
        t_total = time.time()
        latency = {}

        # ── Stage 0: Push sensor to buffer ──────────────────────────────────
        self.push_sensor_reading(sensor_vector)
        x_seq = self._get_lstm_input()
        if x_seq is None:
            logger.debug(f"{self.building_id}: Buffer not full ({len(self._sensor_buffer)}/{self.lookback})")
            return None

        # ── Stage 1: LSTM forecast (~67ms) ────────────────────────────────
        t1 = time.time()
        x_tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        occ_forecast_single = float(self.lstm_model.predict(x_tensor).item())
        # Replicate to H-step forecast (simplification: constant forecast)
        H = self.simulator.H
        occ_forecast = np.full(H, max(0.0, occ_forecast_single))
        latency["lstm_ms"] = (time.time() - t1) * 1000

        # ── Stage 2: DT simulation + agent decision (~27ms) ───────────────
        t2 = time.time()
        best_action, all_scores, best_utility = self.agent.decide(
            current_state=current_state,
            occ_forecast=occ_forecast,
        )
        latency["simulation_ms"] = (time.time() - t2) * 1000

        # ── Stage 3: SHAP + trust scoring (~23ms) ─────────────────────────
        t3 = time.time()
        shap_values = self.shap_explainer.explain(x_seq)
        best_score = next(s for s in all_scores if s.action_id == best_action.action_id)
        trust_result = compute_trust_score(
            shap_values=shap_values,
            C_u_star=best_score.C,
            D_u_star=best_score.D,
            weights=self.trust_weights,
        )
        latency["shap_ms"] = (time.time() - t3) * 1000

        total_ms = (time.time() - t_total) * 1000
        latency["total_ms"] = total_ms

        # Latency SLA check
        if total_ms > self.latency_budget_ms:
            logger.warning(
                f"{self.building_id}: LATENCY SLA EXCEEDED "
                f"({total_ms:.1f}ms > {self.latency_budget_ms}ms)"
            )

        # ── Stage 4: Audit logging ─────────────────────────────────────────
        audit_hash = ""
        if self.audit_logger is not None:
            rec = self.audit_logger.log(
                building_id=self.building_id,
                action_str=str(best_action),
                shap_values=shap_values,
                feature_names=FEATURE_NAMES,
                trust_score=trust_result.tau,
                extra={
                    "latency_ms": round(total_ms, 2),
                    "occ_forecast": round(occ_forecast_single, 1),
                    "best_utility": round(best_utility, 4),
                    "trust_below_threshold": trust_result.below_threshold,
                },
            )
            audit_hash = rec.record_hash

        logger.debug(
            f"{self.building_id} | {trust_result} | "
            f"latency={total_ms:.1f}ms | action={best_action}"
        )

        return DecisionCycleOutput(
            building_id=self.building_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            selected_action=best_action,
            occ_forecast=occ_forecast,
            shap_values=shap_values,
            trust_result=trust_result,
            all_scores=all_scores,
            best_utility=best_utility,
            latency_ms=total_ms,
            latency_breakdown=latency,
            audit_hash=audit_hash,
        )
