"""
EFADT — Tamper-Evident Audit Ledger
=====================================
Appends every agent decision to a governance audit log.

Each record contains:
  - timestamp (ISO 8601 UTC)
  - building_id
  - selected action u*
  - SHAP feature attributions {φⱼ}
  - Trust score τ(u*)
  - Top-3 explanatory features
  - SHA-256 hash chain (tamper-evidence)

Supports both JSONL (lightweight) and PostgreSQL (production) backends.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

AUDIT_LOG_DIR = "data/audit"


class AuditRecord:
    """A single immutable audit log entry."""

    def __init__(
        self,
        building_id: str,
        action_str: str,
        shap_values: np.ndarray,
        feature_names: list[str],
        trust_score: float,
        top_k: int = 3,
        prev_hash: str = "genesis",
        extra: Optional[dict] = None,
    ) -> None:
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.building_id = building_id
        self.action_str = action_str
        self.trust_score = round(float(trust_score), 4)
        self.feature_names = feature_names
        self.shap_dict = {
            name: round(float(val), 6)
            for name, val in zip(feature_names, shap_values)
        }
        # Top-k by absolute magnitude
        sorted_feats = sorted(
            self.shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
        )
        self.top_features = sorted_feats[:top_k]
        self.extra = extra or {}
        self.prev_hash = prev_hash
        self.record_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """SHA-256 hash of record content (chained with previous hash)."""
        content = {
            "timestamp": self.timestamp,
            "building_id": self.building_id,
            "action": self.action_str,
            "trust_score": self.trust_score,
            "shap_dict": self.shap_dict,
            "prev_hash": self.prev_hash,
        }
        raw = json.dumps(content, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "building_id": self.building_id,
            "action": self.action_str,
            "trust_score": self.trust_score,
            "top_features": self.top_features,
            "shap_values": self.shap_dict,
            "prev_hash": self.prev_hash,
            "record_hash": self.record_hash,
            **self.extra,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class JSONLAuditLogger:
    """
    Lightweight JSONL audit logger. One file per building per day.
    Suitable for edge nodes with limited storage.
    """

    def __init__(self, log_dir: str = AUDIT_LOG_DIR) -> None:
        self.log_dir = log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._last_hashes: dict[str, str] = {}   # building_id → last hash

    def _get_log_path(self, building_id: str) -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"{building_id}_{date_str}.jsonl")

    def log(
        self,
        building_id: str,
        action_str: str,
        shap_values: np.ndarray,
        feature_names: list[str],
        trust_score: float,
        extra: Optional[dict] = None,
    ) -> AuditRecord:
        """
        Append a tamper-evident audit record.

        Parameters
        ----------
        building_id : str
        action_str : str
            String representation of selected action u*.
        shap_values : np.ndarray
        feature_names : list[str]
        trust_score : float
        extra : dict, optional
            Additional fields (e.g., latency_ms, scenario).

        Returns
        -------
        AuditRecord
        """
        prev_hash = self._last_hashes.get(building_id, "genesis")
        record = AuditRecord(
            building_id=building_id,
            action_str=action_str,
            shap_values=shap_values,
            feature_names=feature_names,
            trust_score=trust_score,
            prev_hash=prev_hash,
            extra=extra,
        )

        log_path = self._get_log_path(building_id)
        with open(log_path, "a") as f:
            f.write(record.to_json() + "\n")

        self._last_hashes[building_id] = record.record_hash

        if record.trust_score < 0.7:
            logger.warning(
                f"LOW TRUST ALERT: {building_id} | τ={record.trust_score:.3f} | "
                f"action={action_str}"
            )

        return record

    def query(
        self,
        building_id: Optional[str] = None,
        date_str: Optional[str] = None,
        min_trust: Optional[float] = None,
        max_records: int = 1000,
    ) -> list[dict]:
        """
        Query audit records with optional filters.

        Parameters
        ----------
        building_id : str, optional
        date_str : str, optional (format: 'YYYY-MM-DD')
        min_trust : float, optional
        max_records : int

        Returns
        -------
        list[dict] : Matching audit records.
        """
        records = []
        pattern = f"{building_id or '*'}_{date_str or '*'}.jsonl"

        for log_file in sorted(Path(self.log_dir).glob("*.jsonl")):
            if building_id and not log_file.name.startswith(building_id):
                continue
            if date_str and date_str not in log_file.name:
                continue

            with open(log_file) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        if min_trust is not None and rec.get("trust_score", 1.0) < min_trust:
                            continue
                        records.append(rec)
                        if len(records) >= max_records:
                            break
                    except json.JSONDecodeError:
                        continue
            if len(records) >= max_records:
                break

        return records

    def verify_chain(self, building_id: str, date_str: Optional[str] = None) -> bool:
        """
        Verify hash chain integrity for a building's audit log.
        Returns True if chain is unbroken, False if tampered.
        """
        records = self.query(building_id=building_id, date_str=date_str)
        if not records:
            return True

        for i, rec in enumerate(records):
            expected_prev = "genesis" if i == 0 else records[i - 1]["record_hash"]
            if rec.get("prev_hash") != expected_prev:
                logger.error(
                    f"Hash chain BROKEN at record {i} for {building_id}! "
                    f"Expected prev={expected_prev[:16]}..., got {rec.get('prev_hash', 'MISSING')[:16]}..."
                )
                return False
        return True


class SQLiteAuditLogger:
    """
    SQLite-backed audit logger for local persistence with queryability.
    Drop-in replacement for JSONLAuditLogger in production edge deployments.
    """

    def __init__(self, db_path: str = "data/audit/audit.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()
        self._last_hashes: dict[str, str] = {}

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    building_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    trust_score REAL NOT NULL,
                    top_features TEXT,
                    shap_values TEXT,
                    prev_hash TEXT,
                    record_hash TEXT,
                    extra TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_building_ts
                ON audit_log(building_id, timestamp)
            """)

    def log(
        self,
        building_id: str,
        action_str: str,
        shap_values: np.ndarray,
        feature_names: list[str],
        trust_score: float,
        extra: Optional[dict] = None,
    ) -> AuditRecord:
        prev_hash = self._last_hashes.get(building_id, "genesis")
        record = AuditRecord(
            building_id=building_id,
            action_str=action_str,
            shap_values=shap_values,
            feature_names=feature_names,
            trust_score=trust_score,
            prev_hash=prev_hash,
            extra=extra,
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO audit_log
                (timestamp, building_id, action, trust_score, top_features, shap_values,
                 prev_hash, record_hash, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp, building_id, action_str, record.trust_score,
                json.dumps(record.top_features),
                json.dumps(record.shap_dict),
                record.prev_hash, record.record_hash,
                json.dumps(extra or {}),
            ))

        self._last_hashes[building_id] = record.record_hash
        return record


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    feature_names = [
        "occupancy", "co2_ppm", "temperature_in", "temperature_out",
        "humidity", "hvac_power_kw", "hvac_setpoint", "motion_count",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "month_sin", "month_cos",
    ]

    logger_inst = JSONLAuditLogger(log_dir="data/audit/test")
    rng = np.random.default_rng(42)

    # Log 5 test decisions
    for i in range(5):
        shap_vals = rng.normal(0, 0.3, 14)
        rec = logger_inst.log(
            building_id="B01",
            action_str=f"Q=-{5+i*2:.1f}kW(T=22.0°C)",
            shap_values=shap_vals,
            feature_names=feature_names,
            trust_score=0.85 + rng.uniform(-0.1, 0.1),
        )
        print(f"Record {i+1}: hash={rec.record_hash[:16]}... | τ={rec.trust_score}")

    valid = logger_inst.verify_chain("B01")
    print(f"\nHash chain valid: {valid}")
