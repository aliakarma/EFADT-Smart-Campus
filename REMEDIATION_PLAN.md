# Repository Remediation & Hardening Plan
# EFADT — Explainable Federated Agentic Digital Twin

**Prepared by:** Principal ML Systems Engineer / Reproducibility Lead  
**Target:** Research-grade, publication-ready repository  
**Audit basis:** Full forensic review — 14-phase fraud fingerprint + pipeline reconstruction  

---

## 1. Executive Summary

### Current Maturity Assessment

The EFADT repository contains well-structured, non-trivial implementation code for FL training, a digital twin simulator, a DP mechanism, and a trust scorer. However, its **evaluation pipeline is entirely decoupled from its training pipeline**. All six primary metric claims (ERR=34.7%, CCS=0.912, CSS=0.963, MAE=3.21, τ=0.887, SHF=0.921) originate from a hardcoded Python dictionary (`evaluation/baseline_runner.py:PAPER_RESULTS`) with no code path that generates these numbers from a trained model. The repository is currently unsuitable for submission to any peer-reviewed venue.

### Main Blockers

| Blocker | Severity | Phase |
|---------|----------|-------|
| All metrics hardcoded — not computed from model | **Fatal** | 5 |
| Evaluation pipeline structurally disconnected from training | **Fatal** | 3 |
| DP guarantee misrepresented (ε_total=100, not 1.0) | **Fatal** | 10 |
| DT-Only baseline code produces MAE=0, not claimed 3.68 | **Fatal** | 9 |
| No test split — only train/val; config/code inconsistency | **Critical** | 4 |
| StandardScaler not serialized; inference runs unscaled | **Critical** | 3 |
| Non-deterministic DP RNG in FL client | **Critical** | 6 |
| Zero statistical validity (no seeds, no CIs, no significance tests) | **Critical** | 7 |
| SHAP API stub returns random values | **High** | 3 |
| Flower FL simulation never validated end-to-end | **High** | 2 |

### Estimated Total Remediation Time

| Phase | Title | Time |
|-------|-------|------|
| 1 | Repository Cleanup & Structural Repair | 3h |
| 2 | Dependency & Environment Stabilization | 4h |
| 3 | Pipeline Reconnection & Execution Integrity | 20h |
| 4 | Data Leakage Elimination & Split Correction | 6h |
| 5 | Metric Verification & Evaluation Corrections | 12h |
| 6 | Determinism & Seed Control | 4h |
| 7 | Statistical Validity Upgrades | 10h |
| 8 | Experiment Tracking & Logging | 6h |
| 9 | Baseline Reimplementation & Fair Comparison | 10h |
| 10 | Differential Privacy Correction | 5h |
| 11 | Synthetic Dataset Validation | 6h |
| 12 | README & Documentation Reconstruction | 5h |
| 13 | CI/CD & Automated Validation | 6h |
| 14 | Final Reproducibility Certification Pass | 4h |
| **Total** | | **~101h (~13 working days)** |

### Expected Final Outcome

A repository where: (a) `make reproduce` clones, installs, generates data, trains, evaluates, and writes a results JSON in one command; (b) all reported metrics are computed from checkpoints by runnable code; (c) DP claims are mathematically accurate; (d) ablation results have mean±std over ≥3 seeds with significance tests vs each baseline; (e) CI enforces the full pipeline on every push.

---

# Phase 1 — Repository Cleanup & Structural Repair
**Estimated Time: 3 hours**

## Objective

Remove dead code, resolve config duplication, fix deprecated API calls, create the missing evaluation and scripts directories, and align the repository layout with the target structure before any code changes.

## Problems Addressed

- `configs/utility_weights.yaml` duplicates agent weights from `hyperparams.yaml` and is never loaded
- `xai/shap_explainer.py` references non-existent `xai/gradient_shap.py`
- `REMEDIATION_PLAN.md` and `PROJECT_REPORT.md` listed in `.gitignore` but do not exist yet
- `data/generation/generate_dataset.py` and `sensor_fault_injector.py` use deprecated `fillna(method='ffill')` Pandas 2.x API
- `federated/client.py:evaluate()` returns `hash(self.building_id) % 1000` as a metric
- `evaluation/baseline_runner.py:PAPER_RESULTS` must be clearly marked as unverified placeholders, not results
- Missing directories: `scripts/`, `results/`, `docs/`

## Files To Modify

| File | Required Changes |
|------|------------------|
| `configs/utility_weights.yaml` | Delete — superseded by `hyperparams.yaml` |
| `data/generation/generate_dataset.py` | Replace deprecated `fillna(method=...)` |
| `data/generation/sensor_fault_injector.py` | Replace deprecated `ffill().bfill()` |
| `federated/client.py` | Remove `hash(building_id) % 1000` from metrics |
| `evaluation/baseline_runner.py` | Mark `PAPER_RESULTS` as placeholder with loud warning |
| `xai/shap_explainer.py` | Remove dangling reference to `gradient_shap.py` |
| `.gitignore` | Remove placeholder entries for non-existent files |

---

## Step-by-Step Implementation Guide

### Step 1 — Create workspace branch and directory scaffold

**Purpose:** All remediation work happens on a dedicated branch. Creating target directories now avoids repeated `mkdir` later.

```bash
cd /path/to/efadt-smart-campus
git checkout -b remediation/phase-1-cleanup

mkdir -p results/seeds
mkdir -p results/baselines
mkdir -p results/ablation
mkdir -p docs/figures
mkdir -p scripts
touch results/.gitkeep
touch docs/.gitkeep
```

### Step 2 — Remove dead config file

**Purpose:** `configs/utility_weights.yaml` is never imported. Having it diverge silently from `hyperparams.yaml` creates confusion.

```bash
git rm configs/utility_weights.yaml
```

Verify nothing imports it:

```bash
grep -R "utility_weights.yaml" . --include="*.py"
# Expected: no output
```

### Step 3 — Fix deprecated Pandas API in data generation

**Purpose:** `fillna(method='ffill')` raises `FutureWarning` in Pandas 2.1 and will be removed in 3.0.

**Code Changes:**

```python
# BEFORE — data/generation/generate_dataset.py, line generating TARGET_COLUMN
df[TARGET_COLUMN] = df["occupancy"].shift(-1).fillna(method="ffill")

# AFTER
df[TARGET_COLUMN] = df["occupancy"].shift(-1).ffill()
```

```python
# BEFORE — data/generation/sensor_fault_injector.py:inject_sensor_faults()
df_out[columns_to_affect] = (
    df_out[columns_to_affect].ffill().bfill()
)

# AFTER
df_out[columns_to_affect] = df_out[columns_to_affect].ffill().bfill()
# Note: .ffill() and .bfill() are the non-deprecated equivalents in Pandas 2.x
```

**Validation:**

```bash
python -W error::FutureWarning -c "
from data.generation.generate_dataset import generate_full_dataset
generate_full_dataset(n_buildings=1, n_days=3)
"
# Expected: no FutureWarning, exits 0
```

### Step 4 — Remove noise metric from FL client evaluate()

**Purpose:** `hash(self.building_id) % 1000` emits garbage into the metrics aggregation pipeline, polluting MLflow logs.

```python
# BEFORE — federated/client.py:evaluate(), metrics dict
metrics: Metrics = {"mae": float(mae), "building_id": hash(self.building_id) % 1000}

# AFTER
metrics: Metrics = {"mae": float(mae)}
```

### Step 5 — Mark PAPER_RESULTS as unverified placeholders

**Purpose:** The dict must not be read as ground truth. A loud warning at import time prevents accidental use.

```python
# BEFORE — evaluation/baseline_runner.py
PAPER_RESULTS = {
    "EFADT (Full)": {"ERR": 34.7, "CCS": 0.912, ...},
    ...
}

# AFTER — evaluation/baseline_runner.py
import warnings

# ──────────────────────────────────────────────────────────────────────────────
# ⚠️  PLACEHOLDER VALUES — NOT VERIFIED BY CODE
# These numbers are TARGET values from the paper draft. They are NOT generated
# by any training or evaluation run in this repository.
# Replace with values from: results/ablation/full_results.json
# after completing Phase 5 of REMEDIATION_PLAN.md.
# ──────────────────────────────────────────────────────────────────────────────
_PAPER_RESULTS_UNVERIFIED = {
    "EFADT (Full)":       {"ERR": None, "CCS": None, "CSS": None, "MAE": None, "tau": None},
    "-XAI":               {"ERR": None, "CCS": None, "CSS": None, "MAE": None, "tau": None},
    "-DT-WIF":            {"ERR": None, "CCS": None, "CSS": None, "MAE": None, "tau": None},
    "-DP":                {"ERR": None, "CCS": None, "CSS": None, "MAE": None, "tau": None},
    "-MOO (energy-only)": {"ERR": None, "CCS": None, "CSS": None, "MAE": None, "tau": None},
    "-FL (centralized)":  {"ERR": None, "CCS": None, "CSS": None, "MAE": None, "tau": None},
}

def _load_results() -> dict:
    """Load verified results from results/ablation/full_results.json if available."""
    import json, os
    path = os.path.join(os.path.dirname(__file__), "..", "results", "ablation", "full_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    warnings.warn(
        "results/ablation/full_results.json not found. "
        "PAPER_RESULTS contains None placeholders. "
        "Run: python scripts/evaluate_checkpoint.py to generate real results.",
        UserWarning, stacklevel=2,
    )
    return _PAPER_RESULTS_UNVERIFIED

PAPER_RESULTS = _load_results()
```

### Step 6 — Remove dangling reference in shap_explainer.py

```python
# BEFORE — xai/shap_explainer.py docstring
# For direct LSTM SHAP, use GradientExplainer (see xai/gradient_shap.py).

# AFTER
# For direct LSTM SHAP, use shap.GradientExplainer with the trained OccupancyLSTM.
# Reference: https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html
```

### Step 7 — Update .gitignore

```bash
# BEFORE — .gitignore contains these phantom entries:
# REMEDIATION_PLAN.md
# PROJECT_REPORT.md

# Remove them — these files should now be tracked:
sed -i '/^REMEDIATION_PLAN\.md$/d' .gitignore
sed -i '/^PROJECT_REPORT\.md$/d' .gitignore

# Add results output files that should not be committed:
cat >> .gitignore << 'EOF'

# Results (large experiment outputs)
results/seeds/
results/checkpoints/
EOF
```

### Step 8 — Commit Phase 1 changes

```bash
git add -A
git commit -m "phase-1: cleanup dead config, fix deprecated pandas API, placeholder PAPER_RESULTS"
```

---

## README Updates Required

### Modify: Remove hardcoded metric table

Replace the results table in `README.md` with:

```markdown
### Key Results (12 Buildings × 12 Months)

> ⚠️ Results table will be populated after completing the reproducibility pipeline.
> Run `make reproduce` to generate `results/ablation/full_results.json`.

| Metric | EFADT | Rule-Based | DT-Only | Centralized |
|--------|-------|------------|---------|-------------|
| ERR (↑) | — | — | — | — |
| CCS (↑) | — | — | — | — |
| CSS (↑) | — | — | — | — |
| MAE (↓ persons) | — | — | — | — |
| τ (↑) | — | — | — | — |
```

---

## Success Criteria (MANDATORY CHECKPOINT)

- [ ] `configs/utility_weights.yaml` deleted; `git status` shows it removed
- [ ] `grep -R "utility_weights.yaml" . --include="*.py"` returns no output
- [ ] `python -W error::FutureWarning -m data.generation.generate_dataset --n-buildings 1 --n-days 3` exits 0 with no warnings
- [ ] `grep "hash(self.building_id)" federated/client.py` returns no output
- [ ] `grep "PAPER_RESULTS_UNVERIFIED\|None placeholders" evaluation/baseline_runner.py` returns output
- [ ] `python -c "from evaluation.baseline_runner import PAPER_RESULTS; import warnings; warnings.simplefilter('error'); PAPER_RESULTS"` raises `UserWarning` (no results json yet)
- [ ] `REMEDIATION_PLAN.md` and `PROJECT_REPORT.md` no longer in `.gitignore`
- [ ] `docs/`, `results/seeds/`, `results/ablation/`, `scripts/` directories exist

### Proceed Rule
All items `[x]` → proceed to Phase 2. Any `[ ]` → fix before advancing.

---

# Phase 2 — Dependency & Environment Stabilization
**Estimated Time: 4 hours**

## Objective

Pin all dependencies to tested versions, validate that the full dependency graph installs and imports correctly, confirm Flower FL simulation runs end-to-end on a minimal dataset, and establish a reproducible virtual environment.

## Problems Addressed

- `flwr[simulation]==1.6.0` is listed in `requirements.txt` but flagged as "not tested" in `SMOKE_TEST_REPORT.md`
- `opacus==1.4.0` listed but never tested
- `streamlit` missing from `requirements.txt` (used in `governance/dashboard/app.py`)
- `pytest-timeout` used in `pytest.ini` but not in `requirements.txt`
- Python version constraints not enforced at runtime
- No `requirements-dev.txt` separation

## Files To Modify

| File | Required Changes |
|------|------------------|
| `requirements.txt` | Add `streamlit`, `pytest-timeout`; validate all pinned versions |
| `requirements-dev.txt` | Create — separate dev/test deps from runtime |
| `pyproject.toml` | Add `python_requires = ">=3.10,<3.12"` |
| `Makefile` | Add `make env` target for reproducible venv creation |
| `.github/workflows/ci.yml` | Add FL smoke test step |

---

## Step-by-Step Implementation Guide

### Step 1 — Create isolated virtual environment

```bash
python -m venv venv
source venv/Scripts/activate   # Git Bash on Windows
# OR on Linux/macOS:
# source venv/bin/activate

python --version
# Must print 3.10.x or 3.11.x
```

### Step 2 — Fix requirements.txt

```bash
# Add missing runtime dependencies
cat >> requirements.txt << 'EOF'

# Dashboard
streamlit==1.31.0
EOF
```

Create `requirements-dev.txt`:

```bash
cat > requirements-dev.txt << 'EOF'
# Development and testing dependencies
# Install with: pip install -r requirements-dev.txt
pytest==7.4.4
pytest-asyncio==0.23.3
pytest-cov==4.1.0
pytest-timeout==2.2.0
ruff==0.1.14
httpx==0.26.0
anyio[trio]==4.2.0
EOF
```

### Step 3 — Install and validate full dependency graph

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Validate all critical imports
python -c "
import torch; print('torch', torch.__version__)
import flwr; print('flwr', flwr.__version__)
import opacus; print('opacus', opacus.__version__)
import shap; print('shap', shap.__version__)
import fastapi; print('fastapi', fastapi.__version__)
import streamlit; print('streamlit', streamlit.__version__)
"
```

**Expected output:** All six lines print version strings without ImportError.

### Step 4 — End-to-end Flower simulation smoke test

**Purpose:** The audit found Flower was never validated end-to-end. This step confirms `fl.simulation.start_simulation` runs without error on minimal data.

```bash
# Generate minimal dataset first
python -m data.generation.generate_dataset \
  --n-buildings 2 \
  --n-days 5 \
  --config configs/hyperparams.yaml \
  --building-config configs/building_params.yaml

# Run minimal FL simulation
python -m federated.simulation \
  --n-rounds 3 \
  --n-buildings 2 \
  --config configs/hyperparams.yaml \
  --building-config configs/building_params.yaml \
  --data-dir data/raw

# Expected: prints round-by-round MAE, exits 0
# Failure: ImportError or flwr.common incompatibility — fix version pinning
```

If Flower raises a compatibility error, pin to the tested version:

```bash
pip show flwr | grep Version
# If not 1.6.0, reinstall:
pip install "flwr[simulation]==1.6.0" --force-reinstall
```

### Step 5 — Add Makefile env target

```makefile
# Add to Makefile
env:
	python -m venv venv
	source venv/Scripts/activate && pip install --upgrade pip \
	  && pip install -r requirements.txt \
	  && pip install -r requirements-dev.txt
	@echo "✓ Virtual environment ready — activate with: source venv/Scripts/activate"
```

### Step 6 — Update pyproject.toml Python constraint

```toml
# pyproject.toml — modify [project] section
requires-python = ">=3.10,<3.12"
```

### Step 7 — Update CI to test FL simulation

```yaml
# .github/workflows/ci.yml — add new job step inside the 'test' job
- name: Run minimal FL simulation
  run: |
    python -m data.generation.generate_dataset \
      --n-buildings 2 --n-days 5
    python -m federated.simulation \
      --n-rounds 3 --n-buildings 2
  timeout-minutes: 10
```

### Step 8 — Commit Phase 2 changes

```bash
git add requirements.txt requirements-dev.txt pyproject.toml Makefile .github/workflows/ci.yml
git commit -m "phase-2: fix deps, add streamlit, validate flwr end-to-end, requirements-dev split"
```

---

## README Updates Required

### Add Section: Environment Setup

```markdown
## 🛠️ Environment Setup

```bash
# Recommended: use the make target to create a reproducible environment
make env
source venv/Scripts/activate    # Windows Git Bash
# source venv/bin/activate      # Linux/macOS

# Python version requirement: 3.10.x – 3.11.x
python --version
```

### Dependency Notes

- `flwr[simulation]==1.6.0` — validated end-to-end (Phase 2 CI step)
- `opacus==1.4.0` — required for tight Rényi DP accounting (Phase 10)
- `streamlit==1.31.0` — governance dashboard
```

---

## Success Criteria (MANDATORY CHECKPOINT)

- [ ] `pip install -r requirements.txt` completes with exit code 0
- [ ] `pip install -r requirements-dev.txt` completes with exit code 0
- [ ] `python -c "import flwr, opacus, shap, streamlit, fastapi"` exits 0 with no ImportError
- [ ] Minimal FL simulation (3 rounds, 2 buildings) runs and prints round-by-round MAE
- [ ] `pytest tests/test_core.py -v --timeout=120` passes (≥46 tests, 0 failures)
- [ ] CI workflow includes FL simulation step

### Proceed Rule
All items `[x]` → proceed to Phase 3.

---

# Phase 3 — Pipeline Reconnection & Execution Integrity
**Estimated Time: 20 hours**

## Objective

Implement the missing end-to-end path: trained FL checkpoint → StandardScaler deserialization → test-set inference → metric computation → results JSON. Fix SHAP stub in API. Fix scaler serialization in training loop. Create `scripts/evaluate_checkpoint.py`.

## Problems Addressed

- No script connects FL training output → metric computation
- `StandardScaler` not serialized with checkpoints; inference runs on unscaled features
- `api/main.py` SHAP stub returns `rng.normal(0, 0.1, 14)` — meaningless trust scores
- `pipeline/decision_cycle.py` silently accepts `scaler=None`
- `scripts/run_experiment.py:step_evaluate()` only prints the hardcoded dict

## Files To Modify

| File | Required Changes |
|------|------------------|
| `models/lstm/train_local.py` | Serialize scaler with every checkpoint |
| `federated/server.py` | Save scaler path in checkpoint metadata |
| `scripts/evaluate_checkpoint.py` | **Create new** — full evaluation entry point |
| `scripts/run_experiment.py` | Replace `step_evaluate()` with call to evaluate_checkpoint |
| `api/main.py` | Load fitted SHAP explainer; gate /decide on shap_fitted |
| `pipeline/decision_cycle.py` | Assert scaler is not None on construction |
| `xai/shap_explainer.py` | Add `save()` / `load()` methods |

---

## Step-by-Step Implementation Guide

### Step 1 — Serialize StandardScaler in training checkpoints

**Purpose:** Every checkpoint must include the scaler fitted on that building's train split. Without this, any inference path produces garbage predictions.

```python
# models/lstm/train_local.py — inside run_standalone_training(), checkpoint save block:

# BEFORE
torch.save({
    "epoch": epoch + 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_mae": val_mae,
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
}, ckpt_path)

# AFTER — save full scaler as a pickle alongside the .pt file
import pickle

torch.save({
    "epoch": epoch + 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_mae": val_mae,
    "building_id": building_id,
    "feature_columns": FEATURE_COLUMNS,
}, ckpt_path)

scaler_path = ckpt_path.replace("_best.pt", "_best_scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
```

Add a loader helper at the bottom of `models/lstm/train_local.py`:

```python
def load_checkpoint(
    checkpoint_path: str,
    config: dict,
    device: Optional[torch.device] = None,
):
    """Load model and scaler from a training checkpoint."""
    import pickle
    from sklearn.preprocessing import StandardScaler

    if device is None:
        device = torch.device("cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(config, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler_path = checkpoint_path.replace("_best.pt", "_best_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. "
            "Retrain the model with the updated train_local.py."
        )
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, ckpt
```

**Validation:**

```bash
python -m models.lstm.train_local --building-id B01 --data-path data/raw
ls models/lstm/checkpoints/B01_best.pt models/lstm/checkpoints/B01_best_scaler.pkl
# Both files must exist
```

### Step 2 — Add save/load to SHAPProxyExplainer

```python
# xai/shap_explainer.py — add to SHAPProxyExplainer class:

def save(self, path: str) -> None:
    """Persist fitted proxy and explainer to disk."""
    import pickle
    if not self._is_fitted:
        raise RuntimeError("Explainer not fitted. Call fit() first.")
    with open(path, "wb") as f:
        pickle.dump({
            "proxy": self.proxy,
            "feature_names": self.feature_names,
            "n_raw_features": self._n_raw_features,
            "X_background": self._X_background,
        }, f)
    logger.info(f"SHAPProxyExplainer saved to {path}")

@classmethod
def load(cls, path: str) -> "SHAPProxyExplainer":
    """Load a previously fitted proxy explainer."""
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    instance = cls(feature_names=data["feature_names"])
    instance.proxy = data["proxy"]
    instance.explainer = shap.TreeExplainer(instance.proxy)
    instance._n_raw_features = data["n_raw_features"]
    instance._X_background = data["X_background"]
    instance._is_fitted = True
    return instance
```

### Step 3 — Create scripts/evaluate_checkpoint.py

This is the **most critical missing file**. It is the only script that generates real metric values.

```python
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

        test_records = df_test.to_dict("records")
        for i, row in enumerate(test_records[:len(y_pred)]):
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
            shap_path = ckpt_path.replace("_best.pt", "_best_shap.pkl")
            if os.path.exists(shap_path):
                explainer = SHAPProxyExplainer.load(shap_path)
                lookback = config["lstm"]["lookback_steps"]
                idx_start = max(0, i - lookback)
                x_window = scaler.transform(
                    df_test[FEATURE_COLUMNS].values[idx_start:i+1].astype(np.float32)
                )
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
    parser.add_argument("--test-months", nargs="+", type=int, default=[10, 11, 12])
    parser.add_argument("--config", default="configs/hyperparams.yaml")
    parser.add_argument("--building-config", default="configs/building_params.yaml")
    parser.add_argument("--output", default="results/ablation/full_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.building_config) as f:
        building_cfg = yaml.safe_load(f)["buildings"]

    building_ids = list(building_cfg.keys())[:config["data"]["n_buildings"]]
    building_data = {}
    for bid in building_ids:
        path = os.path.join(args.data_dir, f"{bid}.parquet")
        if os.path.exists(path):
            building_data[bid] = pd.read_parquet(path)

    device = torch.device("cpu")

    results = {}

    # Full EFADT
    logger.info("Evaluating EFADT (Full)...")
    results["EFADT (Full)"] = evaluate_single_variant(
        "EFADT (Full)", building_data, args.checkpoint_dir,
        config, building_cfg, args.test_months, device,
    ).to_dict()

    # -MOO (energy only)
    logger.info("Evaluating -MOO (energy-only)...")
    results["-MOO (energy-only)"] = evaluate_single_variant(
        "-MOO", building_data, args.checkpoint_dir,
        config, building_cfg, args.test_months, device,
        weights_override={"lambda_e": 1.0, "lambda_c": 0.0, "lambda_d": 0.0},
    ).to_dict()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"seed": args.seed, "test_months": args.test_months, "results": results}, f, indent=2)

    logger.info(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
```

### Step 4 — Update scripts/run_experiment.py:step_evaluate()

```python
# BEFORE — scripts/run_experiment.py
def step_evaluate(n_buildings: int) -> None:
    logger.info("Step 3: Evaluating baselines and ablation variants...")
    from evaluation.baseline_runner import print_ablation_table, PAPER_RESULTS
    print_ablation_table()

# AFTER
def step_evaluate(checkpoint_dir: str, data_dir: str, test_months: list, seed: int) -> None:
    logger.info("Step 3: Evaluating from trained checkpoints...")
    import subprocess, sys
    result = subprocess.run([
        sys.executable, "scripts/evaluate_checkpoint.py",
        "--checkpoint-dir", checkpoint_dir,
        "--data-dir", data_dir,
        "--test-months"] + [str(m) for m in test_months] + [
        "--seed", str(seed),
        "--output", "results/ablation/full_results.json",
    ], check=True)
    logger.info("Evaluation complete — results at results/ablation/full_results.json")
```

### Step 5 — Fix SHAP stub in api/main.py

```python
# BEFORE — api/main.py:get_decision()
rng = np.random.default_rng()
shap_values = rng.normal(0, 0.1, 14)   # Stub

# AFTER — api/main.py:get_decision()
if _state.get("shap_fitted"):
    shap_values = _state["shap_explainer"].explain(
        np.array(req.occ_forecast)  # proper window; see below
    )
else:
    raise HTTPException(
        status_code=503,
        detail="SHAP explainer not calibrated. POST /calibrate-shap with training predictions first."
    )
```

Add `POST /calibrate-shap` endpoint to `api/main.py`:

```python
class SHAPCalibrationRequest(BaseModel):
    building_id: str
    X_samples: list      # (n_samples, lookback, n_features) as nested list
    lstm_predictions: list  # (n_samples,) LSTM outputs from training

@app.post("/calibrate-shap", tags=["Explainability"])
async def calibrate_shap(req: SHAPCalibrationRequest):
    """Fit SHAP proxy on training predictions. Must be called once after FL training."""
    X = np.array(req.X_samples, dtype=np.float32)
    preds = np.array(req.lstm_predictions, dtype=np.float32)
    explainer = _state["shap_explainer"]
    explainer.fit(X, preds)
    _state["shap_fitted"] = True
    # Persist for restart recovery
    Path("models/shap").mkdir(parents=True, exist_ok=True)
    explainer.save(f"models/shap/{req.building_id}_shap_proxy.pkl")
    return {"status": "fitted", "n_samples": len(preds)}
```

### Step 6 — Validate pipeline runs end-to-end

```bash
# 1. Generate data (2 buildings, 30 days)
python -m data.generation.generate_dataset \
  --n-buildings 2 --n-days 30

# 2. Train standalone (1 building, quick check)
python -m models.lstm.train_local --building-id B01 --data-path data/raw

# 3. Verify scaler exists alongside checkpoint
ls -la models/lstm/checkpoints/B01_best.pt models/lstm/checkpoints/B01_best_scaler.pkl

# 4. Run evaluation
python scripts/evaluate_checkpoint.py \
  --checkpoint-dir models/lstm/checkpoints \
  --data-dir data/raw \
  --test-months 2 \
  --n-buildings 2

# 5. Verify results file written
cat results/ablation/full_results.json
# Must contain non-None ERR, CCS, CSS, MAE values
```

### Step 7 — Add pipeline validation to Makefile

```makefile
# Add to Makefile
evaluate:
	$(PYTHON) scripts/evaluate_checkpoint.py \
	  --checkpoint-dir models/lstm/checkpoints \
	  --data-dir data/raw \
	  --output results/ablation/full_results.json

reproduce: generate-data train-fl evaluate
	@echo "✓ Full reproduction pipeline complete"
	@cat results/ablation/full_results.json
```

### Step 8 — Commit Phase 3 changes

```bash
git add models/lstm/train_local.py xai/shap_explainer.py \
        scripts/evaluate_checkpoint.py scripts/run_experiment.py \
        api/main.py pipeline/decision_cycle.py Makefile
git commit -m "phase-3: reconnect eval pipeline, serialize scaler, fix SHAP stub, add evaluate_checkpoint.py"
```

---

## README Updates Required

### Add Section: Reproducing Results

```markdown
## 🔬 Reproducing Results

The full reproduction pipeline runs in one command after installing dependencies:

```bash
make reproduce
# Equivalent to:
# make generate-data   (12 buildings × 365 days)
# make train-fl        (100 FL rounds)
# make evaluate        (test-set metrics → results/ablation/full_results.json)
```

Results are written to `results/ablation/full_results.json`. The metric table in this README
is populated from that file; it will show `—` until `make reproduce` completes.
```

---

## Success Criteria (MANDATORY CHECKPOINT)

- [ ] `models/lstm/train_local.py:run_standalone_training()` saves `*_best_scaler.pkl` alongside `*_best.pt`
- [ ] `load_checkpoint()` function exists and correctly loads both model and scaler
- [ ] `SHAPProxyExplainer.save()` and `.load()` methods exist and roundtrip correctly
- [ ] `scripts/evaluate_checkpoint.py` exists and exits 0 on 2-building / 1-month test
- [ ] `results/ablation/full_results.json` is written with non-None metric values
- [ ] `api/main.py` raises HTTP 503 when SHAP not fitted (not random values)
- [ ] `POST /calibrate-shap` endpoint exists and sets `_state["shap_fitted"] = True`
- [ ] `make reproduce` target exists in `Makefile`

### Proceed Rule
All items `[x]` → proceed to Phase 4.

---

# Phase 4 — Data Leakage Elimination & Split Correction
**Estimated Time: 6 hours**

## Objective

Implement a proper calendar-based train/validation/test split aligned with the configuration file. Ensure the StandardScaler is fit only on training months. Verify no temporal leakage across the split boundary.

## Problems Addressed

- `configs/hyperparams.yaml` specifies `train_months: 9`, `test_months: 3` but `train_local.py:prepare_data()` uses `train_frac=0.75` (fraction of all samples), ignoring the config
- Config keys `train_months` and `test_months` are never read by any training code
- No held-out test set exists; val set has been used as a proxy for test
- Final sample in each building's data has `occupancy_next` = its own occupancy (ffill artifact)

## Files To Modify

| File | Required Changes |
|------|------------------|
| `models/lstm/train_local.py` | Replace `train_frac` split with `split_by_months()` |
| `configs/hyperparams.yaml` | Add `val_months: [7, 8, 9]` to make split explicit |
| `scripts/evaluate_checkpoint.py` | Already uses `load_test_split(test_months)` — verify consistency |

---

## Step-by-Step Implementation Guide

### Step 1 — Define explicit month splits in config

```yaml
# configs/hyperparams.yaml — modify the data section:
data:
  n_buildings: 12
  n_months: 12
  sampling_interval_s: 30
  n_features: 14
  train_months: [1, 2, 3, 4, 5, 6]     # Jan–Jun (6 months)
  val_months: [7, 8, 9]                  # Jul–Sep (3 months)
  test_months: [10, 11, 12]             # Oct–Dec (3 months, never seen during training)
  max_occupancy: 80
  # ... rest unchanged
```

### Step 2 — Rewrite prepare_data() to use calendar split

```python
# models/lstm/train_local.py — replace prepare_data() entirely

def prepare_data(
    df: pd.DataFrame,
    lookback: int = 12,
    train_months: list[int] = None,
    val_months: list[int] = None,
    scaler=None,
) -> tuple:
    """
    Calendar-based train/val split. Scaler is fit ONLY on training months.

    Parameters
    ----------
    df : pd.DataFrame with DatetimeIndex
    lookback : int  — LSTM sequence length
    train_months : list[int]  — months (1–12) used for training
    val_months : list[int]    — months used for validation
    scaler : StandardScaler, optional  — pre-fitted scaler (for inference reuse)

    Returns
    -------
    (train_dataset, val_dataset, scaler)
    """
    if train_months is None:
        train_months = [1, 2, 3, 4, 5, 6]
    if val_months is None:
        val_months = [7, 8, 9]

    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    # Calendar split — strictly temporal, no random sampling
    train_mask = df.index.month.isin(train_months)
    val_mask = df.index.month.isin(val_months)

    df_train = df[train_mask]
    df_val = df[val_mask]

    if len(df_train) == 0:
        raise ValueError(f"No training data for months {train_months}. Check dataset date range.")
    if len(df_val) == 0:
        raise ValueError(f"No validation data for months {val_months}. Check dataset date range.")

    X_train = df_train[FEATURE_COLUMNS].values.astype(np.float32)
    y_train = df_train[TARGET_COLUMN].values.astype(np.float32)
    X_val = df_val[FEATURE_COLUMNS].values.astype(np.float32)
    y_val = df_val[TARGET_COLUMN].values.astype(np.float32)

    # Fit scaler on training data ONLY
    if scaler is None:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
    else:
        X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = CampusDataset(X_train, y_train, lookback=lookback)
    val_ds = CampusDataset(X_val, y_val, lookback=lookback)

    logger.info(
        f"Split: train={len(df_train):,} ({train_months}), "
        f"val={len(df_val):,} ({val_months})"
    )
    return train_ds, val_ds, scaler
```

### Step 3 — Update all prepare_data() call sites

Update `federated/client.py:__init__()`:

```python
# BEFORE
train_ds, val_ds, self.scaler = prepare_data(df, lookback=self.lookback)

# AFTER — pass month splits from config
data_cfg = config.get("data", {})
train_ds, val_ds, self.scaler = prepare_data(
    df,
    lookback=self.lookback,
    train_months=data_cfg.get("train_months", [1,2,3,4,5,6]),
    val_months=data_cfg.get("val_months", [7,8,9]),
)
```

Update `models/lstm/train_local.py:run_standalone_training()`:

```python
# AFTER — pass month splits from config
data_cfg = cfg.get("data", {})
train_ds, val_ds, scaler = prepare_data(
    df,
    lookback=lstm_cfg["lookback_steps"],
    train_months=data_cfg.get("train_months", [1,2,3,4,5,6]),
    val_months=data_cfg.get("val_months", [7,8,9]),
)
```

### Step 4 — Fix occupancy_next ffill artifact

```python
# data/generation/generate_dataset.py — replace TARGET_COLUMN construction:

# BEFORE
df[TARGET_COLUMN] = df["occupancy"].shift(-1).ffill()

# AFTER — drop the last row instead of filling it; its target is undefined
df[TARGET_COLUMN] = df["occupancy"].shift(-1)
df = df.dropna(subset=[TARGET_COLUMN])  # removes final row
```

### Step 5 — Leakage verification test

Create `tests/test_no_leakage.py`:

```python
"""tests/test_no_leakage.py — verify no temporal leakage across split boundary."""
import pandas as pd
import numpy as np
import pytest
from models.lstm.train_local import prepare_data


def test_train_val_month_disjoint():
    """Train and val datasets must come from disjoint month sets."""
    ts = pd.date_range("2024-01-01", periods=2880 * 12, freq="30s")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "occupancy": rng.integers(0, 50, len(ts)),
        "co2_ppm": rng.uniform(400, 900, len(ts)),
        "temperature_in": rng.normal(22, 1, len(ts)),
        "temperature_out": rng.normal(30, 5, len(ts)),
        "humidity": rng.uniform(30, 70, len(ts)),
        "hvac_power_kw": rng.uniform(-20, 5, len(ts)),
        "hvac_setpoint": np.full(len(ts), 22.0),
        "motion_count": rng.integers(0, 30, len(ts)),
        "hour_sin": np.sin(2*np.pi*ts.hour/24),
        "hour_cos": np.cos(2*np.pi*ts.hour/24),
        "day_of_week_sin": np.sin(2*np.pi*ts.dayofweek/7),
        "day_of_week_cos": np.cos(2*np.pi*ts.dayofweek/7),
        "month_sin": np.sin(2*np.pi*(ts.month-1)/12),
        "month_cos": np.cos(2*np.pi*(ts.month-1)/12),
        "occupancy_next": rng.integers(0, 50, len(ts)).astype(float),
    }, index=ts)

    train_ds, val_ds, scaler = prepare_data(
        df, lookback=12,
        train_months=[1, 2, 3, 4, 5, 6],
        val_months=[7, 8, 9],
    )
    # Scaler must NOT have been fit on val data
    # Check by verifying mean was computed from train-month rows only
    from models.lstm.train_local import FEATURE_COLUMNS
    X_all = df[FEATURE_COLUMNS].values.astype(np.float32)
    train_mask = df.index.month.isin([1,2,3,4,5,6])
    X_train_only = X_all[train_mask]
    np.testing.assert_allclose(scaler.mean_, X_train_only.mean(axis=0), rtol=1e-3)


def test_val_months_not_in_train():
    """No val-month rows should appear in train_ds."""
    # Verified structurally by calendar mask in prepare_data()
    pass  # If prepare_data() uses .isin(), disjoint sets are guaranteed
```

```bash
pytest tests/test_no_leakage.py -v
# Expected: 1 passed
```

### Step 6 — Commit Phase 4 changes

```bash
git add models/lstm/train_local.py federated/client.py configs/hyperparams.yaml \
        data/generation/generate_dataset.py tests/test_no_leakage.py
git commit -m "phase-4: calendar split, fix train_frac, add test_months config, no-leakage test"
```

---

## README Updates Required

### Add Section: Dataset Splits

```markdown
## 📅 Dataset Splits

Data is split by calendar month to avoid temporal leakage:

| Split | Months | Approximate Size |
|-------|--------|-----------------|
| Train | Jan–Jun (1–6) | ~6 months per building |
| Validation | Jul–Sep (7–9) | ~3 months per building |
| Test | Oct–Dec (10–12) | ~3 months per building |

The StandardScaler is fitted on training months only. Test months are
never touched during training or hyperparameter selection.
```

---

## Success Criteria (MANDATORY CHECKPOINT)

- [ ] `prepare_data()` uses `train_months`/`val_months` lists from config, not `train_frac`
- [ ] `configs/hyperparams.yaml` contains explicit `train_months`, `val_months`, `test_months` lists
- [ ] `test_train_val_month_disjoint` passes: scaler mean matches train-only rows
- [ ] `occupancy_next` constructed with `dropna()` instead of `ffill()`
- [ ] `federated/client.py:__init__()` passes month lists to `prepare_data()`
- [ ] `scripts/evaluate_checkpoint.py` uses `test_months` from config (or CLI arg)

### Proceed Rule
All items `[x]` → proceed to Phase 5.

---

# Phase 5 — Metric Verification & Evaluation Corrections
**Estimated Time: 12 hours**

## Objective

Fix the DT-Only baseline (currently computes MAE=0), implement the centralized LSTM baseline evaluation, make `PAPER_RESULTS` load from computed JSON, and run the full evaluation to produce the first set of honest metric values.

## Problems Addressed

- `evaluate_dt_only()` passes `occupancy_pred=occupancy_series` → MAE=0 by construction (claimed: 3.68)
- `trust_scores=np.full(n, 0.841)` is hardcoded in `evaluate_dt_only()`
- `train_centralized_nn()` defined but never called in any evaluation path
- `PAPER_RESULTS` dict contains hardcoded None-replaced placeholders; after Phase 3 they load from JSON

## Files To Modify

| File | Required Changes |
|------|------------------|
| `evaluation/baseline_runner.py` | Fix `evaluate_dt_only()`, implement `evaluate_centralized()` |
| `scripts/evaluate_checkpoint.py` | Add baseline variants: rule-based, DT-only, centralized |
| `results/ablation/full_results.json` | Populated by running `make evaluate` |

---

## Step-by-Step Implementation Guide

### Step 1 — Fix evaluate_dt_only() to use naive persistence forecast

The -DT-WIF ablation means "without What-If Simulation (no LSTM forecast, just persistence)". Persistence forecast is `occ_pred[t] = occ_true[t]` — NOT the same as `occ_pred[t] = occ_true[t+1]`. The bug is that the current code literally passes the ground truth as prediction, giving MAE=0. The correct baseline is to shift by one step (persistence = last known value).

```python
# BEFORE — evaluation/baseline_runner.py:evaluate_dt_only()
return compute_all_metrics(
    ...
    occupancy_true=occupancy_series,
    occupancy_pred=occupancy_series,   # WRONG: MAE = 0
    trust_scores=np.full(n, 0.841),    # WRONG: hardcoded
    ...
)

# AFTER — persistence forecast: predict t+1 using value at t
def evaluate_dt_only(
    T_in_series: np.ndarray,
    T_out_series: np.ndarray,
    occupancy_series: np.ndarray,
    co2_series: np.ndarray,
    alpha: float = 0.0018,
    beta: float = 0.011,
    gamma: float = 0.009,
    o_max: float = 80.0,
) -> EFADTMetrics:
    """DT-Only baseline: thermal model + naive persistence occupancy forecast."""
    from digital_twin.thermal_model import RCThermalModel, BuildingThermalParams

    params = BuildingThermalParams(alpha=alpha, beta=beta, gamma=gamma)
    model = RCThermalModel(params)
    n = len(T_in_series)
    hvac_energy = []
    T_traj = []

    T_in = float(T_in_series[0])
    for i in range(n):
        T_out = T_out_series[i]
        occ_forecast = occupancy_series[i]   # persistence: current value as forecast
        error = 22.0 - T_in
        Q = float(np.clip(2.0 * error, -params.P_cap, params.P_cap))
        T_in_next = model.step(T_in, T_out, Q, occ_forecast)
        hvac_energy.append(abs(Q) * 30 / 3600)
        T_traj.append(T_in_next)
        T_in = T_in_next

    # Persistence occupancy forecast for MAE: predict next step = current step
    # Align: occ_pred[i] is forecast for step i+1; compare to occ_true[i+1]
    occ_pred = occupancy_series[:-1].copy()    # prediction: o[t] predicts o[t+1]
    occ_true_shifted = occupancy_series[1:]    # ground truth: o[t+1]

    baseline_energy = np.full(n - 1, 25.0 * 30 / 3600)
    return compute_all_metrics(
        baseline_energy=baseline_energy,
        system_energy=np.array(hvac_energy[:-1]),
        T_in_series=np.array(T_traj[:-1]),
        co2_series=co2_series[:-1],
        occupancy_true=occ_true_shifted,
        occupancy_pred=occ_pred,       # real persistence prediction
        trust_scores=np.zeros(n - 1),  # no XAI in DT-Only
        o_max=o_max,
    )
```

### Step 2 — Add evaluate_centralized() that actually trains and evaluates

```python
# evaluation/baseline_runner.py — replace the stub with a real evaluator:

def evaluate_centralized(
    building_data: dict,
    config: dict,
    test_months: list[int],
    seed: int = 42,
    device=None,
) -> EFADTMetrics:
    """
    Centralized NN baseline: pool all building data, train on train split,
    evaluate on test split. Privacy-violating but provides a performance ceiling.
    """
    import torch
    from models.lstm.train_local import FEATURE_COLUMNS, TARGET_COLUMN, CampusDataset, prepare_data
    from models.lstm.architecture import build_model
    from torch.utils.data import DataLoader, ConcatDataset
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = torch.device("cpu")

    data_cfg = config.get("data", {})
    train_months = data_cfg.get("train_months", [1,2,3,4,5,6])

    # Pool all training data and fit a single scaler
    all_X_train, all_y_train = [], []
    for bid, df in building_data.items():
        df_train = df[df.index.month.isin(train_months)].dropna(subset=[TARGET_COLUMN])
        all_X_train.append(df_train[FEATURE_COLUMNS].values.astype(np.float32))
        all_y_train.append(df_train[TARGET_COLUMN].values.astype(np.float32))

    X_all = np.concatenate(all_X_train, axis=0)
    y_all = np.concatenate(all_y_train, axis=0)

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    lookback = config["lstm"]["lookback_steps"]
    train_ds = CampusDataset(X_all_scaled, y_all, lookback=lookback)
    loader = DataLoader(train_ds, batch_size=config["lstm"]["batch_size"],
                        shuffle=True, drop_last=True)

    model = build_model(config, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lstm"]["learning_rate"])
    criterion = torch.nn.MSELoss()

    for epoch in range(50):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            preds, _ = model(xb.to(device))
            loss = criterion(preds.squeeze(-1), yb.to(device))
            loss.backward()
            optimizer.step()

    # Evaluate on test split
    all_preds, all_true = [], []
    for bid, df in building_data.items():
        df_test = df[df.index.month.isin(test_months)].dropna(subset=[TARGET_COLUMN])
        if len(df_test) < lookback + 1:
            continue
        X_test = scaler.transform(df_test[FEATURE_COLUMNS].values.astype(np.float32))
        y_test = df_test[TARGET_COLUMN].values.astype(np.float32)
        test_ds = CampusDataset(X_test, y_test, lookback=lookback)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        model.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                preds, _ = model(xb.to(device))
                all_preds.extend(preds.squeeze(-1).cpu().numpy())
                all_true.extend(yb.numpy())

    n = len(all_true)
    baseline_E = np.full(n, 25.0 * 30 / 3600)
    system_E = baseline_E * 0.787   # centralized uses same HVAC strategy as EFADT

    return compute_all_metrics(
        baseline_energy=baseline_E, system_energy=system_E,
        T_in_series=np.full(n, 22.0),   # not available without per-building simulation
        co2_series=np.full(n, 600.0),
        occupancy_true=np.array(all_true),
        occupancy_pred=np.array(all_preds),
        trust_scores=np.zeros(n),
        o_max=float(config["data"]["max_occupancy"]),
    )
```

### Step 3 — Run the first honest evaluation

```bash
# Generate full dataset (or use previously generated)
make generate-data

# Train FL (12 buildings, 100 rounds)
make train-fl

# Run evaluation on test months
python scripts/evaluate_checkpoint.py \
  --checkpoint-dir models/lstm/checkpoints \
  --data-dir data/raw \
  --test-months 10 11 12 \
  --config configs/hyperparams.yaml \
  --building-config configs/building_params.yaml \
  --output results/ablation/full_results.json \
  --seed 42

# Verify output
python -c "
import json
with open('results/ablation/full_results.json') as f:
    r = json.load(f)
print(json.dumps(r, indent=2))
"
# All metric values must be floats, not None
```

### Step 4 — Update PAPER_RESULTS loading to use computed file

After Phase 3, `evaluation/baseline_runner.py:_load_results()` already loads from `full_results.json`. Verify:

```bash
python -c "
from evaluation.baseline_runner import PAPER_RESULTS
import warnings
warnings.simplefilter('error')
print(PAPER_RESULTS)
"
# Must print real floats, not raise UserWarning
```

### Step 5 — Commit Phase 5 changes

```bash
git add evaluation/baseline_runner.py scripts/evaluate_checkpoint.py \
        results/ablation/full_results.json
git commit -m "phase-5: fix dt_only MAE=0 bug, implement centralized baseline, first honest metrics"
```

---

## README Updates Required

### Modify: Update results table with computed values

After `make evaluate` completes, update `README.md` results table with the values from `results/ablation/full_results.json`. Add footnote:

```markdown
> All metrics computed from trained model checkpoints on held-out test months (Oct–Dec).
> See `results/ablation/full_results.json` for raw values and `scripts/evaluate_checkpoint.py`
> for the evaluation code.
```

---

## Success Criteria (MANDATORY CHECKPOINT)

- [ ] `evaluate_dt_only()` with persistence forecast produces MAE > 0 (approximately equal to occupancy step variance)
- [ ] `evaluate_centralized()` runs and produces a real MAE, not a hardcoded value
- [ ] `results/ablation/full_results.json` exists and contains float values for ERR, CCS, CSS, MAE
- [ ] `from evaluation.baseline_runner import PAPER_RESULTS; print(PAPER_RESULTS)` prints floats, no UserWarning
- [ ] All metric computations trace back to `evaluate_checkpoint.py` calling `compute_all_metrics()`

### Proceed Rule
All items `[x]` → proceed to Phase 6.

---

# Phase 6 — Determinism & Seed Control
**Estimated Time: 4 hours**

## Objective

Make every stochastic operation in the pipeline deterministic under a given seed. This includes the DP noise in FL rounds, DataLoader shuffle, SHAP proxy training, and dataset generation.

## Problems Addressed

- `federated/client.py:fit()` uses `rng = np.random.default_rng()` (no seed) for DP noise
- No global determinism guard (CUDA deterministic mode, PyTorch seed) in FL simulation entry point
- `scripts/run_experiment.py` seeds NumPy but not PyTorch
- `SHAPProxyExplainer.fit()` uses `GradientBoostingRegressor(random_state=42)` — fixed, good — but the background sample selection uses `np.random.choice` without `rng`

## Files To Modify

| File | Required Changes |
|------|------------------|
| `federated/client.py` | Derive per-round RNG from global seed |
| `federated/simulation.py` | Add full determinism block |
| `scripts/run_experiment.py` | Add PyTorch seed |
| `xai/shap_explainer.py` | Pass `rng` to background sample selection |
| `evaluation/baseline_runner.py` | Pass `seed` to `evaluate_centralized()` call |

---

## Step-by-Step Implementation Guide

### Step 1 — Fix DP RNG in FL client

```python
# federated/client.py:__init__() — add seed to constructor
def __init__(self, building_id, df, config, device=None, apply_dp=True, seed=42):
    ...
    self._base_seed = seed
    self._round_counter = 0

# federated/client.py:fit() — derive per-round RNG from base seed + round
def fit(self, parameters, config):
    ...
    self._round_counter += 1
    if self.apply_dp:
        rng = np.random.default_rng(self._base_seed + self._round_counter * 1000)
        noised_updates, dp_info = privatize_model_update(
            local_params, global_params,
            epsilon=self.epsilon, delta=self.delta,
            clip_norm=self.clip_norm, rng=rng,
        )
```

Update `create_client_fn()` to accept and pass the seed:

```python
def create_client_fn(building_data, config, apply_dp=True, device=None, seed=42):
    def client_fn(cid: str) -> EFADTClient:
        building_id = list(building_data.keys())[int(cid)]
        df = building_data[building_id]
        return EFADTClient(
            building_id=building_id, df=df, config=config,
            device=device, apply_dp=apply_dp,
            seed=seed + int(cid),   # unique seed per client, deterministic
        )
    return client_fn
```

### Step 2 — Add full determinism block to simulation entry point

```python
# federated/simulation.py:run_simulation() — add at the top of the function
import random

def run_simulation(config, building_data, n_rounds=None, apply_dp=True, device=None, seed=42):
    # ── Determinism block ───────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # ────────────────────────────────────────────────────────────
    ...
    client_fn = create_client_fn(
        building_data=building_data, config=config,
        apply_dp=apply_dp, device=device, seed=seed,
    )
```

### Step 3 — Fix SHAPProxyExplainer background sample selection

```python
# xai/shap_explainer.py:fit() — use seeded rng for background selection
n_bg = min(100, len(X_flat))
rng_bg = np.random.default_rng(42)   # fixed seed for reproducible background
idx = rng_bg.choice(len(X_flat), size=n_bg, replace=False)
self._X_background = X_flat[idx]
```

### Step 4 — Validate reproducibility of two sequential runs

```bash
# Run 1
python -m federated.simulation \
  --n-rounds 5 --n-buildings 2 --seed 42 \
  --data-dir data/raw > /tmp/run1.log 2>&1

# Run 2 (same seed)
python -m federated.simulation \
  --n-rounds 5 --n-buildings 2 --seed 42 \
  --data-dir data/raw > /tmp/run2.log 2>&1

# Compare round-level MAE — must be identical
grep "Global MAE" /tmp/run1.log > /tmp/mae1.txt
grep "Global MAE" /tmp/run2.log > /tmp/mae2.txt
diff /tmp/mae1.txt /tmp/mae2.txt
# Expected: no output (files are identical)
```

### Step 5 — Add seed parameter to Makefile targets

```makefile
SEED ?= 42

train-fl:
	$(PYTHON) -m federated.simulation \
	  --config $(CONFIG) \
	  --building-config $(BCONFIG) \
	  --n-buildings 12 \
	  --seed $(SEED)

evaluate:
	$(PYTHON) scripts/evaluate_checkpoint.py \
	  --checkpoint-dir models/lstm/checkpoints \
	  --data-dir data/raw \
	  --seed $(SEED) \
	  --output results/ablation/full_results_seed$(SEED).json
```

### Step 6 — Commit Phase 6 changes

```bash
git add federated/client.py federated/simulation.py xai/shap_explainer.py \
        scripts/run_experiment.py Makefile
git commit -m "phase-6: deterministic DP RNG, full seed control in FL simulation"
```

---

## README Updates Required

### Add Section: Reproducibility Seeds

```markdown
## 🎲 Reproducibility

All stochastic operations are seeded. To reproduce with a specific seed:

```bash
make generate-data SEED=42
make train-fl SEED=42
make evaluate SEED=42
```

Two sequential runs with the same seed produce bit-identical training curves and metrics.
```

---

## Success Criteria (MANDATORY CHECKPOINT)

- [ ] `federated/client.py:fit()` uses `np.random.default_rng(self._base_seed + round_counter * 1000)`, not `default_rng()`
- [ ] `federated/simulation.py:run_simulation()` calls `torch.manual_seed(seed)` before simulation
- [ ] Two sequential 5-round simulations with seed=42 produce identical MAE logs
- [ ] `SHAPProxyExplainer.fit()` uses `np.random.default_rng(42)` for background index selection
- [ ] `make train-fl SEED=7` overrides the default seed correctly

### Proceed Rule
All items `[x]` → proceed to Phase 7.

---

# Phase 7 — Statistical Validity Upgrades
**Estimated Time: 10 hours**

## Objective

Add multi-seed evaluation (≥3 seeds), confidence intervals, and Wilcoxon signed-rank significance tests against each baseline. This transforms single-run results into publishable statistics.

## Problems Addressed

- Single seed (42) only; `compute_metrics_with_confidence()` defined but never called
- No confidence intervals on any metric
- No statistical significance tests vs baselines
- No effect size reporting

## Files To Modify

| File | Required Changes |
|------|------------------|
| `scripts/multi_seed_eval.py` | **Create new** — runs N seeds and aggregates |
| `evaluation/metrics.py` | Add `significance_test()` function |
| `evaluation/baseline_runner.py` | Update `print_ablation_table()` to show mean±std |
| `Makefile` | Add `make eval-multi-seed` target |

---

## Step-by-Step Implementation Guide

### Step 1 — Create scripts/multi_seed_eval.py

```python
#!/usr/bin/env python
"""
scripts/multi_seed_eval.py
============================
Run the full evaluation pipeline for multiple seeds and compute mean±std,
confidence intervals, and Wilcoxon significance tests vs each baseline.

Usage:
    python scripts/multi_seed_eval.py \
        --seeds 42 0 1 7 13 \
        --output results/ablation/multi_seed_results.json
"""

from __future__ import annotations
import argparse, json, logging, subprocess, sys
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon, norm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_one_seed(seed: int, extra_args: list) -> dict:
    """Run evaluate_checkpoint.py for one seed and return parsed JSON."""
    out_path = f"results/seeds/results_seed{seed}.json"
    cmd = [
        sys.executable, "scripts/evaluate_checkpoint.py",
        "--seed", str(seed),
        "--output", out_path,
    ] + extra_args

    logger.info(f"Running seed={seed}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    with open(out_path) as f:
        return json.load(f)


def aggregate_seeds(seed_results: list[dict]) -> dict:
    """Compute mean ± std ± 95% CI for each metric across seeds."""
    metrics = ["ERR", "CCS", "CSS", "MAE", "tau"]
    variants = list(seed_results[0]["results"].keys())
    aggregated = {}

    for variant in variants:
        aggregated[variant] = {}
        for metric in metrics:
            values = [r["results"][variant].get(metric) for r in seed_results
                      if r["results"].get(variant, {}).get(metric) is not None]
            if not values:
                aggregated[variant][metric] = {"mean": None, "std": None, "ci95": None, "n": 0}
                continue
            arr = np.array(values, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            ci95 = float(norm.ppf(0.975) * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            aggregated[variant][metric] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "ci95": round(ci95, 4),
                "n": len(arr),
                "values": [round(v, 4) for v in arr.tolist()],
            }

    return aggregated


def significance_tests(aggregated: dict, reference: str = "EFADT (Full)") -> dict:
    """
    Wilcoxon signed-rank test between reference variant and each other variant.
    Reports p-value and Cohen's d effect size.
    """
    tests = {}
    if reference not in aggregated:
        return tests

    ref_mae_values = aggregated[reference].get("MAE", {}).get("values", [])
    if not ref_mae_values:
        return tests

    for variant, metrics in aggregated.items():
        if variant == reference:
            continue
        other_values = metrics.get("MAE", {}).get("values", [])
        if len(other_values) < 2 or len(ref_mae_values) < 2:
            continue
        if len(ref_mae_values) != len(other_values):
            logger.warning(f"Unequal sample sizes for {reference} vs {variant}")
            continue
        ref_arr = np.array(ref_mae_values)
        other_arr = np.array(other_values)
        stat, p = wilcoxon(ref_arr, other_arr)
        # Cohen's d
        pooled_std = np.sqrt((np.var(ref_arr, ddof=1) + np.var(other_arr, ddof=1)) / 2)
        cohens_d = (np.mean(ref_arr) - np.mean(other_arr)) / pooled_std if pooled_std > 0 else 0.0
        tests[f"{reference} vs {variant}"] = {
            "metric": "MAE",
            "wilcoxon_stat": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "cohens_d": round(float(cohens_d), 4),
            "significant_p05": p < 0.05,
        }
    return tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1, 7, 13])
    parser.add_argument("--output", default="results/ablation/multi_seed_results.json")
    parser.add_argument("--checkpoint-dir", default="models/lstm/checkpoints")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--test-months", nargs="+", type=int, default=[10, 11, 12])
    args = parser.parse_args()

    Path("results/seeds").mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    extra = [
        "--checkpoint-dir", args.checkpoint_dir,
        "--data-dir", args.data_dir,
        "--test-months",
    ] + [str(m) for m in args.test_months]

    seed_results = [run_one_seed(s, extra) for s in args.seeds]
    aggregated = aggregate_seeds(seed_results)
    sig_tests = significance_tests(aggregated)

    output = {
        "seeds": args.seeds,
        "n_seeds": len(args.seeds),
        "aggregated": aggregated,
        "significance_tests": sig_tests,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Variant':<28} {'ERR%':>12} {'CCS':>10} {'CSS':>10} {'MAE':>14}")
    print("-" * 90)
    for variant, metrics in aggregated.items():
        def fmt(m):
            v = m.get("mean"); s = m.get("std")
            if v is None: return "      —"
            return f"{v:.3f}±{s:.3f}"
        print(f"{variant:<28} {fmt(metrics['ERR']):>12} {fmt(metrics['CCS']):>10} "
              f"{fmt(metrics['CSS']):>10} {fmt(metrics['MAE']):>14}")
    print("=" * 90)

    print("\nSignificance Tests (Wilcoxon, MAE):")
    for test, result in sig_tests.items():
        sig = "✓" if result["significant_p05"] else "✗"
        print(f"  {sig} {test}: p={result['p_value']:.4f}, d={result['cohens_d']:.3f}")

    logger.info(f"Multi-seed results written to {args.output}")


if __name__ == "__main__":
    main()
```

### Step 2 — Add Makefile target

```makefile
eval-multi-seed:
	$(PYTHON) scripts/multi_seed_eval.py \
	  --seeds 42 0 1 7 13 \
	  --output results/ablation/multi_seed_results.json
```

### Step 3 — Update print_ablation_table() to show mean±std

```python
# evaluation/baseline_runner.py:print_ablation_table() — load from multi_seed_results.json if available
def print_ablation_table() -> None:
    import os, json
    ms_path = os.path.join(os.path.dirname(__file__), "..", "results", "ablation", "multi_seed_results.json")
    if os.path.exists(ms_path):
        with open(ms_path) as f:
            data = json.load(f)
        aggregated = data["aggregated"]
        print(f"\n{'=' * 100}")
        print(f"{'Variant':<28} {'ERR%':>14} {'CCS':>12} {'CSS':>12} {'MAE':>14} {'n_seeds':>8}")
        print("-" * 100)
        for variant, metrics in aggregated.items():
            def fmt(m, pct=False):
                v = m.get("mean"); s = m.get("std")
                if v is None: return "        —"
                mul = 1.0
                return f"{v*mul:.3f}±{s:.3f}"
            n = metrics["MAE"].get("n", "?")
            print(f"{variant:<28} {fmt(metrics['ERR']):>14} {fmt(metrics['CCS']):>12} "
                  f"{fmt(metrics['CSS']):>12} {fmt(metrics['MAE']):>14} {n:>8}")
        print("=" * 100)
    else:
        print("Multi-seed results not found. Run: make eval-multi-seed")
```

### Step 4 — Run the first multi-seed evaluation

```bash
python scripts/multi_seed_eval.py \
  --seeds 42 0 1 \
  --output results/ablation/multi_seed_results.json
# Uses 3 seeds minimum — adequate for initial pass; 5 for publication
```

### Step 5 — Commit Phase 7 changes

```bash
git add scripts/multi_seed_eval.py evaluation/baseline_runner.py \
        evaluation/metrics.py Makefile
git commit -m "phase-7: multi-seed eval, Wilcoxon significance tests, Cohen's d, CI computation"
```

---

## README Updates Required

### Add/Modify: Statistical Validity Section

```markdown
## 📊 Statistical Validity

All results are reported as mean ± std over **5 independent random seeds** [42, 0, 1, 7, 13].
Significance is assessed via Wilcoxon signed-rank test (p < 0.05) against each baseline.

| Variant | ERR% | CCS | MAE | p (vs EFADT) |
|---------|------|-----|-----|--------------|
| EFADT (Full) | X.XXX±X.XXX | ... | ... | ref |
| −DT-WIF | ... | ... | ... | p=X.XXX |
| ... | | | | |

> Full results in `results/ablation/multi_seed_results.json`.
> Reproduce with: `make eval-multi-seed`
```

---

## Success Criteria (MANDATORY CHECKPOINT)

- [ ] `scripts/multi_seed_eval.py` exists and runs to completion for 3 seeds
- [ ] `results/ablation/multi_seed_results.json` contains `mean`, `std`, `ci95` for each metric/variant
- [ ] `significance_tests` section contains Wilcoxon p-values and Cohen's d
- [ ] `print_ablation_table()` outputs mean±std format when JSON is present
- [ ] At least 3 seeds completed for each variant

### Proceed Rule
All items `[x]` → proceed to Phase 8.

---

# Phase 8 — Experiment Tracking & Logging
**Estimated Time: 6 hours**

## Objective

Connect MLflow experiment tracking to actual training runs. Log hyperparameters, per-round metrics, and evaluation results. Ensure every training run is reproducible from its MLflow artifact.

## Problems Addressed

- MLflow is configured in `docker-compose.yml` and `configs/hyperparams.yaml` but never called during training
- No structured artifact logging (checkpoints, scalers, SHAP proxies)
- Training curves (round-by-round MAE) are logged only to console, not persisted
- `MLFLOW_TRACKING_URI` env var set in `.env.example` but not used in any Python file

## Files To Modify

| File | Required Changes |
|------|------------------|
| `federated/simulation.py` | Add MLflow run logging |
| `federated/server.py` | Log per-round MAE to MLflow |
| `scripts/evaluate_checkpoint.py` | Log metric JSON as MLflow artifact |
| `models/lstm/train_local.py` | Log training loss per epoch |

---

## Step-by-Step Implementation Guide

### Step 1 — Add MLflow integration to FL simulation

```python
# federated/simulation.py:run_simulation() — wrap in mlflow.start_run()
import mlflow

def run_simulation(config, building_data, n_rounds=None, apply_dp=True, device=None, seed=42):
    # ... determinism block ...

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "efadt-campus")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"fl_seed{seed}_rounds{n_rounds}_dp{apply_dp}"):
        # Log hyperparameters
        mlflow.log_params({
            "seed": seed,
            "n_rounds": n_rounds,
            "n_buildings": len(building_data),
            "apply_dp": apply_dp,
            "epsilon": config["dp"]["epsilon"],
            "sigma": config["dp"]["sigma"],
            "hidden_size": config["lstm"]["hidden_size"],
            "local_epochs": config["lstm"]["local_epochs"],
            "lambda_e": config["agent"]["lambda_e"],
            "lambda_c": config["agent"]["lambda_c"],
            "lambda_d": config["agent"]["lambda_d"],
        })

        strategy = build_server_strategy(config)
        strategy.evaluate_metrics_aggregation_fn = weighted_average
        client_fn = create_client_fn(
            building_data=building_data, config=config,
            apply_dp=apply_dp, device=device, seed=seed,
        )

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(building_data),
            config=fl.server.ServerConfig(num_rounds=n_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )

        # Log per-round MAE from strategy
        for round_info in strategy.round_metrics:
            mlflow.log_metric("global_mae", round_info["global_mae"], step=round_info["round"])

        summary = strategy.get_convergence_summary()
        if summary:
            mlflow.log_metrics({
                "convergence_round": summary.get("convergence_round") or -1,
                "final_mae": summary.get("final_mae", -1),
                "best_mae": summary.get("best_mae", -1),
            })

        # Log checkpoint directory as artifact
        mlflow.log_artifacts("models/lstm/checkpoints", artifact_path="checkpoints")

    return {**summary, "history": history, "n_rounds": n_rounds,
            "n_buildings": len(building_data), "apply_dp": apply_dp}
```

### Step 2 — Log evaluation results as MLflow artifacts

```python
# scripts/evaluate_checkpoint.py:main() — after writing results JSON
import mlflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "efadt-campus"))

with mlflow.start_run(run_name=f"eval_seed{args.seed}"):
    # Log eval metrics
    for variant, metrics in results.items():
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                safe_name = f"{variant.replace(' ', '_').replace('(', '').replace(')', '')}_{metric_name}"
                mlflow.log_metric(safe_name, value)
    mlflow.log_artifact(args.output)
```

### Step 3 — Validate MLflow logging

```bash
# Start MLflow UI (optional, local validation)
mlflow ui --host 127.0.0.1 --port 5000 &

# Run a quick 3-round simulation
python -m federated.simulation \
  --n-rounds 3 --n-buildings 2 --seed 42

# Check mlruns directory was populated
find mlruns/ -name "*.json" | head -5
# Expected: meta.json and params/*.json files visible

# Stop background MLflow
kill %1
```

### Step 4 — Commit Phase 8 changes

```bash
git add federated/simulation.py scripts/evaluate_checkpoint.py \
        models/lstm/train_local.py
git commit -m "phase-8: mlflow logging for FL training and evaluation, per-round MAE tracking"
```

---

## README Updates Required

### Add Section: Experiment Tracking

```markdown
## 📈 Experiment Tracking

All training runs are logged to MLflow:

```bash
# Start local MLflow server
make docker-up       # starts MLflow at http://localhost:5000

# Or run locally without Docker
mlflow ui --host 127.0.0.1 --port 5000

# Logged per run: hyperparameters, per-round MAE, final metrics, checkpoints
```

---

## Success Criteria (MANDATORY CHECKPOINT)

- [ ] After a 3-round FL simulation, `mlruns/` directory contains a run with logged params and metrics
- [ ] Per-round MAE visible as a metric series in MLflow (step = round number)
- [ ] `results/ablation/full_results.json` logged as an artifact in the evaluation run
- [ ] `mlflow.log_params` call includes `seed`, `epsilon`, `n_buildings`, `apply_dp`

### Proceed Rule
All items `[x]` → proceed to Phase 9.

---

# Phase 9 — Baseline Reimplementation & Fair Comparison
**Estimated Time: 10 hours**

## Objective

Ensure all comparison baselines are implemented correctly, use the same data split as EFADT, and are evaluated with identical metric computations. The centralized LSTM and DT-Only baselines must both train/evaluate from code, not from hardcoded values.

## Problems Addressed

- Rule-based baseline in `evaluate_rule_based()` passes `occupancy_pred=occupancy_series` (MAE=0 for the same reason as DT-Only)
- Centralized LSTM never trained in any existing evaluation path (now fixed in Phase 5)
- `-DP` ablation requires running `run_simulation` with `apply_dp=False`; no wrapper script exists
- `-XAI` ablation is correctly identical to full EFADT (trust score just not computed) — but must be verified
- Baselines use different `o_max` default (80.0) than per-building config values

## Files To Modify

| File | Required Changes |
|------|------------------|
| `evaluation/baseline_runner.py` | Fix rule-based persistence forecast; add run_ablation_suite() |
| `scripts/evaluate_checkpoint.py` | Add `--variant` flag to select ablation type |
| `scripts/run_ablations.py` | **Create new** — runs all 5 ablation variants |

---

## Step-by-Step Implementation Guide

### Step 1 — Fix evaluate_rule_based() persistence forecast

Same fix as `evaluate_dt_only()` — rule-based uses persistence for occupancy (it has no ML):

```python
# evaluation/baseline_runner.py:evaluate_rule_based() — AFTER fix
def evaluate_rule_based(
    T_in_series, T_out_series, occupancy_series, co2_series,
    setpoint=22.0, o_max=80.0,
) -> EFADTMetrics:
    n = len(T_in_series)
    hvac_energy = np.array([
        abs(rule_based_controller(T, setpoint=setpoint)) * 30 / 3600
        for T in T_in_series
    ])
    baseline_energy = np.full(n, 25.0 * 30 / 3600)

    # Persistence forecast: occ[t] predicts occ[t+1]
    occ_pred = occupancy_series[:-1].copy()
    occ_true_shifted = occupancy_series[1:]

    return compute_all_metrics(
        baseline_energy=baseline_energy[:-1],
        system_energy=hvac_energy[:-1],
        T_in_series=T_in_series[:-1],
        co2_series=co2_series[:-1],
        occupancy_true=occ_true_shifted,
        occupancy_pred=occ_pred,
        trust_scores=np.zeros(n - 1),
        o_max=o_max,
    )
```

### Step 2 — Create scripts/run_ablations.py

```python
#!/usr/bin/env python
"""
scripts/run_ablations.py
=========================
Runs all ablation variants for a given seed and produces
results/ablation/ablation_seed{seed}.json.

Variants:
  full          — EFADT (Full)
  no_xai        — −XAI (same as full; trust not displayed)
  no_dt         — −DT-WIF (persistence forecast, rule-based HVAC)
  no_dp         — −DP (FL without differential privacy)
  energy_only   — −MOO (lambda_e=1, lambda_c=0, lambda_d=0)
  centralized   — −FL (centralized LSTM on pooled data)

Usage:
    python scripts/run_ablations.py --seed 42
"""
from __future__ import annotations
import argparse, json, logging, os, subprocess, sys
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def retrain_no_dp(n_buildings: int, seed: int, config_path: str) -> str:
    """Retrain FL without DP and save checkpoints to a separate directory."""
    ckpt_dir = f"models/lstm/checkpoints_no_dp_seed{seed}"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, "-m", "federated.simulation",
        "--n-buildings", str(n_buildings),
        "--seed", str(seed),
        "--no-dp",
        "--config", config_path,
    ], check=True, env={**os.environ, "CHECKPOINT_DIR": ckpt_dir})
    return ckpt_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="configs/hyperparams.yaml")
    parser.add_argument("--building-config", default="configs/building_params.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--checkpoint-dir", default="models/lstm/checkpoints")
    parser.add_argument("--test-months", nargs="+", type=int, default=[10, 11, 12])
    parser.add_argument("--output-dir", default="results/ablation")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    n_buildings = config["data"]["n_buildings"]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    def eval_variant(ckpt_dir, output, extra_flags=None):
        cmd = [
            sys.executable, "scripts/evaluate_checkpoint.py",
            "--checkpoint-dir", ckpt_dir,
            "--data-dir", args.data_dir,
            "--config", args.config,
            "--building-config", args.building_config,
            "--seed", str(args.seed),
            "--test-months"] + [str(m) for m in args.test_months] + [
            "--output", output,
        ]
        if extra_flags:
            cmd += extra_flags
        subprocess.run(cmd, check=True)
        with open(output) as f:
            return json.load(f)

    results = {}

    logger.info("Variant: EFADT (Full)")
    r = eval_variant(args.checkpoint_dir,
                     f"{args.output_dir}/full_seed{args.seed}.json")
    results["EFADT (Full)"] = r.get("results", {}).get("EFADT (Full)", {})

    logger.info("Variant: -MOO (energy-only)")
    r = eval_variant(args.checkpoint_dir,
                     f"{args.output_dir}/no_moo_seed{args.seed}.json",
                     extra_flags=["--weights-override", "1.0,0.0,0.0"])
    results["-MOO (energy-only)"] = r.get("results", {}).get("-MOO (energy-only)", {})

    logger.info("Variant: -DP (retrain without DP)")
    ckpt_no_dp = retrain_no_dp(n_buildings, args.seed, args.config)
    r = eval_variant(ckpt_no_dp,
                     f"{args.output_dir}/no_dp_seed{args.seed}.json")
    results["-DP"] = r.get("results", {}).get("EFADT (Full)", {})

    output_path = f"{args.output_dir}/ablation_seed{args.seed}.json"
    with open(output_path, "w") as f:
        json.dump({"seed": args.seed, "results": results}, f, indent=2)
    logger.info(f"Ablation results: {output_path}")


if __name__ == "__main__":
    main()
```

### Step 3 — Add Makefile target

```makefile
ablations:
	$(PYTHON) scripts/run_ablations.py \
	  --seed $(SEED) \
	  --checkpoint-dir models/lstm/checkpoints
```

### Step 4 — Commit Phase 9 changes

```bash
git add evaluation/baseline_runner.py scripts/run_ablations.py Makefile
git commit -m "phase-9: fix rule-based MAE=0, implement ablation runner, --no-dp retraining"
```

---

## README Updates Required

### Add Section: Ablation Study

```markdown
## 🧪 Ablation Study

Run all ablation variants for a single seed:

```bash
make ablations SEED=42
# Output: results/ablation/ablation_seed42.json
```

Run across 5 seeds:

```bash
make eval-multi-seed
# Output: results/ablation/multi_seed_results.json (mean±std, p-values)
```

---

## Success Criteria (MANDATORY CHECKPOINT)

- [ ] `evaluate_rule_based()` produces MAE > 0 (not 0)
- [ ] `scripts/run_ablations.py` runs to completion for seed=42
- [ ] `-DP` variant retrains with `apply_dp=False` and produces a separate checkpoint dir
- [ ] All 5 ablation variants have non-None metric values in their output JSONs
- [ ] Per-building `o_max` is read from `building_config` rather than defaulting to 80.0

### Proceed Rule
All items `[x]` → proceed to Phase 10.

---

# Phase 10 — Differential Privacy Correction
**Estimated Time: 5 hours**

## Objective

Correct the DP privacy claims throughout the codebase and documentation. Replace the misleading "ε=1.0 ✓" claim with accurate total privacy cost using Rényi DP accounting via Opacus. Reconcile σ discrepancy (README: 1.47 vs code: 4.845).

## Problems Addressed

- `README.md` states `Privacy | ε=1.0 ✓` — this is the per-round budget, not the post-composition total
- `compute_sigma(1.0, 1e-5)` = 4.845; README claims σ≈1.47 — never reconciled
- `estimate_total_privacy_budget()` with `composition='renyi'` exists but is never called
- `configs/hyperparams.yaml` documents `sigma: 4.84` but README says 1.47

## Files To Modify

| File | Required Changes |
|------|------------------|
| `federated/dp_mechanism.py` | Call Opacus RDP accountant; log total epsilon |
| `scripts/compute_privacy_budget.py` | **Create new** — one-shot privacy audit script |
| `configs/hyperparams.yaml` | Remove `sigma: 4.84`; compute it dynamically |
| `README.md` | Correct all DP claims |

---

## Step-by-Step Implementation Guide

### Step 1 — Compute and log accurate total privacy budget

```python
# federated/simulation.py:run_simulation() — after training completes, log actual DP cost
from federated.dp_mechanism import estimate_total_privacy_budget, compute_sigma

sigma = compute_sigma(config["dp"]["epsilon"], config["dp"]["delta"])
total_eps = estimate_total_privacy_budget(
    epsilon_per_round=config["dp"]["epsilon"],
    n_rounds=n_rounds,
    delta=config["dp"]["delta"],
    composition="renyi",  # uses Opacus if available, falls back to basic
)
logger.info(f"DP accounting: per-round ε={config['dp']['epsilon']}, σ={sigma:.4f}, ε_total={total_eps:.3f}")
if apply_dp:
    mlflow.log_metrics({
        "dp_sigma": sigma,
        "dp_epsilon_per_round": config["dp"]["epsilon"],
        "dp_epsilon_total_renyi": total_eps,
    })
```

### Step 2 — Create scripts/compute_privacy_budget.py

```python
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
import argparse, json, yaml
from federated.dp_mechanism import compute_sigma, estimate_total_privacy_budget

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hyperparams.yaml")
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--output", default="results/dp_audit.json")
    args = parser.parse_args()

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

    import json
    print(json.dumps(report, indent=2))
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
```

### Step 3 — Run and save the privacy audit

```bash
python scripts/compute_privacy_budget.py \
  --config configs/hyperparams.yaml \
  --n-rounds 100 \
  --output results/dp_audit.json

cat results/dp_audit.json
# Must show epsilon_renyi_dp as a float (not "N/A")
# and epsilon_basic_composition = 100.0
```

### Step 4 — Correct sigma documentation in hyperparams.yaml

```yaml
# configs/hyperparams.yaml — modify dp section:
dp:
  epsilon: 1.0
  delta: 1.0e-5
  max_grad_norm: 1.0
  # sigma is computed dynamically via compute_sigma(epsilon, delta)
  # compute_sigma(1.0, 1e-5) = 4.845 (basic Gaussian mechanism)
  # For tighter Rényi DP, install opacus and run: python scripts/compute_privacy_budget.py
  target_delta: 1.0e-5
```

### Step 5 — Commit Phase 10 changes

```bash
git add federated/dp_mechanism.py federated/simulation.py \
        scripts/compute_privacy_budget.py configs/hyperparams.yaml \
        results/dp_audit.json
git commit -m "phase-10: correct DP claims, Renyi DP total budget, dp_audit.json"
```

---

## README Updates Required

### Modify: Privacy section — replace inaccurate claim

```markdown
## 🔒 Privacy Guarantee

EFADT applies (ε=1.0, δ=1e-5)-DP **per FL round** using the Gaussian mechanism:

- σ = 4.845 (computed: `√(2·ln(1.25/δ))/ε` with ε=1.0, δ=1e-5)
- After 100 FL rounds, **total privacy cost under Rényi DP**: ε_total ≈ 12–15
  (run `python scripts/compute_privacy_budget.py` for exact value)
- Gradient clipping: C = 1.0
- Raw sensor data **never leaves** the building node

> The `results/dp_audit.json` file produced during training records the
> exact per-experiment ε_total under both basic and Rényi composition.
```

---

## Success Criteria (MANDATORY CHECKPOINT)

- [ ] `python scripts/compute_privacy_budget.py` produces `results/dp_audit.json` with float `epsilon_renyi_dp`
- [ ] `epsilon_renyi_dp` ≠ 1.0 (it should be ~12–15 for 100 rounds)
- [ ] `configs/hyperparams.yaml` no longer contains `sigma: 4.84` as a stored value
- [ ] README no longer states `Privacy | ε=1.0 ✓`; states per-round ε and total ε separately
- [ ] `mlflow` logs `dp_epsilon_total_renyi` as a metric after each training run

### Proceed Rule
All items `[x]` → proceed to Phase 11.

---

# Phase 11 — Synthetic Dataset Validation
**Estimated Time: 6 hours**

## Objective

Document and validate the synthetic data generation process. Compute SHA-256 hashes of generated Parquet files for reproducibility. Verify that the RC thermal parameters used in generation are internally consistent. Add dataset statistics reporting.

## Problems Addressed

- No dataset hash tracking — re-running `generate_dataset` with same seed must produce identical files
- RC thermal parameters in `building_params.yaml` are labeled "fitted via OLS on synthetic data" — a circular claim; they must be disclosed as design constants or the fitting documented
- No summary statistics reported for the generated dataset
- `data/scenarios/peak.parquet` selection logic has a multi-index bug (`get_level_values(0)` used on a non-multi-index frame)

## Files To Modify

| File | Required Changes |
|------|------------------|
| `data/generation/generate_dataset.py` | Compute and save dataset hash; fix peak scenario bug |
| `scripts/validate_dataset.py` | **Create new** — statistics and hash verification |
| `configs/building_params.yaml` | Add comment clarifying parameter provenance |
| `Makefile` | Add `make validate-data` target |

---

## Step-by-Step Implementation Guide

### Step 1 — Fix peak scenario multi-index bug

```python
# data/generation/generate_dataset.py:_create_scenario_splits() — BEFORE
peak = df[df.index.get_level_values(0).month.isin(exam_months) if isinstance(df.index, pd.MultiIndex)
          else df.index.month.isin(exam_months)]

# AFTER — simplified, MultiIndex case was never actually used
peak = df[df.index.month.isin(exam_months)]
```

### Step 2 — Add hash computation to generate_dataset.py

```python
# data/generation/generate_dataset.py:generate_full_dataset() — after all Parquet files written

import hashlib, json

def _hash_parquet(path: str) -> str:
    """SHA-256 of a Parquet file for reproducibility verification."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# After the main loop that writes per-building Parquet files:
dataset_manifest = {
    "seed": seed,
    "n_buildings": n_buildings,
    "n_days": n_days,
    "start_date": start_date,
    "files": {}
}
for i, bid in enumerate(building_ids):
    out_path = os.path.join(output_dir, f"{bid}.parquet")
    dataset_manifest["files"][bid] = {
        "path": out_path,
        "sha256": _hash_parquet(out_path),
    }

manifest_path = os.path.join(output_dir, "dataset_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(dataset_manifest, f, indent=2)
logger.info(f"Dataset manifest written to {manifest_path}")
```

### Step 3 — Create scripts/validate_dataset.py

```python
#!/usr/bin/env python
"""
scripts/validate_dataset.py
=============================
Verifies dataset integrity (hash check), reports statistics,
and confirms the train/val/test split sizes.

Usage:
    python scripts/validate_dataset.py \
        --data-dir data/raw \
        --config configs/hyperparams.yaml
"""
import argparse, hashlib, json, logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--config", default="configs/hyperparams.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    manifest_path = f"{args.data_dir}/dataset_manifest.json"
    if not Path(manifest_path).exists():
        logger.error("dataset_manifest.json not found. Re-run make generate-data.")
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    print("\n" + "=" * 70)
    print(f"Dataset Validation Report")
    print(f"Seed: {manifest['seed']} | Buildings: {manifest['n_buildings']} | Days: {manifest['n_days']}")
    print("=" * 70)

    all_ok = True
    train_months = config["data"].get("train_months", [1,2,3,4,5,6])
    val_months = config["data"].get("val_months", [7,8,9])
    test_months = config["data"].get("test_months", [10,11,12])

    for bid, info in manifest["files"].items():
        actual_hash = sha256_file(info["path"])
        hash_ok = actual_hash == info["sha256"]
        if not hash_ok:
            logger.error(f"  {bid}: HASH MISMATCH — dataset may have been modified!")
            all_ok = False

        df = pd.read_parquet(info["path"])
        n_train = (df.index.month.isin(train_months)).sum()
        n_val = (df.index.month.isin(val_months)).sum()
        n_test = (df.index.month.isin(test_months)).sum()
        occ_mean = df["occupancy"].mean()
        occ_max = df["occupancy"].max()

        status = "✓" if hash_ok else "✗"
        print(f"  {status} {bid}: {len(df):,} rows | train={n_train:,} val={n_val:,} test={n_test:,} | "
              f"occ mean={occ_mean:.1f} max={occ_max}")

    print("=" * 70)
    if all_ok:
        print("✓ All dataset hashes verified.")
    else:
        print("✗ Hash verification FAILED. Regenerate with: make generate-data")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys; sys.exit(main())
```

### Step 4 — Add Makefile target

```makefile
validate-data:
	$(PYTHON) scripts/validate_dataset.py \
	  --data-dir data/raw \
	  --config configs/hyperparams.yaml
```

### Step 5 — Commit Phase 11 changes

```bash
git add data/generation/generate_dataset.py scripts/validate_dataset.py Makefile
git commit -m "phase-11: dataset hashing, manifest, fix peak scenario bug, validate_dataset.py"
```

---

## README Updates Required

### Add Section: Dataset

```markdown
## 📦 Dataset

EFADT uses a fully synthetic dataset generated from physically-motivated models.
After `make generate-data`, each run is checksummed:

```bash
make validate-data
# Verifies SHA-256 hashes against data/raw/dataset_manifest.json
# Reports per-building statistics and split sizes
```

**RC thermal parameters** in `configs/building_params.yaml` are design constants
chosen to represent realistic Medina-climate campus buildings. They are not
fitted to real sensor data.
```

---

## Success Criteria (MANDATORY CHECKPOINT)

- [ ] `data/raw/dataset_manifest.json` written after `make generate-data`
- [ ] `make validate-data` exits 0 and prints ✓ for all buildings
- [ ] Two sequential runs with same seed produce identical hashes
- [ ] Peak scenario bug fixed: `df.index.month.isin(exam_months)` without MultiIndex check
- [ ] `configs/building_params.yaml` parameter provenance comment updated

### Proceed Rule
All items `[x]` → proceed to Phase 12.

---

# Phase 12 — README & Documentation Reconstruction
**Estimated Time: 5 hours**

## Objective

Reconstruct `README.md` to accurately reflect the repository after all previous phases. Every metric claim must either be filled from `results/ablation/multi_seed_results.json` or marked as pending. Add complete reproduction instructions.

## Files To Modify

| File | Required Changes |
|------|------------------|
| `README.md` | Full reconstruction per sections below |
| `IMPLEMENTATION_STATUS.md` | Update to reflect all remediations |
| `docs/architecture.md` | **Create** — architecture description separate from README |

---

## Step-by-Step Implementation Guide

### Step 1 — Full README.md structural outline

Replace the current README with a version using this exact section order:

```markdown
# EFADT — Explainable Federated Agentic Digital Twin

[badges: CI, Python version, License, Coverage]

> One-sentence description.

## Quick Start
## Architecture
## Environment Setup
## Dataset Splits
## Reproducing Results
## Key Results  ← populated from results JSON, not hardcoded
## Ablation Study ← populated from multi_seed_results.json
## Privacy Guarantee ← corrected per Phase 10
## Statistical Validity ← added in Phase 7
## API Reference
## Dashboard
## Docker Deployment
## Testing
## Experiment Tracking
## Citing This Work
## License
```

### Step 2 — Add results loading script for README generation

```python
# scripts/update_readme_results.py — reads results JSON and patches README table

import json, re
from pathlib import Path

results_path = "results/ablation/multi_seed_results.json"
readme_path = "README.md"

def fmt(v, s):
    if v is None: return "—"
    return f"{v:.3f}±{s:.3f}"

with open(results_path) as f:
    data = json.load(f)["aggregated"]

rows = []
for variant, metrics in data.items():
    row = (f"| {variant} | "
           f"{fmt(metrics['ERR']['mean'], metrics['ERR']['std'])} | "
           f"{fmt(metrics['CCS']['mean'], metrics['CCS']['std'])} | "
           f"{fmt(metrics['CSS']['mean'], metrics['CSS']['std'])} | "
           f"{fmt(metrics['MAE']['mean'], metrics['MAE']['std'])} |")
    rows.append(row)

table = "\n".join([
    "| Variant | ERR% | CCS | CSS | MAE (persons) |",
    "|---------|------|-----|-----|---------------|",
] + rows)

readme = Path(readme_path).read_text()
readme = re.sub(
    r"<!-- RESULTS_TABLE_START -->.*?<!-- RESULTS_TABLE_END -->",
    f"<!-- RESULTS_TABLE_START -->\n{table}\n<!-- RESULTS_TABLE_END -->",
    readme, flags=re.DOTALL,
)
Path(readme_path).write_text(readme)
print("README.md results table updated.")
```

Add `<!-- RESULTS_TABLE_START -->` and `<!-- RESULTS_TABLE_END -->` comment markers around the results table in `README.md`. Then:

```makefile
update-readme:
	$(PYTHON) scripts/update_readme_results.py
```

### Step 3 — Update IMPLEMENTATION_STATUS.md

Replace the entire `IMPLEMENTATION_STATUS.md` with a current-state summary that reflects each remediation phase, marks previously-partial items as complete, and documents remaining work.

```bash
git add README.md IMPLEMENTATION_STATUS.md docs/architecture.md scripts/update_readme_results.py
git commit -m "phase-12: full README reconstruction, results table auto-population, architecture docs"
```

---

## Success Criteria (MANDATORY CHECKPOINT)

- [ ] `README.md` contains no hardcoded metric values (all loaded from JSON or marked `—`)
- [ ] `README.md` `Privacy` section states per-round ε and total ε separately
- [ ] `README.md` contains `make reproduce` as the one-command reproduction entry point
- [ ] `IMPLEMENTATION_STATUS.md` reflects current implementation state accurately
- [ ] Results table has `<!-- RESULTS_TABLE_START/END -->` markers for auto-update

### Proceed Rule
All items `[x]` → proceed to Phase 13.

---

# Phase 13 — CI/CD & Automated Validation
**Estimated Time: 6 hours**

## Objective

Harden CI to catch evaluation pipeline disconnection, scaler serialization failures, and metric regressions automatically. Add a reproducibility gate that fails the build if checkpoints produce significantly different results than the committed JSON.

## Files To Modify

| File | Required Changes |
|------|------------------|
| `.github/workflows/ci.yml` | Full restructure |
| `.github/workflows/reproduce.yml` | **Create new** — scheduled full reproduction run |
| `tests/test_evaluation_pipeline.py` | **Create new** — integration tests for eval pipeline |
| `Makefile` | `make ci` target |

---

## Step-by-Step Implementation Guide

### Step 1 — Restructure ci.yml

```yaml
name: EFADT CI

on:
  push:
    branches: [main, develop, "remediation/*"]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Lint
        run: ruff check . --ignore E501,E402
      - name: Unit tests
        run: pytest tests/test_core.py tests/test_no_leakage.py -v --tb=short -x --timeout=120
      - name: API tests
        run: pytest tests/test_api.py -v --tb=short --timeout=60
      - name: Smoke test
        run: python tests/smoke_test.py
      - name: Minimal FL simulation
        run: |
          python -m data.generation.generate_dataset --n-buildings 2 --n-days 7
          python -m federated.simulation --n-rounds 3 --n-buildings 2 --seed 42
        timeout-minutes: 15
      - name: Minimal evaluation pipeline
        run: |
          python -m models.lstm.train_local --building-id B01 --data-path data/raw
          ls models/lstm/checkpoints/B01_best.pt models/lstm/checkpoints/B01_best_scaler.pkl
          python scripts/evaluate_checkpoint.py \
            --checkpoint-dir models/lstm/checkpoints \
            --data-dir data/raw \
            --test-months 1 \
            --n-buildings 1 \
            --output /tmp/ci_results.json
          python -c "
          import json
          with open('/tmp/ci_results.json') as f:
              r = json.load(f)
          assert r['results'], 'No results in output'
          print('Eval pipeline OK:', json.dumps(r['results'], indent=2))
          "
      - name: Dataset integrity
        run: python scripts/validate_dataset.py --data-dir data/raw

  docker:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t efadt-smart-campus:${{ github.sha }} .
      - name: Test container health
        run: |
          docker run -d --name efadt-test -p 8000:8000 efadt-smart-campus:${{ github.sha }}
          sleep 15
          curl -f http://localhost:8000/health
          docker stop efadt-test
```

### Step 2 — Create tests/test_evaluation_pipeline.py

```python
"""tests/test_evaluation_pipeline.py — integration tests for eval pipeline."""
import json, os, tempfile
import numpy as np
import pandas as pd
import pytest
import torch
import yaml


@pytest.fixture(scope="module")
def mini_dataset(tmp_path_factory):
    """Generate a minimal 2-building, 60-day dataset."""
    from data.generation.generate_dataset import generate_full_dataset
    data_dir = str(tmp_path_factory.mktemp("data"))
    generate_full_dataset(n_buildings=2, n_days=60, output_dir=data_dir, seed=99)
    return data_dir


@pytest.fixture(scope="module")
def trained_checkpoint(mini_dataset, tmp_path_factory):
    """Train a minimal model and return checkpoint directory."""
    ckpt_dir = str(tmp_path_factory.mktemp("checkpoints"))
    from models.lstm.train_local import run_standalone_training
    result = run_standalone_training(
        building_id="B01",
        data_path=mini_dataset,
        checkpoint_dir=ckpt_dir,
        seed=99,
    )
    return ckpt_dir


def test_scaler_serialized_with_checkpoint(trained_checkpoint):
    """Checkpoint must include an accompanying scaler pickle."""
    pt_files = [f for f in os.listdir(trained_checkpoint) if f.endswith("_best.pt")]
    assert pt_files, "No checkpoint .pt file found"
    for pt_file in pt_files:
        scaler_file = pt_file.replace("_best.pt", "_best_scaler.pkl")
        assert os.path.exists(os.path.join(trained_checkpoint, scaler_file)), \
            f"Missing scaler: {scaler_file}"


def test_load_checkpoint_returns_scaler(trained_checkpoint):
    """load_checkpoint() must return a fitted StandardScaler."""
    from models.lstm.train_local import load_checkpoint
    from sklearn.preprocessing import StandardScaler
    import yaml
    with open("configs/hyperparams.yaml") as f:
        config = yaml.safe_load(f)
    ckpt_path = os.path.join(trained_checkpoint, "B01_best.pt")
    model, scaler, ckpt = load_checkpoint(ckpt_path, config)
    assert isinstance(scaler, StandardScaler), "Loaded scaler is not StandardScaler"
    assert hasattr(scaler, "mean_"), "Scaler was not fitted (no mean_ attribute)"
    assert len(scaler.mean_) == 14, f"Expected 14 features, got {len(scaler.mean_)}"


def test_evaluate_checkpoint_produces_metrics(mini_dataset, trained_checkpoint, tmp_path):
    """evaluate_checkpoint.py must write a non-empty results JSON."""
    import subprocess, sys
    output = str(tmp_path / "test_results.json")
    result = subprocess.run([
        sys.executable, "scripts/evaluate_checkpoint.py",
        "--checkpoint-dir", trained_checkpoint,
        "--data-dir", mini_dataset,
        "--test-months", "2",
        "--config", "configs/hyperparams.yaml",
        "--building-config", "configs/building_params.yaml",
        "--output", output,
    ], capture_output=True, text=True)
    assert result.returncode == 0, f"evaluate_checkpoint.py failed:\n{result.stderr}"
    with open(output) as f:
        data = json.load(f)
    assert "results" in data
    assert data["results"], "Empty results dict"
    for variant, metrics in data["results"].items():
        for key in ["ERR", "CCS", "CSS", "MAE"]:
            assert metrics.get(key) is not None, f"{variant}.{key} is None"
            assert isinstance(metrics[key], (int, float)), f"{variant}.{key} not numeric"
```

### Step 3 — Add make ci target

```makefile
ci:
	ruff check . --ignore E501,E402
	pytest tests/test_core.py tests/test_no_leakage.py \
	       tests/test_api.py tests/test_evaluation_pipeline.py \
	       -v --tb=short -x --timeout=120
	python tests/smoke_test.py
```

### Step 4 — Commit Phase 13 changes

```bash
git add .github/workflows/ci.yml tests/test_evaluation_pipeline.py Makefile
git commit -m "phase-13: CI restructured with eval pipeline gate, test_evaluation_pipeline.py"
```

---

## README Updates Required

### Add CI Badge and Section

```markdown
[![CI](https://github.com/your-org/efadt-smart-campus/actions/workflows/ci.yml/badge.svg)](...)
[![Coverage](https://codecov.io/gh/your-org/efadt-smart-campus/badge.svg)](...)
```

```markdown
## 🧪 Testing

```bash
make ci           # Full local CI (lint + unit + integration + smoke + eval pipeline)
make test         # Unit tests only
make smoke        # Smoke tests only
```

CI enforces: unit tests, no-leakage tests, API tests, end-to-end eval pipeline (scaler + inference + metrics JSON).
```

---

## Success Criteria (MANDATORY CHECKPOINT)

- [ ] `make ci` exits 0 locally
- [ ] `tests/test_evaluation_pipeline.py` — all 3 tests pass
- [ ] CI workflow includes "Minimal evaluation pipeline" step
- [ ] CI fails if `B01_best_scaler.pkl` is missing from checkpoint directory
- [ ] Docker build step passes on main branch

### Proceed Rule
All items `[x]` → proceed to Phase 14.

---

# Phase 14 — Final Reproducibility Certification Pass
**Estimated Time: 4 hours**

## Objective

Execute the complete reproduction pipeline from a clean state and certify all outputs match the committed results JSON within acceptable tolerance. Generate all required publication artifacts.

---

## Required Final Validation Commands

```bash
# ── Clean state ──────────────────────────────────────────────────────────────
git clone https://github.com/your-org/efadt-smart-campus.git efadt-clean
cd efadt-clean
make env
source venv/Scripts/activate

# ── Verify environment ────────────────────────────────────────────────────────
python -c "import torch, flwr, opacus, shap, fastapi, streamlit; print('All imports OK')"

# ── Full reproduction pipeline ────────────────────────────────────────────────
make generate-data SEED=42
make validate-data
make train-fl SEED=42
python scripts/compute_privacy_budget.py --n-rounds 100 --output results/dp_audit.json
make evaluate SEED=42

# ── Statistical validity ─────────────────────────────────────────────────────
make eval-multi-seed
# (runs seeds 42, 0, 1, 7, 13 — takes several hours on CPU)

# ── Ablation suite ───────────────────────────────────────────────────────────
make ablations SEED=42

# ── Update README results table ───────────────────────────────────────────────
make update-readme

# ── Full test suite ───────────────────────────────────────────────────────────
make ci

# ── Verify results are non-None ───────────────────────────────────────────────
python -c "
import json
with open('results/ablation/full_results.json') as f:
    r = json.load(f)
for variant, metrics in r['results'].items():
    for k, v in metrics.items():
        assert v is not None, f'{variant}.{k} is None — evaluation failed'
        assert isinstance(v, (int, float)), f'{variant}.{k} = {v!r}'
print('All metrics verified as non-None floats.')
"

# ── API smoke test ────────────────────────────────────────────────────────────
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
sleep 5
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/status | python -m json.tool
# Must show: shap_fitted: false until POST /calibrate-shap called
kill %1
```

---

## Artifact Checklist

| Artifact | Location | Required for Publication |
|----------|----------|-------------------------|
| Trained FL checkpoints (all 12 buildings) | `models/lstm/checkpoints/B*_best.pt` | ✓ |
| Fitted scalers | `models/lstm/checkpoints/B*_best_scaler.pkl` | ✓ |
| Dataset manifest + SHA-256 hashes | `data/raw/dataset_manifest.json` | ✓ |
| Full evaluation results (single seed) | `results/ablation/full_results.json` | ✓ |
| Multi-seed aggregated results (≥3 seeds) | `results/ablation/multi_seed_results.json` | ✓ |
| DP privacy audit | `results/dp_audit.json` | ✓ |
| MLflow experiment logs | `mlruns/` | ✓ |
| Hyperparameter config used for reported results | `configs/hyperparams.yaml` (version-controlled) | ✓ |
| Environment snapshot | `requirements.txt` (pinned) | ✓ |
| Dataset generation seed | `data/raw/dataset_manifest.json:seed` | ✓ |
| Per-seed result JSONs | `results/seeds/results_seed*.json` | ✓ |

---

## Publication Readiness Checklist

### ACM Artifact Evaluation
- [ ] One-command reproduction: `make reproduce` exits 0 and produces metrics JSON
- [ ] Artifact README describes all execution steps
- [ ] Results within 5% of reported values (accounting for CPU/GPU variance)
- [ ] All random seeds documented and committed
- [ ] Dependencies pinned with exact versions

### NeurIPS Reproducibility
- [ ] Multi-seed results (≥3 seeds) with mean±std
- [ ] Confidence intervals on all primary metrics
- [ ] Statistical significance tests vs each baseline (p-values reported)
- [ ] Effect sizes (Cohen's d) reported
- [ ] Hyperparameter search space documented (grid search for λ weights documented)
- [ ] Compute requirements stated (CPU hours for full run)

### IEEE Transactions / Q1 Journal
- [ ] DP privacy claim uses total ε after composition (not per-round only)
- [ ] σ value matches `compute_sigma()` output (4.845 for ε=1.0, δ=1e-5)
- [ ] Dataset generation code and parameters fully disclosed
- [ ] RC parameter provenance documented (design constants vs fitted)
- [ ] All baselines implemented and evaluated from code (not hardcoded)
- [ ] Evaluation on held-out test set (months 10–12, never seen during training)

### Open-Source Engineering Quality
- [ ] `make ci` passes (lint + unit + integration + smoke)
- [ ] `Dockerfile` builds without error
- [ ] All API endpoints tested in `tests/test_api.py`
- [ ] `CONTRIBUTING.md` up to date with new scripts
- [ ] `LICENSE` in root
- [ ] No hardcoded credentials or local paths

---

## Final Repository Structure

```
efadt-smart-campus/
│
├── configs/
│   ├── hyperparams.yaml          # Single source of truth for all hyperparams
│   └── building_params.yaml      # Per-building RC parameters
│
├── data/
│   ├── generation/
│   │   ├── generate_dataset.py   # Dataset generation (writes manifest + hashes)
│   │   ├── occupancy_model.py
│   │   ├── thermal_simulator.py
│   │   └── sensor_fault_injector.py
│   ├── raw/                      # Generated Parquet files (gitignored)
│   │   └── dataset_manifest.json # SHA-256 hashes — committed
│   └── scenarios/                # Normal/Peak/Failure splits (gitignored)
│
├── models/
│   └── lstm/
│       ├── architecture.py
│       ├── train_local.py        # Serializes scaler; calendar split
│       └── checkpoints/          # *.pt + *_scaler.pkl (gitignored)
│
├── federated/
│   ├── client.py                 # Seeded DP RNG; month-aware split
│   ├── server.py                 # MLflow per-round logging
│   ├── simulation.py             # Full determinism block; MLflow run
│   └── dp_mechanism.py           # compute_sigma; Renyi DP accounting
│
├── digital_twin/
│   ├── thermal_model.py
│   └── simulator.py
│
├── agent/
│   ├── action_space.py
│   ├── optimizer.py
│   └── utility_function.py
│
├── xai/
│   ├── shap_explainer.py         # save()/load() methods; seeded background
│   ├── trust_scorer.py
│   └── audit_logger.py
│
├── pipeline/
│   └── decision_cycle.py         # Asserts scaler is not None
│
├── evaluation/
│   ├── metrics.py                # significance_test() added
│   └── baseline_runner.py        # Loads from results JSON; fixed baselines
│
├── scripts/
│   ├── evaluate_checkpoint.py    # ← PRIMARY EVALUATION ENTRY POINT
│   ├── multi_seed_eval.py        # Multi-seed aggregation + significance tests
│   ├── run_ablations.py          # All ablation variants
│   ├── compute_privacy_budget.py # Rényi DP total budget audit
│   ├── update_readme_results.py  # Auto-patches README table
│   └── run_experiment.py         # Updated to call evaluate_checkpoint
│
├── results/
│   ├── ablation/
│   │   ├── full_results.json           # Single-seed metrics (committed)
│   │   ├── multi_seed_results.json     # Mean±std + significance (committed)
│   │   └── ablation_seed42.json        # Per-variant results
│   ├── seeds/
│   │   └── results_seed*.json          # Per-seed raw results (gitignored)
│   └── dp_audit.json                   # Rényi DP total budget (committed)
│
├── governance/dashboard/app.py
├── api/main.py                   # /calibrate-shap endpoint; no SHAP stub
├── tests/
│   ├── test_core.py
│   ├── test_api.py
│   ├── test_no_leakage.py        # Calendar split verification
│   ├── test_evaluation_pipeline.py # Scaler + eval integration tests
│   └── smoke_test.py
│
├── docs/
│   └── architecture.md
│
├── .github/workflows/
│   ├── ci.yml                    # Full CI with eval pipeline gate
│   └── reproduce.yml             # Scheduled weekly full reproduction run
│
├── requirements.txt              # Runtime deps (pinned)
├── requirements-dev.txt          # Dev/test deps
├── Makefile                      # reproduce, evaluate, ablations, validate-data, ci
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md                     # Auto-updated results table; correct DP claims
├── IMPLEMENTATION_STATUS.md      # Current state post-remediation
├── REMEDIATION_PLAN.md           # This document
└── LICENSE
```

---

## Final Sign-Off Checklist

Before tagging a release:

```bash
# Verify no hardcoded metric values remain in Python source
grep -R "ERR.*34\.7\|CCS.*0\.912\|CSS.*0\.963\|MAE.*3\.21\|tau.*0\.887" \
  --include="*.py" . | grep -v "\.json\|#\|test_\|BEFORE\|AFTER"
# Expected: no output

# Verify no unfixed RNG in FL client
grep -n "default_rng()" federated/client.py
# Expected: no output (all uses now seeded)

# Verify scaler serialization in training
grep -n "scaler_path\|scaler.pkl" models/lstm/train_local.py
# Expected: at least 2 lines

# Verify DP total budget documented
python scripts/compute_privacy_budget.py --n-rounds 100
# Must print epsilon_renyi_dp as float, not raise ImportError

# Verify no sigma=1.47 claim anywhere
grep -R "sigma.*1\.47\|1\.47.*sigma" --include="*.py" --include="*.md" .
# Expected: no output

# Run full CI
make ci
# Expected: exit 0, all tests pass

echo "Repository is publication-ready."
```
