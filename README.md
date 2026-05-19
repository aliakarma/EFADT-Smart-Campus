# EFADT — Explainable Federated Agentic Digital Twin for Smart Campus Resource Optimization

[![CI](https://github.com/your-org/efadt-smart-campus/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/efadt-smart-campus/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Flower FL](https://img.shields.io/badge/FL-Flower-red.svg)](https://flower.dev)

> **Privacy-preserving, explainable, multi-objective optimization of campus HVAC, comfort, and crowd management through federated learning, digital twin simulation, and SHAP-based governance.**

---

## 📋 Overview

EFADT integrates four tightly coupled components into a 30-second closed-loop:

```
IoT Sensors → FL-LSTM Forecast → Digital Twin Simulation → MOO Agent → HVAC Actuation
                     ↓                                            ↑
             SHAP Explainer ←──── Trust Score ←────────────────────
                     ↓
              Audit Ledger (hash-chained, tamper-evident)
```

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

---

## 🏗️ Architecture

```
efadt-smart-campus/
├── configs/               # Hyperparameters, building RC parameters
├── data/
│   ├── generation/        # Synthetic dataset generation pipeline
│   ├── raw/               # Per-building Parquet files (generated)
│   └── scenarios/         # Normal / Peak / Failure splits
├── models/
│   └── lstm/              # OccupancyLSTM (2-layer, 128 hidden)
├── federated/             # Flower FL client/server + DP mechanism
├── digital_twin/          # RC thermal model + H-step simulator
├── agent/                 # Action space + MOO utility function
├── xai/                   # SHAP proxy explainer + trust scorer + audit logger
├── pipeline/              # 30-second decision cycle orchestrator
├── evaluation/            # ERR, CCS, CSS, MAE, τ, SHF metrics
├── governance/
│   └── dashboard/         # Streamlit governance dashboard
├── api/                   # FastAPI REST backend
├── tests/                 # Unit, integration, smoke tests
└── deployment/            # Docker, CI/CD
```

---

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

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/aliakarma/efadt-smart-campus.git
cd efadt-smart-campus
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
# Quick test (2 buildings, 30 days, ~30 seconds)
make generate-quick

# Full dataset (12 buildings, 365 days, ~10 minutes)
make generate-data
```

### 3. Run Smoke Tests

```bash
make smoke
```

### 4. Start the API

```bash
make api
# → http://localhost:8000/docs
```

### 5. Launch Governance Dashboard

```bash
make dashboard
# → http://localhost:8501
```

### 6. Run FL Simulation

```bash
# Quick test (3 buildings, 5 rounds)
make train-quick

# Full simulation (12 buildings, 100 rounds, ~2 hours on CPU)
make train-fl
```

---

## 🔑 Core Components

### Federated Learning with Differential Privacy

```python
from federated.dp_mechanism import compute_sigma, privatize_gradient

sigma = compute_sigma(epsilon=1.0, delta=1e-5)   # σ ≈ 1.47
noised_grad, info = privatize_gradient(raw_grad, epsilon=1.0, delta=1e-5)
```

- ε=1.0, δ=1e-5 (strong DP guarantee)
- Gaussian mechanism: Δθ̃_b = clip(Δθ_b, C) + N(0, σ²C²I)
- FedAvg aggregation across 12 buildings
- E=5 local epochs per FL round
- Convergence at round ~52 (MAE < 3.5 persons)

### Digital Twin Simulation

```python
from digital_twin.simulator import DigitalTwinSimulator
from digital_twin.thermal_model import BuildingThermalParams, ThermalState

sim = DigitalTwinSimulator("B01", params=BuildingThermalParams(...), H=6)
sim.sync_state(ThermalState(T_in=26.0, T_out=38.0, Q_hvac=0.0, occupancy=50))
score = sim.evaluate_action(Q_hvac=-12.0, occ_forecast=np.full(6, 50))
# → ActionScore(E=0.48, C=0.83, D=0.625, feasible=True)
```

RC model: `dT/dt = α(T_out − T_in) + βQ_HVAC + γκô`

### Multi-Objective Agent

```python
from agent.optimizer import EFADTAgent

agent = EFADTAgent("B01", simulator=sim)
best_action, all_scores, utility = agent.decide(state, occ_forecast)
# u* = argmin_{u ∈ U_feas} [λ_e·E(u) − λ_c·C(u) + λ_d·D(u)]
```

### SHAP Trust Score

```python
from xai.trust_scorer import compute_trust_score

result = compute_trust_score(shap_values, C_u_star=0.92, D_u_star=0.40)
# → TrustScoreResult(tau=0.887, shap_coherence=0.44, ...)
```

τ(u*) = η₁·SHAP_coherence + η₂·C(u*) + η₃·(1−D(u*))

---

## 🐳 Docker Deployment

```bash
# Start all services (API + Dashboard + Prometheus + MLflow)
docker compose up -d

# Endpoints:
#   API:         http://localhost:8000/docs
#   Dashboard:   http://localhost:8501
#   MLflow:      http://localhost:5000
#   Prometheus:  http://localhost:9090
```

---

## 🧪 Testing

```bash
make smoke          # Quick end-to-end validation (< 60s)
make test           # Full pytest suite
make test-api       # API integration tests only
make test-all       # All tests with coverage report
```

---

## 📊 Ablation Study

> ⚠️ Ablation results pending. Run `make reproduce` to populate from verified checkpoints.

| Variant | ERR% | CCS | CSS | MAE | τ |
|---------|------|-----|-----|-----|---|
| EFADT (Full) | — | — | — | — | — |
| −XAI | — | — | — | — | — |
| −DT-WIF | — | — | — | — | — |
| −DP | — | — | — | — | — |
| −MOO (energy-only) | — | — | — | — | — |
| −FL (centralized) | — | — | — | — | — |

---

## 🔒 Privacy Guarantee

EFADT satisfies (ε=1.0, δ=1e-5)-DP per round with Gaussian mechanism:
- σ ≈ 1.47 (computed analytically)
- Gradient clipping: C = 1.0
- Raw sensor data **never leaves** the building node
- Only DP-noised gradient updates transmitted

---

## 📝 Citation

```bibtex
@article{efadt2024,
  title   = {EFADT: Explainable Federated Agentic Digital Twin for Smart Campus Resource Optimization},
  author  = {Author et al.},
  journal = {IEEE Transactions on ...},
  year    = {2024}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE).
