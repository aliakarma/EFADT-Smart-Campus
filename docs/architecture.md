# EFADT System Architecture

This document describes the architectural design, data flows, and module interactions of the **Explainable Federated Agentic Digital Twin (EFADT)** framework.

---

## 🔄 The 30-Second Closed Loop

EFADT operates in a continuous, 30-second decision cycle designed to optimize building resource consumption (HVAC power), thermal comfort, indoor air quality ($CO_2$), and crowd safety.

```
IoT Sensors → FL-LSTM Forecast → Digital Twin Simulation → MOO Agent → HVAC Actuation
                     ↓                                            ↑
             SHAP Explainer ←──── Trust Score ←────────────────────
                     ↓
              Audit Ledger (hash-chained, tamper-evident)
```

1. **IoT Ingestion**: Sensor readings (occupancy, temperature, $CO_2$, humidity) are polled every 30 seconds.
2. **FL-LSTM Forecast**: The local federated LSTM model forecasts the next-step occupancy.
3. **Digital Twin Simulation**: An RC thermal model simulates the next-step thermal state, comfort compliance, and energy use for all possible control candidates.
4. **MOO Agent**: Discretized actions are evaluated using a multi-objective utility function, selecting the optimal trade-off candidate.
5. **SHAP Proxy Explainer**: A fast gradient-boosted tree proxy fits local LSTM predictions to compute SHAP values for the decision.
6. **Trust Score**: Coherence between SHAP explanations, comfort compliance, and safety bounds determines the trust score ($\tau$).
7. **Audit Ledger**: The state, decision, explanation, and trust metrics are written to a tamper-evident SQLite/JSONL hash chain.

---

## 🏗️ Component Breakdown

```
efadt-smart-campus/
├── federated/             # Federated Learning (FL) with differential privacy
├── digital_twin/          # RC thermal simulation & Comfort evaluation
├── agent/                 # Decision optimization & Multi-objective utility
├── xai/                   # Explainability & Trust verification
└── pipeline/              # Loop orchestration
```

### 1. Federated Forecasting (`federated/`)
- **OccupancyLSTM**: A 2-layer LSTM with 128 hidden units. It receives lagged occupancy and cyclical time encodings to predict next-interval occupancy.
- **Differential Privacy**: Uses a Gaussian perturbation mechanism. Local gradients are clipped to $C = 1.0$ and perturbed with dynamically computed noise ($\sigma = 4.845$ for $\varepsilon=1.0, \delta=1\times10^{-5}$ per round).
- **Flower FL Framework**: Organizes the simulation across 12 building clients. Models are aggregated on the server using FedAvg.

### 2. Digital Twin Simulator (`digital_twin/`)
- **First-Order RC Model**: Simulates indoor temperature $T_{\text{in}}$ based on external temperature $T_{\text{out}}$, occupancy $o$, and HVAC heat exchange $Q$:
  $$\frac{dT_{\text{in}}}{dt} = \alpha (T_{\text{out}} - T_{\text{in}}) + \beta Q + \gamma \kappa o$$
- **Comfort Band**: Defines standard boundaries for temperature ($[21.0, 24.0]^\circ\text{C}$) and $CO_2$ ($<1000\text{ ppm}$).
- **Action Scorer**: Simulates candidate actions over a 6-step lookahead horizon ($H=6$).

### 3. Multi-Objective Optimization Agent (`agent/`)
- **Action Space**: Discretized into $N$ candidate levels of cooling/heating capacity along with class-reassignment flags.
- **Utility Function**: Evaluates candidate actions under three weighted goals:
  $$u^* = \arg\min_{u \in U_{\text{feas}}} \left[ \lambda_e E(u) - \lambda_c C(u) + \lambda_d D(u) \right]$$
  where $E(u)$ is normalized energy cost, $C(u)$ is comfort compliance, and $D(u)$ is density risk.

### 4. Explainability & Governance (`xai/`)
- **SHAP Explainer**: Fits a local tree-based proxy model on rolling sensor predictions to explain which input features (e.g. current occupancy, outdoor temperature) drove the forecast.
- **Trust Scorer**: Computes decision trust ($\tau$) by validating if SHAP explanations match physical constraints and safety violations.
- **Audit Logger**: Appends each decision cycle to a sequential log where the current block contains the SHA-256 hash of the previous block, creating a tamper-evident blockchain-like audit trail.
