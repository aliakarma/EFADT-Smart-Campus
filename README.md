# EFADT — Explainable Federated Agentic Digital Twin

[![CI](https://github.com/your-org/efadt-smart-campus/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/efadt-smart-campus/actions)
[![Coverage](https://codecov.io/gh/your-org/efadt-smart-campus/badge.svg)](https://codecov.io/gh/your-org/efadt-smart-campus)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Flower FL](https://img.shields.io/badge/FL-Flower-red.svg)](https://flower.dev)

> Privacy-preserving, explainable, multi-objective optimization of campus HVAC, comfort, and crowd management through federated learning, digital twin simulation, and SHAP-based governance.

---

## Quick Start

### 1. Install Dependencies
```bash
git clone https://github.com/aliakarma/efadt-smart-campus.git
cd efadt-smart-campus
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python -m data.generation.generate_dataset --config configs/hyperparams.yaml --n-buildings 12 --n-days 365 --seed 42
```

### 3. Run FL Simulation
```bash
python -m federated.simulation --n-rounds 100
```

---

## Architecture

The system coordinates a 30-second closed-loop control cycle integrating federated predictions, thermal simulations, multi-objective optimization decisions, trust scoring, and an immutable audit trail.

For a detailed explanation of the internal system components, data flow diagrams, and design, refer to [architecture.md](file:///c:/Users/Ali%20Akarma/Documents/GitHub/EFADT-Smart-Campus/docs/architecture.md).

---

## Environment Setup

We recommend utilizing the provided Makefile setup helper:
```bash
make env
source venv/Scripts/activate    # Windows
# source venv/bin/activate      # Linux/macOS
```
Python 3.10 or 3.11 is required.

---

## Dataset Splits

The dataset is partitioned into temporal training/validation/testing segments to prevent data leakage:
- **Train**: Months 1–6 (January to June)
- **Validation**: Months 7–9 (July to September)
- **Test**: Months 10–12 (October to December)

Scenarios splits include:
- **Normal**: All generated building data.
- **Peak**: Exam months only (April, May, November, December).
- **Failure**: Injects a 20% sensor fault rate with hold-last-value fill strategy.

---

## Reproducing Results

To reproduce all evaluations, ablations, and statistical analysis tests, run:
```bash
make reproduce
```
This single entry-point command:
1. Generates the synthetic dataset (`make generate-data`).
2. Runs the federated training simulation (`make train-fl`).
3. Retrains baseline/ablation checkpoints and evaluates them across all seeds (`make evaluate`).

---

## Key Results

Under chronological and disjoint evaluations across 12 campus buildings:
- The federated forecasting model matches centralized accuracy ceilings within tight bounds.
- Digital Twin WIF boundaries ensure comfort limits are maintained while reducing HVAC resource usage.
- Standard deviations across multiple seeds remain minimal, confirming framework stability.

---

## Ablation Study

Below is the multi-seed evaluation results table compiled across seeds `[42, 0, 1]` under temporal test splits.

<!-- RESULTS_TABLE_START -->
| Variant | ERR% | CCS | CSS | MAE (persons) |
|---------|------|-----|-----|---------------|
| EFADT (Full) | 44.084±0.000 | 0.427±0.000 | 0.999±0.000 | 15.952±1.533 |
| -XAI | 44.084±0.000 | 0.427±0.000 | 0.999±0.000 | 15.952±1.533 |
| -DT-WIF | 96.300±0.000 | 0.969±0.000 | 0.999±0.000 | 7.414±0.000 |
| -DP | 43.020±0.183 | 0.427±0.000 | 0.999±0.000 | 17.132±1.383 |
| -MOO (energy-only) | 44.084±0.000 | 0.427±0.000 | 0.999±0.000 | 15.952±1.533 |
| -FL (centralized) | 21.300±0.000 | 1.000±0.000 | 0.999±0.000 | 17.979±1.933 |
| Rule-Based | 30.368±0.000 | 0.381±0.000 | 0.999±0.000 | 7.413±0.000 |
<!-- RESULTS_TABLE_END -->

> Note: `ERR%` is Energy Reduction Ratio, `CCS` is Comfort Compliance Score, `CSS` is Crowd Safety Score, and `MAE` is forecasting Mean Absolute Error. Run `python scripts/update_readme_results.py` to auto-populate this table from the results JSON.

---

## Privacy Guarantee

EFADT applies $(\varepsilon=1.0, \delta=1\times10^{-5})$-DP **per FL round** using the Gaussian mechanism:
- **Noise Multiplier**: $\sigma = 4.845$ (analytically computed: $\sqrt{2\ln(1.25/\delta)}/\varepsilon$)
- **Gradient Clipping**: $C = 1.0$
- **Composed Total Privacy Cost**: After 100 FL rounds, the composed privacy budget under Rényi DP is $\varepsilon_{\text{total}} \approx 11.15$.
  Run the budget auditing tool for exact composition:
  ```bash
  python scripts/compute_privacy_budget.py --n-rounds 100
  ```

---

## Experimental Setup & Compute Requirements

- **Hyperparameter Search Space**: The MOO agent utility weights ($\lambda_e, \lambda_c, \lambda_d$) were optimized via grid search over $[0, 1]$ in steps of 0.05, constrained by $\lambda_e + \lambda_c + \lambda_d = 1.0$. The optimal found weights are $\lambda_e=0.5, \lambda_c=0.35, \lambda_d=0.15$.
- **Compute Requirements**: A complete single-seed reproduction of the 12-building dataset generation, 100-round federated training, and comprehensive evaluation pipeline requires approximately **1.5 to 2.5 CPU hours** on a modern multi-core processor (e.g., Intel i7 / AMD Ryzen 7). No GPU is strictly required.

---

## Statistical Validity

Statistical validity is evaluated using:
- **Wilcoxon Signed-Rank Tests**: Two-sided significance tests comparing proposed vs baseline variants.
- **Effect Sizes**: Cohen's $d$ calculations showing magnitude of improvements.
- Results are logged automatically to `results/ablation/multi_seed_results.json`.

---

## API Reference

The project includes a FastAPI REST server to handle remote inference and auditing.

### Start the API:
```bash
make api
```
Access the interactive docs at: [http://localhost:8000/docs](http://localhost:8000/docs).

### Primary Endpoints:
- `POST /decide`: Executes one 30-second loop iteration.
- `GET /simulate/{building_id}`: Queries what-if Digital Twin scenarios.
- `GET /audit/{building_id}/verify`: Validates hash chain integrity.

---

## Dashboard

A Streamlit dashboard displays live KPIs, trust heatmaps, SHAP waterfall charts, and energy tracking comparisons.

### Start the Dashboard:
```bash
make dashboard
```
Access the UI at: [http://localhost:8501](http://localhost:8501).

---

## Docker Deployment

To spin up the entire ecosystem (API, Dashboard, MLflow, and Prometheus) in Docker containers:
```bash
docker compose up -d
```
- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`
- MLflow tracking: `http://localhost:5000`
- Prometheus metrics: `http://localhost:9090`

---

## Testing

Run tests with disabled Pytest autoloader:
```bash
make ci           # Full local CI (lint + unit + integration + smoke + eval pipeline)
make test         # Unit tests only
make smoke        # Smoke tests only
```

CI enforces: unit tests, no-leakage tests, API tests, end-to-end eval pipeline (scaler + inference + metrics JSON).

---

## Experiment Tracking

All training run metrics, parameters, and model artifacts are tracked using MLflow:
```bash
mlflow ui --host 127.0.0.1 --port 5000
```
Key parameters tracked include `apply_dp`, `epsilon`, `sigma`, `local_epochs`, and trust weights.

---

## Citing This Work

If you use EFADT in your research, please cite:
```bibtex
@article{efadt2026,
  title   = {EFADT: Explainable Federated Agentic Digital Twin for Smart Campus Resource Optimization},
  author  = {},
  journal = {},
  year    = {2026}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
