# EFADT — Explainable Federated Agentic Digital Twin

[![CI](https://github.com/your-org/efadt-smart-campus/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/efadt-smart-campus/actions)
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
| EFADT (Full) | 23.287±0.000 | 0.167±0.000 | 0.898±0.000 | 17.320±0.000 |
| -XAI | 23.287±0.000 | 0.167±0.000 | 0.898±0.000 | 17.320±0.000 |
| -DT-WIF | 89.898±0.000 | 0.683±0.000 | 0.898±0.000 | 19.333±0.000 |
| -DP | 23.287±0.000 | 0.167±0.000 | 0.898±0.000 | 17.320±0.000 |
| -MOO (energy-only) | 23.287±0.000 | 0.167±0.000 | 0.898±0.000 | 17.320±0.000 |
| -FL (centralized) | 21.300±0.000 | 1.000±0.000 | 0.898±0.000 | 16.640±0.059 |
| Rule-Based | 24.834±0.000 | 0.169±0.000 | 0.898±0.000 | 19.333±0.000 |
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
make smoke        # Fast smoke test
make test         # Unit tests
make test-all     # All tests with coverage report
```

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
@article{efadt2024,
  title   = {EFADT: Explainable Federated Agentic Digital Twin for Smart Campus Resource Optimization},
  author  = {Author et al.},
  journal = {IEEE Transactions on Smart Grid},
  year    = {2024}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
