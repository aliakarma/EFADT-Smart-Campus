# Repository Remediation & Hardening Plan

## 1. Executive Summary

**Current Repository Maturity Assessment:**
The EFADT-Smart-Campus repository exhibits a severe dichotomy. The software engineering infrastructure—featuring FastAPI, Streamlit, Docker, and Pytest—is at an industry-standard production tier. However, the scientific integrity is critically compromised. The repository currently operates as a simulation of an experiment rather than a true empirical study. Primary evaluation metrics, ablation tables, and comparative baselines are strictly hardcoded as literal values. The training pipeline is completely unlinked from the evaluation pipeline.

**Main Blockers Preventing Publication/Reproducibility:**

1. Fabricated, hardcoded metrics in `evaluation/baseline_runner.py` (`PAPER_RESULTS`).
2. A completely disconnected execution pipeline (`scripts/run_experiment.py` trains the FL model but abandons the weights, printing the fabricated table instead).
3. Explicit reverse-engineering of target metrics using targeted random normal distributions in `evaluation/metrics.py`.
4. Lack of empirical baselines (baselines are not actually executed against a test set).
5. No statistical validity (reported results do not span multiple random seeds).

**Estimated Total Remediation Time:** 35–50 hours.

**Expected Final Outcome:**
A repository that can cleanly execute a full empirical pipeline—data generation, training, model evaluation, and baseline comparison—yielding dynamically computed, mathematically valid metrics across multiple seeds. This will elevate the repository from a state of scientific fabrication to a state suitable for ACM Artifact Evaluation and IEEE/NeurIPS publication.

---

# Phase 1 — Pipeline Reconnection & Execution Integrity

Estimated Time: 8 hours

## Objective

Eradicate all hardcoded metrics and reverse-engineered evaluation stubs. Reconnect the trained Federated Learning LSTM global checkpoints directly to the evaluation script to ensure all reported metrics are strictly derived from actual model inference.

## Problems Addressed

* Fabricated ablation table (`PAPER_RESULTS`).
* Reverse-engineered metric generation.
* Disconnected `scripts/run_experiment.py`.

## Files To Modify

| File | Required Changes |
| --- | --- |
| `evaluation/baseline_runner.py` | Delete `PAPER_RESULTS` and `print_ablation_table`. Write `evaluate_efadt_model`. |
| `evaluation/metrics.py` | Delete the reverse-engineering code block under `__main__`. |
| `scripts/run_experiment.py` | Connect `step_run_fl` output to `step_evaluate`. |

## Step-by-Step Implementation Guide

### Step 1 — Eradicate Fabricated Metric Generators

Purpose: Remove the code actively faking the EFADT metrics.

Implementation:

```bash
# Remove the fake metric generator in evaluation/metrics.py
sed -i '/if __name__ == "__main__":/,$d' evaluation/metrics.py

# Remove the hardcoded ablation table in baseline_runner.py
sed -i '/PAPER_RESULTS = {/,/}/d' evaluation/baseline_runner.py
sed -i '/def print_ablation_table/,+12d' evaluation/baseline_runner.py

```

### Step 2 — Implement Genuine EFADT Evaluation

Purpose: Create a function that loads the saved FL global parameters, reconstructs the model, and evaluates the test split using the `DigitalTwinSimulator`.

Implementation:
**Code Changes in `evaluation/baseline_runner.py`:**

```python
# BEFORE (Non-existent)
# AFTER
import torch
import flwr as fl
import pickle
from models.lstm.architecture import build_model
from digital_twin.simulator import DigitalTwinSimulator

def evaluate_efadt_model(config_path: str, checkpoint_path: str, test_data_dict: dict) -> dict:
    """Loads a global FL checkpoint and dynamically computes metrics on test data."""
    # Load config and instantiate model
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device=device)
    
    # Load checkpoint parameters
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
    
    # Apply parameters to model
    params_dict = zip(model.state_dict().keys(), ckpt["parameters"])
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    
    # --- TODO: Implement evaluation loop over test_data_dict using model ---
    # 1. Run model.predict(X_test)
    # 2. Run outputs through DigitalTwinSimulator
    # 3. Compute metrics via evaluation/metrics.py:compute_all_metrics
    # Return EFADTMetrics object
    pass 

```

### Step 3 — Reconnect the Experiment Pipeline

Purpose: Ensure running the experiment actually passes the trained model to the evaluation step.

Implementation:
**Code Changes in `scripts/run_experiment.py`:**

```python
# BEFORE
def step_evaluate(n_buildings: int) -> None:
    logger.info("Step 3: Evaluating baselines and ablation variants...")
    from evaluation.baseline_runner import print_ablation_table, PAPER_RESULTS
    print_ablation_table()

# AFTER
def step_evaluate(n_buildings: int, checkpoint_path: str, test_data: dict) -> None:
    logger.info("Step 3: Dynamically evaluating EFADT model...")
    from evaluation.baseline_runner import evaluate_efadt_model
    metrics = evaluate_efadt_model("configs/hyperparams.yaml", checkpoint_path, test_data)
    logger.info(f"Dynamically Computed Metrics: {metrics}")

```

## README Updates Required

Add Section:

## Reproducibility

To reproduce the dynamic evaluation results from the trained model:

```bash
python scripts/run_experiment.py --full

```

*Note: The ablation table results reported below are dynamically computed via the pipeline above and will vary slightly based on hardware determinism and random seeds.*

Modify Existing Section:

* Remove the static, hardcoded metrics table from the README until Phase 4 completes and valid numbers are generated.

## Success Criteria (MANDATORY CHECKPOINT)

* [ ] `PAPER_RESULTS` dictionary is completely removed from the repository.
* [ ] `scripts/run_experiment.py` successfully loads a `.pkl` checkpoint from `models/lstm/checkpoints/` after training.
* [ ] The pipeline outputs a dynamically computed `EFADTMetrics` object.
* [ ] No `rng.normal` offsets are used to calculate the final `ERR` or `MAE`.
* [ ] README updated to reflect dynamic computation.

### Proceed Rule

* If ALL items are `[x]`, proceed to next phase
* Otherwise DO NOT continue to next phase
* Fix remaining items before advancing

---

# Phase 2 — Baseline Reimplementation & Fair Comparison

Estimated Time: 12 hours

## Objective

Replace the fabricated baseline claims with actual executable code. Implement the training and evaluation loops for the Centralized-NN, FL-Only, DT-Only, and Rule-Based variants using the exact same test dataset splits.

## Problems Addressed

* Missing baseline implementations.
* Unverifiable comparative claims in the abstract/README.

## Files To Modify

| File | Required Changes |
| --- | --- |
| `evaluation/baseline_runner.py` | Implement `train_centralized_nn`, `evaluate_rule_based`, `evaluate_dt_only`. |
| `scripts/run_experiment.py` | Add logic to iterate over all baseline functions and compile results. |

## Step-by-Step Implementation Guide

### Step 1 — Centralized-NN Baseline

Purpose: Train a single LSTM on horizontally pooled data to simulate the non-federated (-FL) ablation.

Implementation:

```bash
# In evaluation/baseline_runner.py, update train_centralized_nn

```

**Code Changes:**

```python
# BEFORE
# A stub exists but is never linked to the main experiment pipeline.

# AFTER
def run_centralized_baseline(train_data: dict, test_data: dict, cfg: dict) -> EFADTMetrics:
    # 1. Pool all DataFrames in train_data
    # 2. Fit single StandardScaler
    # 3. Train CentralizedLSTM
    # 4. Evaluate on pooled test_data
    # 5. Return compute_all_metrics(...)
    pass

```

### Step 2 — Implement Rule-Based Evaluation Over Test Set

Purpose: Evaluate the `rule_based_controller` stub dynamically over the generated test split.

Implementation:
**Code Changes:**

```python
# In evaluation/baseline_runner.py
# Ensure evaluate_rule_based actually iterates over the DataFrames loaded in scripts/run_experiment.py
# rather than accepting arbitrary np.ndarrays without a data loader.

```

### Step 3 — Compile Ablation Results Dynamically

Purpose: Aggregate the outputs of all dynamically run baselines into a single summary table.

Implementation:
**Code Changes in `scripts/run_experiment.py`:**

```python
# AFTER
def step_evaluate(n_buildings: int, checkpoint_path: str, train_data: dict, test_data: dict):
    results = {}
    # 1. EFADT
    results["EFADT"] = evaluate_efadt_model(..., checkpoint_path, test_data)
    # 2. Centralized
    results["Centralized"] = run_centralized_baseline(train_data, test_data, ...)
    # 3. Rule-Based
    results["Rule-Based"] = evaluate_rule_based_baseline(test_data)
    
    # Output markdown table
    print("| Variant | ERR% | CCS | CSS | MAE |")
    for name, res in results.items():
        print(f"| {name} | {res.ERR:.1f} | {res.CCS:.3f} | {res.CSS:.3f} | {res.MAE:.2f} |")

```

## README Updates Required

Modify Existing Section:

* Add a new "Baselines" sub-heading explicitly stating that the Centralized-NN, FL-Only, and DT-Only baselines are actively computed against the test split, not statically defined.

## Success Criteria (MANDATORY CHECKPOINT)

* [ ] `train_centralized_nn` successfully trains a model on pooled data.
* [ ] `evaluate_rule_based` executes without errors over the test dataset.
* [ ] `scripts/run_experiment.py` prints a dynamic Markdown table comparing at least 3 baselines.
* [ ] The metrics for baselines fluctuate appropriately when the random seed is changed.

### Proceed Rule

* If ALL items are `[x]`, proceed to next phase
* Otherwise DO NOT continue to next phase
* Fix remaining items before advancing

---

# Phase 3 — Statistical Validity Upgrades & Determinism

Estimated Time: 6 hours

## Objective

Enforce strict determinism across all environments and wrap the entire experimental pipeline in a multi-seed loop to generate statistically valid confidence intervals for all claims.

## Problems Addressed

* Results reported on a single, opaque seed.
* Lack of statistical significance testing.
* Uncontrolled CUDA/Numpy determinism.

## Files To Modify

| File | Required Changes |
| --- | --- |
| `scripts/run_experiment.py` | Add seed looping and statistical aggregation. |
| `models/lstm/train_local.py` | Enforce `torch.backends.cudnn.deterministic`. |

## Step-by-Step Implementation Guide

### Step 1 — Enforce Absolute Determinism

Purpose: Ensure that running the exact same seed produces the exact same PyTorch weights.

Implementation:
**Code Changes in `models/lstm/train_local.py` (and `architecture.py`):**

```python
# AFTER (Add to the top of the training execution block)
import torch
import numpy as np
import random
import os

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

```

### Step 2 — Multi-Seed Experiment Loop

Purpose: Run the entire pipeline across 5 independent seeds.

Implementation:
**Code Changes in `scripts/run_experiment.py`:**

```python
# BEFORE
def main():
    ...
    step_generate_data(n_buildings, n_days, args.seed)
    summary = step_run_fl(n_buildings, n_rounds, not args.no_dp, args.seed)
    step_evaluate(n_buildings)

# AFTER
def main():
    ...
    seeds = [42, 43, 44, 45, 46]
    all_results = { "EFADT": [], "Centralized": [] }
    
    for seed in seeds:
        logger.info(f"--- RUNNING EXPERIMENT FOR SEED {seed} ---")
        seed_everything(seed)
        step_generate_data(n_buildings, n_days, seed)
        summary = step_run_fl(n_buildings, n_rounds, not args.no_dp, seed)
        metrics_dict = step_evaluate(n_buildings, summary["checkpoint_path"], train_data, test_data)
        
        for variant, metrics in metrics_dict.items():
            all_results[variant].append(metrics)
            
    # Compute mean and std dev
    from evaluation.metrics import compute_metrics_with_confidence
    for variant, runs in all_results.items():
        agg = compute_metrics_with_confidence(runs)
        logger.info(f"{variant}: ERR={agg['ERR']['mean']:.1f}±{agg['ERR']['std']:.1f}")

```

## README Updates Required

Modify Existing Section:

* Update the results table to use the format `Mean ± StdDev`.
* Add a methodology note: *"All reported results are averaged over 5 independent random seeds (42-46) with strict CUDA determinism enforced."*

## Success Criteria (MANDATORY CHECKPOINT)

* [ ] `seed_everything()` is called before data generation and model initialization.
* [ ] The experiment runner executes the pipeline 5 times.
* [ ] Final output includes standard deviations for ERR, CCS, CSS, and MAE.
* [ ] Running the script twice with the same seed arrays yields bit-exact identical metric outputs.

### Proceed Rule

* If ALL items are `[x]`, proceed to next phase
* Otherwise DO NOT continue to next phase
* Fix remaining items before advancing

---

# Phase 4 — DP Tightening & Dependency Stabilization

Estimated Time: 4 hours

## Objective

Validate the Differential Privacy claims. The current Gaussian mechanism bound relies on basic composition which is mathematically loose. We must strictly integrate Opacus for Rényi DP accounting, and pin the environment properly.

## Problems Addressed

* `estimate_total_privacy_budget` relies on a mocked/loose implementation due to Opacus not being explicitly initialized in the FL loop.
* Potential environment drift in `requirements.txt`.

## Files To Modify

| File | Required Changes |
| --- | --- |
| `federated/dp_mechanism.py` | Force standard Opacus RDP accountant usage. |
| `requirements.txt` | Ensure specific compatible versions of `torch`, `flwr`, and `opacus`. |

## Step-by-Step Implementation Guide

### Step 1 — Pin Dependencies

Purpose: Ensure Opacus and Flwr compatibility.

Implementation:

```bash
# Rebuild environment
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
# Verify opacus is available
python -c "import opacus; print(opacus.__version__)"

```

### Step 2 — Tighten DP Accounting

Purpose: Remove the basic composition warning and utilize true Rényi DP.

Implementation:
**Code Changes in `federated/dp_mechanism.py`:**

```python
# BEFORE
def estimate_total_privacy_budget(..., composition="basic"):
    # defaults to loose bounding

# AFTER
def estimate_total_privacy_budget(epsilon_per_round: float, n_rounds: int, delta: float = 1e-5) -> float:
    from opacus.accountants import RDPAccountant
    accountant = RDPAccountant()
    sigma = compute_sigma(epsilon_per_round, delta)
    for _ in range(n_rounds):
        # Sample rate = 1.0 because all local data is used per round
        accountant.step(noise_multiplier=sigma, sample_rate=1.0)
    eps, _ = accountant.get_privacy_spent(delta=delta)
    return eps

```

## README Updates Required

Add Section:

## Privacy Guarantees

* Explicitly state that total privacy budget is calculated using Rényi Differential Privacy (RDP) via the `opacus` library.
* Provide the exact formula or command used to derive the final $\epsilon$ value.

## Success Criteria (MANDATORY CHECKPOINT)

* [ ] `requirements.txt` installs without resolution conflicts.
* [ ] `estimate_total_privacy_budget` successfully imports and utilizes `opacus.accountants.RDPAccountant`.
* [ ] The logged total $\epsilon$ after 100 rounds is tighter than the basic composition bound.

### Proceed Rule

* If ALL items are `[x]`, proceed to next phase
* Otherwise DO NOT continue to next phase
* Fix remaining items before advancing

---

# Phase 5 — Final Reproducibility Certification

Estimated Time: 4 hours

## Objective

Certify that the repository is end-to-end reproducible, scientifically valid, and passes all ACM Artifact Evaluation guidelines.

## Required Final Validation Commands

```bash
# 1. Clean the environment
make clean
rm -rf data/raw/*.parquet
rm -rf models/lstm/checkpoints/*.pkl

# 2. Re-install strictly from requirements
pip install -r requirements.txt

# 3. Run the full reproducibility pipeline (Data Gen -> FL Train -> DP Auth -> Eval -> Aggregation)
python scripts/run_experiment.py --full > final_experiment_log.txt

# 4. Verify output formats
grep "EFADT:" final_experiment_log.txt
grep "±" final_experiment_log.txt

```

## Artifact Checklist

* [ ] **Trained models**: Checkpoints dynamically saved to `models/lstm/checkpoints/`.
* [ ] **Logs**: `final_experiment_log.txt` contains full standard out.
* [ ] **Configs**: `hyperparams.yaml` directly controls `run_experiment.py`.
* [ ] **Seeds**: Explicitly logged per run (42, 43, 44, 45, 46).
* [ ] **Metrics**: Mean and StdDev dynamically printed to stdout.
* [ ] **Synthetic Data Limitations**: README contains a disclaimer that tests are run on synthetically generated RC-ODE thermal data, and provides API stubs for real IoT integration.

## Publication Readiness Checklist

* **ACM Artifact Evaluation readiness**: [Pass] All scripts executable via 1 command (`run_experiment.py --full`).
* **NeurIPS reproducibility readiness**: [Pass] Determinism enforced, seeds published, multi-run statistics available.
* **IEEE/Q1 journal readiness**: [Pass] Baseline methods implemented fairly and empirical data fully replaces hardcoded claims.

## Final Repository Structure

```text
efadt-smart-campus/
├── README.md                 # Updated with dynamic results & real instructions
├── requirements.txt          # Pinned
├── configs/
│   └── hyperparams.yaml
├── data/
│   └── generation/           # Generators
├── models/
│   └── lstm/                 # True inference and training
├── federated/                # Flwr client/server & Opacus DP
├── digital_twin/             # RC thermal evaluation
├── evaluation/
│   ├── metrics.py            # Strictly mathematical, NO RNG offsets
│   └── baseline_runner.py    # True baseline models, NO hardcoded dicts
└── scripts/
    └── run_experiment.py     # Master orchestrator (Data -> Train -> Eval)

```