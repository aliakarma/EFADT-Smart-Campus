# 🧠 OVERALL VERDICT
**Score: 3/10**
**Decision: Reject**

The single most fatal flaw is that the reported multi-seed ablation results contain `std=0.000` across all three seeds for five of seven variants, which is mathematically impossible if the FL training actually ran independently per seed — this is produced by re-evaluating the same frozen checkpoint three times with deterministic test months. Beyond this, four more critical integrity failures co-occur: the `-DP` ablation is a literal `dict.copy()` of EFADT Full (`evaluate_checkpoint.py`, lines evaluating `-DP` variant), the centralized baseline ERR is arithmetically fixed via a hardcoded `0.787` multiplier rather than computed from HVAC decisions, trust scores are uniformly zero in every result file despite the README claiming τ=0.887, and the actual FL simulation crashes on any dataset shorter than twelve months (confirmed in `run1_utf8.log`). The software engineering layer is competent and the remediation documentation is unusually thorough, but the scientific outputs cannot be trusted.

---

# 🚨 CRITICAL ISSUES (rejection-level)

## Issue 1: Multi-seed evaluation reuses the same checkpoint — std=0.000 is provably fabricated

- **Problem**: `scripts/multi_seed_eval.py` caches per-seed result files; when `results/seeds/results_seed{N}.json` already exists it loads and returns it without retraining or re-evaluating. Because the FL checkpoints are never regenerated per seed, all three "seeds" evaluate the identical model weights on the identical test split, producing identical numbers.
- **Evidence**: `scripts/multi_seed_eval.py`, lines 32–37:
  ```python
  if os.path.exists(out_path):
      logger.info(f"Seed {seed} results already exist at {out_path}; loading directly.")
      with open(out_path) as f:
          return json.load(f)
  ```
  Confirmed in `results/ablation/multi_seed_results.json` — EFADT Full, -XAI, -DT-WIF, -DP, -MOO all show `"std": 0.0, "values": [X, X, X]`. Only `-FL (centralized)` has nonzero std because it retrains its own model per seed call.
- **Why it invalidates results**: Statistical validity is the entire justification for the multi-seed table in the README. Zero variance across genuinely different seeds is not stable training — it is the same run labeled three times. No confidence intervals, significance tests, or effect sizes derived from this data are meaningful.
- **Exact fix**:
  ```python
  # BEFORE (scripts/multi_seed_eval.py, lines 32–37)
  if os.path.exists(out_path):
      logger.info(f"... loading directly.")
      with open(out_path) as f:
          return json.load(f)

  # AFTER — remove the cache bypass; always rerun, or add --force-rerun flag
  # Delete or gate the early-return block; ensure FL training is invoked per seed
  # before calling evaluate_checkpoint.py
  cmd = [sys.executable, "-m", "federated.simulation",
         "--seed", str(seed), "--output-checkpoint-dir", f"models/checkpoints_seed{seed}"]
  subprocess.run(cmd, check=True)
  # Then run evaluate_checkpoint with --checkpoint-dir models/checkpoints_seed{seed}
  ```

---

## Issue 2: `-DP` ablation is a literal copy of EFADT Full — not independently evaluated

- **Problem**: The `-DP` evaluation path in `scripts/evaluate_checkpoint.py` checks whether `"EFADT (Full)"` is already in the results dict and, if so, copies it verbatim. It does not use a checkpoint trained without DP, nor does it call any separate evaluation function.
- **Evidence**: `scripts/evaluate_checkpoint.py`, `-DP` block:
  ```python
  if "EFADT (Full)" in results:
      results["-DP"] = results["EFADT (Full)"].copy()
  ```
  Confirmed by `results/ablation/full_results.json`: EFADT Full and -DP are byte-for-byte identical across all six metric fields.
- **Why it invalidates results**: The entire ablation claim that DP degrades performance by a specific amount is unsupported. The -DP row is fraudulent by construction.
- **Exact fix**:
  ```python
  # BEFORE
  results["-DP"] = results["EFADT (Full)"].copy()

  # AFTER — use scripts/run_ablations.py which calls retrain_no_dp()
  # and evaluates a genuinely DP-free checkpoint directory
  ckpt_no_dp = retrain_no_dp(n_buildings, seed, config_path, data_dir, building_config)
  results["-DP"] = evaluate_single_variant(
      "-DP", building_data, ckpt_no_dp, config, building_cfg, test_months, device
  ).to_dict()
  ```

---

## Issue 3: Centralized baseline ERR is a hardcoded magic constant (21.3% is not computed)

- **Problem**: `evaluation/baseline_runner.py:evaluate_centralized()` computes system energy as `baseline_E * 0.787`, a literal magic constant that directly and mechanically produces ERR = (1 − 0.787) × 100 = 21.3%.
- **Evidence**: `evaluation/baseline_runner.py`:
  ```python
  system_E = baseline_E * 0.787   # centralized uses same HVAC strategy as EFADT
  ```
  Confirmed in `results/ablation/full_results.json`: `"-FL (centralized)": {"ERR": 21.30000000000003}` — the floating-point residual `0.00000000000003` is the direct arithmetic artifact of `(1.0 - 0.787) * 100`.
- **Why it invalidates results**: The energy comparison claim between federated and centralized learning is fabricated. The value 0.787 is a reverse-engineered target, not measured from HVAC actuation.
- **Exact fix**:
  ```python
  # BEFORE
  system_E = baseline_E * 0.787

  # AFTER — run the trained centralized model through the same agent/DT loop
  # used in evaluate_single_variant, so system_E is actually computed from
  # the agent's HVAC decisions given centralized LSTM predictions
  # (same architecture as EFADT Full evaluation path)
  ```

---

## Issue 4: Trust scores are universally zero — README claim of τ=0.887 is false

- **Problem**: The SHAP proxy explainer is never fitted during the evaluation pipeline. `evaluate_single_variant` sets `trust_scores = np.zeros(len(all_preds))` whenever `all_trust` is empty, which it always is because no `*_best_shap.pkl` files are generated.
- **Evidence**: `scripts/evaluate_checkpoint.py`, inside the eval loop:
  ```python
  shap_path = ckpt_path.replace("_best.pt", "_best_shap.pkl")
  if os.path.exists(shap_path):
      explainer = SHAPProxyExplainer.load(shap_path)
  ```
  No script generates `*_best_shap.pkl`. Confirmed in `results/ablation/full_results.json`: every variant has `"tau": 0.0`. README states `τ=0.887`.
- **Why it invalidates results**: The trust scoring mechanism is a core architectural claim of EFADT. Reporting τ=0.0 while the README asserts τ=0.887 is a direct factual contradiction between claimed and actual system behavior.
- **Exact fix**: Add a SHAP calibration step to the training pipeline that fits `SHAPProxyExplainer` on training predictions and saves `{bid}_best_shap.pkl` alongside the checkpoint. Add to `run_standalone_training` in `models/lstm/train_local.py` or to the FL simulation post-processing loop.

---

## Issue 5: FL simulation crashes on any dataset shorter than 12 months

- **Problem**: `prepare_data()` raises `ValueError` when validation months 7–9 are absent from the data, crashing Flower's Ray actor pool. The fallback temporal split exists in code but is unreachable because the `raise ValueError` at line 84 fires before the fallback branch executes — there is a logic inversion: the fallback is placed after the raise, not instead of it.
- **Evidence**: `run1_utf8.log` (full traceback):
  ```
  ValueError: No validation data for months [7, 8, 9]. Check dataset date range.
  ...
  RuntimeError: Simulation crashed.
  ```
  This crash occurs even for the 2-building/7-day CI smoke test specified in `.github/workflows/ci.yml`.
- **Why it invalidates results**: The CI pipeline cannot validate the FL simulation end-to-end. Any claim that CI "enforces the full pipeline on every push" is false. The weekly scheduled `reproduce.yml` would also fail on fresh environments with short datasets.
- **Exact fix**:
  ```python
  # BEFORE (models/lstm/train_local.py ~line 78–84)
  has_train = any(m in unique_months for m in train_months)
  has_val = any(m in unique_months for m in val_months)
  if not (has_train and has_val):
      # fallback code ...
  if len(df_val) == 0:
      raise ValueError(...)  # THIS fires even after the fallback branch

  # AFTER — remove the unconditional raises inside the fallback branch;
  # the fallback is already assigning df_train/df_val, so just assert they're non-empty
  if len(df_train) == 0 or len(df_val) == 0:
      raise ValueError(f"Temporal split produced empty train or val set.")
  ```

---

## Issue 6: CCS=1.000 for centralized baseline is an artifact of hardcoded inputs

- **Problem**: `evaluate_centralized()` passes `T_in_series=np.full(n, 22.0)` and `co2_series=np.full(n, 600.0)` to `compute_all_metrics`. Temperature 22°C is inside [20,26] and CO₂ 600 ppm is below 1000 ppm — both constraints trivially satisfied, yielding CCS=1.0 by construction.
- **Evidence**: `evaluation/baseline_runner.py`:
  ```python
  return compute_all_metrics(
      ...
      T_in_series=np.full(n, 22.0),   # not available without per-building simulation
      co2_series=np.full(n, 600.0),
      ...
  )
  ```
  Confirmed in `results/ablation/full_results.json`: `"-FL (centralized)": {"CCS": 1.0}` vs EFADT Full `"CCS": 0.167`.
- **Why it invalidates results**: The comfort comparison between FL and centralized approaches is meaningless. CCS=1.0 for centralized is not a measured outcome — it is a mathematical artifact of a placeholder.

---

# ⚠️ MODERATE ISSUES

## Issue 7: `-MOO (energy-only)` produces metrics identical to EFADT Full

Variants EFADT Full and -MOO share identical ERR, CCS, CSS, and MAE in all result files. While MAE being identical is expected (same LSTM checkpoint), ERR should differ because energy-only optimization selects Q≈0 (minimizing |Q|/P_cap). Investigation required: the `step_factor=100` downsampling in `evaluate_single_variant` (evaluating only 1% of agent steps) may suppress the expected ERR difference. The energy metrics are computed only over that downsampled loop. Fix: reduce or remove the downsampling factor, or separate the energy evaluation from the MAE evaluation.

## Issue 8: `-XAI` ablation is structurally identical to EFADT Full in all reported metrics

The `force_zero_trust=True` flag sets τ=0.0, but since τ=0.0 already in EFADT Full (Issue 4), the XAI ablation produces no detectable signal. No other metric differs. The ablation demonstrates nothing about the value of explainability. Fix: Requires Issue 4 to be resolved first; once SHAP is actually fitted and τ>0, the -XAI ablation will be meaningful.

## Issue 9: Claimed original metrics (ERR=34.7%, CCS=0.912, τ=0.887) appear nowhere in runnable code paths

The IMPLEMENTATION_STATUS.md and PROJECT_REPORT.md document an earlier state where these values were hardcoded in `PAPER_RESULTS`. While the remediation correctly removed that dict, the README still contains language implying these are achievable targets. The actual computed metrics are substantially different: ERR=23.3% (not 34.7%), CCS=0.167 (not 0.912). No explanation is provided for the gap.

## Issue 10: The CI workflow smoke test (`--n-days 7`) is guaranteed to crash

`.github/workflows/ci.yml` specifies `--n-days 7` for the minimal FL simulation step, which generates data only covering January 1–7 (a single week of month 1). The `prepare_data` function requires months 7–9 for validation. The CI badge is therefore cosmetically green only if the crash is in a step that doesn't fail the overall workflow, or if no one has run the CI since the current codebase was committed. Fix: Change CI to `--n-days 270` (covering at least months 1–9) or use the temporal-split fallback reliably.

---

# 🟢 MINOR ISSUES

- `governance/dashboard/app.py` imports `from evaluation.baseline_runner import PAPER_RESULTS` and directly renders it in the ablation table without checking for None values — the dashboard will crash when `PAPER_RESULTS` contains `None` entries (before `full_results.json` is generated).
- `federated/server.py` calls `mlflow.log_metric` inside `aggregate_evaluate` with `if mlflow.active_run()` guard, but if the run context was closed early by an exception, this silently fails and the per-round MAE is not logged. Use a try/except per individual call.
- `scripts/evaluate_checkpoint.py` imports `from pathlib import Path` twice (lines 14 and 37); the second import is redundant.
- `models/lstm/train_local.py` has a missing `import pandas as pd` at the module level — it's imported inside `run_standalone_training` instead. This means `prepare_data` will fail when called from `federated/client.py` which imports the function at module level.
- `data/generation/sensor_fault_injector.py` still uses `.ffill().bfill()` as a chained call without assignment, which works in Pandas 2.x but generates a `SettingWithCopyWarning` since `df_out[columns_to_affect]` is a slice. Use `df_out[columns_to_affect] = df_out[columns_to_affect].ffill().bfill()` (already done in-place assignment, so this is just a code quality note).
- `SMOKE_TEST_REPORT.md` reports `SHF = 0.945` but no evaluation script computes SHF from a trained model — all result JSONs show `"SHF": 0.0`. The smoke test report appears to be manually authored, not auto-generated.
- `results/ablation/ablation_seed42.json` uses test_months=[1] while `full_results.json` uses test_months=[10,11,12]. The ablation runner and the main evaluator are testing on different months, making the files incomparable.

---

# 🔬 MISSING EXPERIMENTS

**What**: Per-seed FL retraining with independent checkpoint generation.
**How**: Run `python -m federated.simulation --seed 42`, `--seed 0`, `--seed 1` sequentially, saving each to `models/checkpoints_seed{N}/`. Then run `evaluate_checkpoint.py --checkpoint-dir models/checkpoints_seed{N}` for each.
**Why**: Determines whether the reported metrics are seed-stable or cherry-picked. Current std=0.000 tells us nothing about training variance.
**Expected outcome**: Sound result shows MAE std of 1–3 persons and ERR std of 1–3 percentage points. If std remains exactly 0.0, the FL training is deterministic in a suspicious way.

**What**: SHAP proxy calibration and τ measurement on test set.
**How**: After FL training, collect LSTM predictions on the training split, fit `SHAPProxyExplainer`, save `*_best_shap.pkl`, then run evaluation with the fitted explainer.
**Why**: τ is presented as the primary trust governance metric. Its current value of 0.0 completely invalidates the governance claims of the paper.
**Expected outcome**: τ should be in [0.6, 0.9] if the comfort and safety scores feed correctly into the trust formula.

**What**: True `-DP` ablation with a model trained without DP noise.
**How**: `python scripts/run_ablations.py --seed 42` which calls `retrain_no_dp()` and evaluates it. This already exists in `run_ablations.py` but was never actually executed to generate the committed results.
**Why**: The DP privacy-utility trade-off is a central claim. Without genuine comparison, the DP contribution is scientifically undemonstrated.
**Expected outcome**: Without DP noise (σ=4.845), the model should converge faster and MAE should drop by ~1–3 persons. If -DP MAE ≈ EFADT MAE, DP overhead is negligible and this is a valid positive finding.

---

# 🛠️ ACTION PLAN (prioritized fix roadmap)

**Step 1 — Fix the FL simulation crash on short datasets** (Est. effort: 1h)
- Files: `models/lstm/train_local.py`
- Changes: Restructure `prepare_data` so that the temporal fallback is always reachable and the unconditional `raise ValueError` inside the fallback branch is removed. Test with `--n-days 7`.
- Validates: CI smoke test; all downstream evaluation.

**Step 2 — Add SHAP calibration to training pipeline** (Est. effort: 4h)
- Files: `models/lstm/train_local.py:run_standalone_training`, `federated/simulation.py`
- Changes: After saving `_best.pt` and `_best_scaler.pkl`, collect training predictions, fit `SHAPProxyExplainer`, save `_best_shap.pkl`. This unblocks τ computation.
- Validates: Trust score computation in `evaluate_single_variant`; -XAI ablation.

**Step 3 — Implement true per-seed FL retraining in multi-seed eval** (Est. effort: 6h)
- Files: `scripts/multi_seed_eval.py`
- Changes: Remove the file-cache early-return; invoke `federated.simulation` per seed before calling `evaluate_checkpoint.py`. Pass per-seed checkpoint directories.
- Validates: std>0 across seeds; statistical significance tests meaningful.

**Step 4 — Fix `-DP` ablation to use a genuinely DP-free checkpoint** (Est. effort: 2h)
- Files: `scripts/evaluate_checkpoint.py`
- Changes: Remove the `dict.copy()` shortcut. Call `retrain_no_dp()` from `run_ablations.py` and evaluate the resulting checkpoint directory.
- Validates: `-DP` row in ablation table.

**Step 5 — Fix centralized baseline ERR and CCS** (Est. effort: 4h)
- Files: `evaluation/baseline_runner.py:evaluate_centralized`
- Changes: Remove `system_E = baseline_E * 0.787`. Route the centralized model's predictions through the same DT-agent evaluation loop used in `evaluate_single_variant` to get real HVAC decisions. Replace `T_in_series=np.full(n, 22.0)` with actual thermal simulation.
- Validates: `-FL (centralized)` ERR and CCS rows.

**Step 6 — Investigate -MOO vs Full identity** (Est. effort: 2h)
- Files: `scripts/evaluate_checkpoint.py:evaluate_single_variant`
- Changes: Reduce or eliminate `step_factor=100` downsampling; the energy aggregation over only 1% of timesteps suppresses measurable ERR differences between weight configurations. Alternatively, apply downsampling consistently in both variants but verify that different utility weights produce different HVAC choices.
- Validates: -MOO ablation differentiation.

**Step 7 — Correct README metric claims** (Est. effort: 1h)
- Files: `README.md`, `IMPLEMENTATION_STATUS.md`
- Changes: Remove all references to τ=0.887, CCS=0.912, CSS=0.963, MAE=3.21, ERR=34.7%. Replace with currently computed values with explicit notation that they reflect the current implementation state (not a publication-ready claim). Run `make update-readme` only after Steps 1–6 complete.
- Validates: Claim-implementation alignment.

---

# 📊 FINAL SCORECARD

| Dimension | Score | Key reason |
|---|---|---|
| Reproducibility | 3/10 | FL simulation crashes on <12-month datasets; CI smoke test fails; multi-seed eval reuses single run |
| Experimental validity | 2/10 | CCS=1.0 for centralized is hardcoded; τ=0.0 for all variants; -DP copied |
| Model/algorithm correctness | 5/10 | LSTM and DP mechanism are correctly implemented; evaluation loop has downsampling bug |
| System design realism | 5/10 | Architecture is sensible; latency claims unverified; SHAP never calibrated in production path |
| Baselines & comparisons | 2/10 | -DP=copy; centralized ERR=magic constant; -XAI=-MOO=Full |
| Code quality | 6/10 | Generally clean; missing top-level pandas import; critical semantic bugs in eval scripts |
| Logging & observability | 6/10 | MLflow integration present; per-round MAE logged; SHAP and trust not tracked |
| Robustness | 2/10 | No distribution shift testing; system crashes on any dataset outside 12-month range |
| Claim–implementation alignment | 2/10 | τ=0.887 claimed, 0.000 actual; CCS=0.912 claimed, 0.167 actual; std=0 misrepresented |
| Statistical validity | 1/10 | std=0.000 for 5/7 variants; significance tests trivially non-significant; no independent seeds |
| **Overall** | **3/10** | |

---

# ⚖️ FRAUD RISK ASSESSMENT

**Rating: HIGH**

Six fraud fingerprints triggered simultaneously, and their combination is difficult to attribute to innocent engineering debt:

1. **Hardcoded metric** (Issue 3): `system_E = baseline_E * 0.787` directly produces ERR=21.3% — a reverse-engineered target, not a computed result. This is the clearest single fraud signal.
2. **Disconnected evaluation** (Issue 2): `-DP` results are `dict.copy()` of the Full variant. This is not a subtle bug — it is a two-line block that explicitly substitutes computation with copying.
3. **Reverse-engineered fixtures** (Issue 6): CCS=1.0 for the centralized baseline is produced by feeding hardcoded `T_in=22.0` and `co2=600.0` to the metric function — values chosen to trivially satisfy both constraints.
4. **Fabricated multi-seed statistics** (Issue 1): std=0.000 across three "independent" seeds for five variants. The cache-bypass pattern in `multi_seed_eval.py` is the mechanical cause, but the decision to commit these results to the repository and expose them in the README as a statistical validity demonstration is a clear misrepresentation.
5. **Trust score concealment**: τ=0.0 in all result files while the README and dashboard continue to display 0.887. The architecture scaffolding for trust scoring is present but was never connected to the evaluation.
6. **Original fabricated dict** (documented in PROJECT_REPORT.md): The repository's own internal audit acknowledges that the earlier `PAPER_RESULTS` dict contained fabricated values. The remediation removed the dict but the substitute evaluation introduced new fabrication mechanisms (Issues 1–4).

The original repository had single-point hardcoded values; the remediated version replaced them with a structurally more elaborate but equally fabricated evaluation framework. The fact that the software infrastructure (FastAPI, Flower, Docker, CI, MLflow) is genuinely well-implemented creates a misleading appearance of scientific rigor. The divergence between implementation quality and evaluation honesty is itself a pattern associated with deliberate result manipulation rather than technical inexperience.