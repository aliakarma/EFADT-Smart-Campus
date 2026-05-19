@echo off
echo ==============================================================
echo EFADT-Smart-Campus: Final Reproducibility Certification
echo ==============================================================

echo [1/6] Verifying environment...
python -c "import torch, flwr, opacus, shap, fastapi, streamlit; print('All imports OK')"
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/6] Generating full dataset (12 buildings, 365 days, Seed 42)...
python -m data.generation.generate_dataset --config configs/hyperparams.yaml --building-config configs/building_params.yaml --n-buildings 12 --n-days 365 --seed 42
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/6] Validating dataset checksums...
python scripts/validate_dataset.py --data-dir data/raw --config configs/hyperparams.yaml
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/6] Running federated simulation (12 clients, 100 rounds)...
python -m federated.simulation --config configs/hyperparams.yaml --building-config configs/building_params.yaml --n-buildings 12 --seed 42
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/6] Computing final privacy budget...
python scripts/compute_privacy_budget.py --n-rounds 100 --output results/dp_audit.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/6] Running multi-seed ablation evaluation...
echo (NOTE: This takes several hours on CPU to evaluate all variants across all building data)
python scripts/multi_seed_eval.py --seeds 42 0 1 --output results/ablation/multi_seed_results.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [7/7] Updating README metrics...
python scripts/update_readme_results.py

echo ==============================================================
echo Reproducibility Pass Complete.
echo All artifacts generated. Ready for publication.
echo ==============================================================
