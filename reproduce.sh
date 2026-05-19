#!/bin/bash
set -e

echo "=============================================================="
echo "EFADT-Smart-Campus: Final Reproducibility Certification"
echo "=============================================================="

echo "[1/6] Verifying environment..."
python -c "import torch, flwr, opacus, shap, fastapi, streamlit; print('All imports OK')"

echo "[2/6] Generating full dataset (12 buildings, 365 days, Seed 42)..."
make generate-data SEED=42

echo "[3/6] Validating dataset checksums..."
make validate-data

echo "[4/6] Running federated simulation (12 clients, 100 rounds)..."
make train-fl SEED=42

echo "[5/6] Computing final privacy budget..."
python scripts/compute_privacy_budget.py --n-rounds 100 --output results/dp_audit.json

echo "[6/6] Running multi-seed ablation evaluation..."
echo "(NOTE: This takes several hours on CPU to evaluate all variants across all building data)"
make eval-multi-seed

echo "[7/7] Updating README metrics..."
make update-readme

echo "=============================================================="
echo "Reproducibility Pass Complete."
echo "All artifacts generated. Ready for publication."
echo "=============================================================="
