# ============================================================
# EFADT — Makefile
# ============================================================
.PHONY: help setup generate-data train-fl test smoke lint api dashboard clean zip

PYTHON   := python
PIP      := pip
CONFIG   := configs/hyperparams.yaml
BCONFIG  := configs/building_params.yaml

help:
	@echo ""
	@echo "EFADT — Smart Campus Resource Optimizer"
	@echo "========================================"
	@echo ""
	@echo "  make setup           Install dependencies"
	@echo "  make generate-data   Generate 12-building synthetic dataset"
	@echo "  make generate-quick  Generate 2-building, 30-day dataset (fast)"
	@echo "  make train-fl        Run federated learning simulation (100 rounds)"
	@echo "  make train-quick     Run FL simulation (3 buildings, 5 rounds)"
	@echo "  make smoke           Run smoke tests"
	@echo "  make test            Run full pytest suite"
	@echo "  make lint            Run ruff linter"
	@echo "  make api             Start FastAPI server (port 8000)"
	@echo "  make dashboard       Start Streamlit dashboard (port 8501)"
	@echo "  make docker-up       Start all services via Docker Compose"
	@echo "  make docker-down     Stop Docker services"
	@echo "  make clean           Remove generated data and checkpoints"
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────────────
env:
	python -m venv venv
	source venv/Scripts/activate && pip install --upgrade pip \
	  && pip install -r requirements.txt \
	  && pip install -r requirements-dev.txt
	@echo "✓ Virtual environment ready — activate with: source venv/Scripts/activate"

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Dependencies installed"

# ── Data Generation ───────────────────────────────────────────────────────────
generate-data:
	$(PYTHON) -m data.generation.generate_dataset \
	  --config $(CONFIG) \
	  --building-config $(BCONFIG) \
	  --n-buildings 12 \
	  --n-days 365

generate-quick:
	$(PYTHON) -m data.generation.generate_dataset \
	  --config $(CONFIG) \
	  --building-config $(BCONFIG) \
	  --n-buildings 2 \
	  --n-days 30

# ── Federated Learning ────────────────────────────────────────────────────────
train-fl: generate-data
	$(PYTHON) -m federated.simulation \
	  --config $(CONFIG) \
	  --building-config $(BCONFIG) \
	  --n-buildings 12

train-quick:
	$(PYTHON) -m data.generation.generate_dataset \
	  --n-buildings 3 --n-days 14
	$(PYTHON) -m federated.simulation \
	  --n-rounds 5 --n-buildings 3

evaluate:
	$(PYTHON) scripts/evaluate_checkpoint.py \
	  --checkpoint-dir models/lstm/checkpoints \
	  --data-dir data/raw \
	  --output results/ablation/full_results.json

eval-multi-seed:
	$(PYTHON) scripts/multi_seed_eval.py \
	  --seeds 42 0 1 \
	  --output results/ablation/multi_seed_results.json

reproduce: generate-data train-fl evaluate

	@echo "✓ Full reproduction pipeline complete"
	@cat results/ablation/full_results.json

# ── Testing ────────────────────────────────────────────────────────────────────
smoke:
	$(PYTHON) tests/smoke_test.py

test:
	pytest tests/test_core.py -v --tb=short -x

test-api:
	pytest tests/test_api.py -v --tb=short

test-all:
	pytest tests/ -v --tb=short --cov=. --cov-report=term-missing

# ── Code Quality ─────────────────────────────────────────────────────────────
lint:
	ruff check . --ignore E501,E402

format:
	ruff format .

# ── Services ─────────────────────────────────────────────────────────────────
api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run governance/dashboard/app.py --server.port 8501

# ── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api

# ── Utilities ────────────────────────────────────────────────────────────────
clean:
	rm -rf data/raw/*.parquet data/scenarios/*.parquet data/audit/*.jsonl
	rm -rf models/lstm/checkpoints/*.pt models/lstm/checkpoints/*.pkl
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

zip:
	cd /home/claude && zip -r efadt-smart-campus.zip efadt-smart-campus/ \
	  --exclude "*.pyc" --exclude "__pycache__/*" --exclude "*.parquet" \
	  --exclude "*.pt" --exclude "*.pkl" --exclude "data/audit/*"
	@echo "Archive created: /home/claude/efadt-smart-campus.zip"
