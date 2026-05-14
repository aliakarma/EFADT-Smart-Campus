# Contributing to EFADT

Thank you for your interest in contributing to EFADT!

## Development Setup

```bash
git clone https://github.com/your-org/efadt-smart-campus.git
cd efadt-smart-campus
pip install -r requirements.txt
pip install ruff pytest pytest-asyncio
```

## Running Tests

```bash
make smoke      # Quick validation
make test       # Full pytest suite
```

## Code Style

- Format with `ruff format .`
- Lint with `ruff check . --ignore E501`
- Add type hints to all new functions
- Add docstrings following NumPy style

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Add tests for any new functionality
4. Ensure `make smoke` and `make test` pass
5. Open a PR against `main` with a clear description

## Areas for Contribution

- **Tighter DP accounting** — Integrate Opacus RDP accountant
- **FedProx variant** — Implement proximal term for heterogeneous FL
- **Real sensor adapter** — MQTT/BACnet IoT connector
- **LIME explainer** — Alternative to SHAP for comparison
- **Kubernetes manifests** — Production deployment configs

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
