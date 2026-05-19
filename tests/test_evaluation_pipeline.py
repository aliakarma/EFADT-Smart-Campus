"""tests/test_evaluation_pipeline.py — integration tests for eval pipeline."""
import json
import os
import pytest
import yaml


@pytest.fixture(scope="module")
def mini_dataset(tmp_path_factory):
    """Generate a minimal 2-building, 60-day dataset."""
    from data.generation.generate_dataset import generate_full_dataset
    data_dir = str(tmp_path_factory.mktemp("data"))
    generate_full_dataset(n_buildings=2, n_days=60, output_dir=data_dir, seed=99)
    return data_dir


@pytest.fixture(scope="module")
def trained_checkpoint(mini_dataset, tmp_path_factory):
    """Train a minimal model and return checkpoint directory."""
    ckpt_dir = str(tmp_path_factory.mktemp("checkpoints"))
    from models.lstm.train_local import run_standalone_training
    # Set epochs=2 to ensure the integration test runs fast
    result = run_standalone_training(
        building_id="B01",
        data_path=mini_dataset,
        checkpoint_dir=ckpt_dir,
        seed=99,
        epochs=2,
    )
    return ckpt_dir


def test_scaler_serialized_with_checkpoint(trained_checkpoint):
    """Checkpoint must include an accompanying scaler pickle."""
    pt_files = [f for f in os.listdir(trained_checkpoint) if f.endswith("_best.pt")]
    assert pt_files, "No checkpoint .pt file found"
    for pt_file in pt_files:
        scaler_file = pt_file.replace("_best.pt", "_best_scaler.pkl")
        assert os.path.exists(os.path.join(trained_checkpoint, scaler_file)), \
            f"Missing scaler: {scaler_file}"


def test_load_checkpoint_returns_scaler(trained_checkpoint):
    """load_checkpoint() must return a fitted StandardScaler."""
    from models.lstm.train_local import load_checkpoint
    from sklearn.preprocessing import StandardScaler
    import yaml
    with open("configs/hyperparams.yaml") as f:
        config = yaml.safe_load(f)
    ckpt_path = os.path.join(trained_checkpoint, "B01_best.pt")
    model, scaler, ckpt = load_checkpoint(ckpt_path, config)
    assert isinstance(scaler, StandardScaler), "Loaded scaler is not StandardScaler"
    assert hasattr(scaler, "mean_"), "Scaler was not fitted (no mean_ attribute)"
    assert len(scaler.mean_) == 14, f"Expected 14 features, got {len(scaler.mean_)}"


def test_evaluate_checkpoint_produces_metrics(mini_dataset, trained_checkpoint, tmp_path):
    """evaluate_checkpoint.py must write a non-empty results JSON."""
    import subprocess
    import sys
    output = str(tmp_path / "test_results.json")
    result = subprocess.run([
        sys.executable, "scripts/evaluate_checkpoint.py",
        "--checkpoint-dir", trained_checkpoint,
        "--data-dir", mini_dataset,
        "--test-months", "2",
        "--config", "configs/hyperparams.yaml",
        "--building-config", "configs/building_params.yaml",
        "--output", output,
        "--variant", "EFADT (Full)",
    ], capture_output=True, text=True)
    assert result.returncode == 0, f"evaluate_checkpoint.py failed:\n{result.stderr}"
    with open(output) as f:
        data = json.load(f)
    assert "results" in data
    assert data["results"], "Empty results dict"
    for variant, metrics in data["results"].items():
        for key in ["ERR", "CCS", "CSS", "MAE"]:
            assert metrics.get(key) is not None, f"{variant}.{key} is None"
            assert isinstance(metrics[key], (int, float)), f"{variant}.{key} not numeric"
