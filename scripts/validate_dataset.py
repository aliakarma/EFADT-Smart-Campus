#!/usr/bin/env python
"""
scripts/validate_dataset.py
=============================
Verifies dataset integrity (hash check), reports statistics,
and confirms the train/val/test split sizes.

Usage:
    python scripts/validate_dataset.py \
        --data-dir data/raw \
        --config configs/hyperparams.yaml
"""
import argparse
import hashlib
import json
import logging
import pandas as pd
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--config", default="configs/hyperparams.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    manifest_path = Path(args.data_dir) / "dataset_manifest.json"
    if not manifest_path.exists():
        logger.error(f"dataset_manifest.json not found at {manifest_path}. Re-run make generate-data.")
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    print("\n" + "=" * 70)
    print(f"Dataset Validation Report")
    print(f"Seed: {manifest['seed']} | Buildings: {manifest['n_buildings']} | Days: {manifest['n_days']}")
    print("=" * 70)

    all_ok = True
    train_months = config["data"].get("train_months", [1,2,3,4,5,6])
    val_months = config["data"].get("val_months", [7,8,9])
    test_months = config["data"].get("test_months", [10,11,12])

    for bid, info in manifest["files"].items():
        actual_hash = sha256_file(info["path"])
        hash_ok = actual_hash == info["sha256"]
        if not hash_ok:
            logger.error(f"  {bid}: HASH MISMATCH — dataset may have been modified!")
            all_ok = False

        df = pd.read_parquet(info["path"])
        n_train = (df.index.month.isin(train_months)).sum()
        n_val = (df.index.month.isin(val_months)).sum()
        n_test = (df.index.month.isin(test_months)).sum()
        occ_mean = df["occupancy"].mean()
        occ_max = df["occupancy"].max()

        status = "OK" if hash_ok else "FAIL"
        print(f"  [{status}] {bid}: {len(df):,} rows | train={n_train:,} val={n_val:,} test={n_test:,} | "
              f"occ mean={occ_mean:.1f} max={occ_max}")

    print("=" * 70)
    if all_ok:
        print("SUCCESS: All dataset hashes verified.")
    else:
        print("FAILURE: Hash verification FAILED. Regenerate with: make generate-data")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
