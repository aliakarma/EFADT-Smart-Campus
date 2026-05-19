import os
import json
import glob

REQUIRED_PATHS = [
    "data/raw/dataset_manifest.json",
    "results/ablation/full_results.json",
    "results/dp_audit.json",
    "configs/hyperparams.yaml",
    "requirements.txt"
]

def check_json_metrics(path):
    with open(path) as f:
        r = json.load(f)
    
    # Handle multi_seed format vs single seed format
    metrics_dict = r.get("aggregated") if "aggregated" in r else r.get("results")
    if not metrics_dict:
        print(f"[FAIL] {path} missing metrics dictionary")
        return False
        
    for variant, metrics in metrics_dict.items():
        for k, v in metrics.items():
            # In multi-seed, v is a dict with 'mean', 'std', etc.
            if isinstance(v, dict):
                v_mean = v.get("mean")
                if v_mean is None or not isinstance(v_mean, (int, float)):
                    print(f"[FAIL] {variant}.{k}.mean is invalid ({v_mean}) in {path}")
                    return False
            else:
                if v is None or not isinstance(v, (int, float)):
                    print(f"[FAIL] {variant}.{k} is not numeric ({v}) in {path}")
                    return False
    return True

def main():
    print("======================================================")
    print("   EFADT Publication Artifacts Verification")
    print("======================================================")
    
    missing = False
    
    # Check required single files
    for p in REQUIRED_PATHS:
        if os.path.exists(p):
            print(f"[OK] Found {p}")
        else:
            print(f"[FAIL] Missing {p}")
            missing = True
            
    # Check checkpoints
    pts = glob.glob("models/lstm/checkpoints/*_best.pt")
    pkls = glob.glob("models/lstm/checkpoints/*_best_scaler.pkl")
    if len(pts) > 0:
        print(f"[OK] Found {len(pts)} PyTorch checkpoints")
    else:
        print(f"[WARNING] PyTorch checkpoints missing or not yet generated.")
        
    if len(pkls) > 0:
        print(f"[OK] Found {len(pkls)} scaler pickles")
    else:
        print(f"[WARNING] Scaler pickles missing or not yet generated.")

    # Check metrics
    for p in ["results/ablation/full_results.json", "results/ablation/multi_seed_results.json"]:
        if os.path.exists(p):
            if check_json_metrics(p):
                print(f"[OK] Metrics verified in {p}")
            else:
                missing = True
        
    print("======================================================")
    if missing:
        print("STATUS: SOME ARTIFACTS OR METRICS ARE MISSING OR INVALID.")
        exit(1)
    else:
        print("STATUS: ALL AVAILABLE ARTIFACTS VERIFIED SUCCESSFULLY.")
        exit(0)

if __name__ == "__main__":
    main()
