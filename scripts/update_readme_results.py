#!/usr/bin/env python
"""
scripts/update_readme_results.py
==================================
Reads results/ablation/multi_seed_results.json and updates the results table
in README.md enclosed by <!-- RESULTS_TABLE_START --> and <!-- RESULTS_TABLE_END -->.
"""
import json
import os
import re
from pathlib import Path

def fmt(v, s):
    if v is None:
        return "—"
    return f"{v:.3f}±{s:.3f}"

def main():
    project_root = Path(__file__).resolve().parent.parent
    results_path = project_root / "results" / "ablation" / "multi_seed_results.json"
    readme_path = project_root / "README.md"

    if not results_path.exists():
        print(f"Error: {results_path} does not exist. Run evaluations first.")
        return 1

    with open(results_path) as f:
        data = json.load(f)["aggregated"]

    rows = []
    for variant, metrics in data.items():
        row = (f"| {variant} | "
               f"{fmt(metrics['ERR']['mean'], metrics['ERR']['std'])} | "
               f"{fmt(metrics['CCS']['mean'], metrics['CCS']['std'])} | "
               f"{fmt(metrics['CSS']['mean'], metrics['CSS']['std'])} | "
               f"{fmt(metrics['MAE']['mean'], metrics['MAE']['std'])} |")
        rows.append(row)

    table = "\n".join([
        "| Variant | ERR% | CCS | CSS | MAE (persons) |",
        "|---------|------|-----|-----|---------------|",
    ] + rows)

    readme_content = readme_path.read_text(encoding="utf-8")
    
    # We will search for all blocks enclosed in RESULTS_TABLE_START and RESULTS_TABLE_END
    # and replace them with the updated table.
    pattern = r"<!-- RESULTS_TABLE_START -->.*?<!-- RESULTS_TABLE_END -->"
    replacement = f"<!-- RESULTS_TABLE_START -->\n{table}\n<!-- RESULTS_TABLE_END -->"
    
    new_content, count = re.subn(pattern, replacement, readme_content, flags=re.DOTALL)
    
    if count == 0:
        print("Warning: No RESULTS_TABLE_START/END comment markers found in README.md.")
    else:
        readme_path.write_text(new_content, encoding="utf-8")
        print(f"Successfully updated {count} table(s) in README.md.")
        
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
