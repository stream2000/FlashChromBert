"""Linear probing ablation: pretrained backbone vs random-init backbone.

Both runs freeze the entire BERT backbone and only train the [CLS] → linear
head.  The gap in AUC directly measures what representations pretraining
produced, removing the confound of full fine-tuning.

Expected result:
  pretrained: AUC >> 0.50  (backbone encodes promoter-state patterns)
  random:     AUC ~ 0.50   (random embeddings carry no signal)
"""
import json
import os
import subprocess
from pathlib import Path

RUNS = [
    ("pretrained", "configs/ft_linear_probe_pretrained.yaml"),
    ("random",     "configs/ft_linear_probe_random.yaml"),
]

LOG_DIR = Path("logs/ablation_linear_probe")


def run(name: str, config: str) -> dict | None:
    print(f"\n>>> [linear_probe] starting: {name}")
    cmd = f"source ./activate.sh && fcbert-finetune --config {config}"
    try:
        subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
        print(f">>> [linear_probe] done: {name}")
    except subprocess.CalledProcessError as e:
        print(f">>> [linear_probe] FAILED: {name} — {e}")
        return None

    report_file = LOG_DIR / f"{name}_report.json"
    if report_file.exists():
        with open(report_file) as f:
            return json.load(f)
    return None


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, config in RUNS:
        r = run(name, config)
        if r:
            results[name] = r

    print("\n" + "=" * 60)
    print("LINEAR PROBING ABLATION SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        score = r.get("best_score")
        print(f"  {name:12s}  best val_auc = {score:.4f}" if score else f"  {name}: no result")

    summary_path = LOG_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
