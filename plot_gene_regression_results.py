import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RUNS_ROOT = Path("runs/gene_regression_448_all_folds")
OUT_DIR = RUNS_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Per-fold summary ----------
summary_files = sorted(RUNS_ROOT.glob("fold_*/summary.json"))
rows = []

for path in summary_files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows.append({
        "fold_name": data["fold_name"],
        "best_epoch": data["best_epoch"],
        "best_mean_pearson": data["best_mean_pearson"],
        "best_mse": data["best_metrics"]["mse"],
        "best_mae": data["best_metrics"]["mae"],
    })

df = pd.DataFrame(rows).sort_values("fold_name")
df.to_csv(RUNS_ROOT / "cv_results.csv", index=False)

# Barplot Pearson
plt.figure(figsize=(8, 5))
plt.bar(df["fold_name"], df["best_mean_pearson"])
plt.ylabel("Best mean Pearson")
plt.title("Gene regression performance by LOSO fold")
plt.tight_layout()
plt.savefig(OUT_DIR / "pearson_by_fold.png", dpi=200)
plt.close()

# ---------- History plot ----------
history_paths = sorted(RUNS_ROOT.glob("fold_*/history.csv"))

plt.figure(figsize=(8, 5))
for path in history_paths:
    hist = pd.read_csv(path)
    fold_name = path.parent.name
    plt.plot(hist["epoch"], hist["mean_pearson"], label=fold_name)

plt.xlabel("Epoch")
plt.ylabel("Mean Pearson")
plt.title("Validation mean Pearson across epochs")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "mean_pearson_curves.png", dpi=200)
plt.close()

print("Saved figures to:", OUT_DIR)