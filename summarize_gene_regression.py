import json
from pathlib import Path

import pandas as pd

RUNS_ROOT = Path("runs/gene_regression_448_all_folds")
OUT_CSV = RUNS_ROOT / "cv_results.csv"
OUT_JSON = RUNS_ROOT / "cv_summary.json"

summary_files = sorted(RUNS_ROOT.glob("fold_*/summary.json"))

if not summary_files:
    raise RuntimeError(f"Aucun summary.json trouvé dans {RUNS_ROOT}")

rows = []

for path in summary_files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    row = {
        "fold_name": data["fold_name"],
        "best_epoch": data["best_epoch"],
        "best_mean_pearson": data["best_mean_pearson"],
        "best_loss": data["best_metrics"]["loss"],
        "best_mse": data["best_metrics"]["mse"],
        "best_mae": data["best_metrics"]["mae"],
        "freeze_mode": data["freeze_mode"],
    }
    rows.append(row)

df = pd.DataFrame(rows).sort_values("fold_name")
df.to_csv(OUT_CSV, index=False)

summary = {
    "n_folds": int(len(df)),
    "mean_best_mean_pearson": float(df["best_mean_pearson"].mean()),
    "std_best_mean_pearson": float(df["best_mean_pearson"].std(ddof=1)) if len(df) > 1 else 0.0,
    "mean_best_mse": float(df["best_mse"].mean()),
    "mean_best_mae": float(df["best_mae"].mean()),
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\nPer-fold results:")
print(df.to_string(index=False))

print("\nCV summary:")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nSaved: {OUT_CSV}")
print(f"Saved: {OUT_JSON}")