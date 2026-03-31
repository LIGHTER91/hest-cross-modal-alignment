import subprocess
import sys
from pathlib import Path

FOLDS_ROOT = Path("manifest_out/gene_features_448")
RUNS_ROOT = Path("runs/gene_regression_448_all_folds")

EPOCHS = 10
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 4
NUM_WORKERS = 2
FREEZE_MODE = "layer4"

fold_dirs = sorted([p for p in FOLDS_ROOT.iterdir() if p.is_dir() and p.name.startswith("fold_")])

if not fold_dirs:
    raise RuntimeError(f"Aucun fold trouvé dans {FOLDS_ROOT}")

RUNS_ROOT.mkdir(parents=True, exist_ok=True)

for fold_dir in fold_dirs:
    out_dir = RUNS_ROOT / fold_dir.name
    cmd = [
        sys.executable,
        "train_gene_regression.py",
        "--fold-dir", str(fold_dir),
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--grad-accum-steps", str(GRAD_ACCUM_STEPS),
        "--num-workers", str(NUM_WORKERS),
        "--freeze-mode", FREEZE_MODE,
        "--out-dir", str(out_dir),
    ]

    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Échec sur {fold_dir.name}")