## Dataset setup and reproduction commands

### 1) Create the environment

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Log in to Hugging Face

HEST is distributed through Hugging Face and requires accepting the dataset terms once on the dataset page.

```bash
huggingface-cli login
```

### 3) Download the subset used in this project

This project uses the following HEST samples:

- `TENX180`
- `TENX182`
- `TENX183`
- `TENX184`
- `TENX185`

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

repo_id = "MahmoodLab/hest"
local_dir = "hest_data"
ids_to_query = ["TENX180", "TENX182", "TENX183", "TENX184", "TENX185"]
patterns = [f"*{sample_id}[_.]**" for sample_id in ids_to_query]

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=patterns,
    local_dir=local_dir,
)
PY
```

### 4) Build the spot manifest

The preprocessing pipeline in this repository expects a file at:

```text
manifest_out/manifest_spots.csv
```

with at least these columns:

- `sample_id`
- `spot_id`
- `x`
- `y`
- `wsi_path`
- `h5ad_path`

### 5) Extract patches from WSI

```bash
python extract_patches_from_wsi.py \
  --spots-manifest manifest_out/manifest_spots.csv \
  --out-dir hest_data/patches_auto_448 \
  --out-manifest manifest_out/manifest_spots_with_patches_448.csv \
  --patch-size 448
```

### 6) Create LOSO folds

```bash
python make_loso_folds.py
```

### 7) Build gene features

```bash
python build_gene_features.py \
  --loso-manifest manifest_out/manifest_spots_loso_448.csv \
  --out-dir manifest_out/gene_features_448 \
  --n-hvg 50
```

### 8) Train the main gene regression baseline

```bash
python train_gene_regression.py \
  --fold-dir manifest_out/gene_features_448/fold_0 \
  --epochs 10 \
  --batch-size 16 \
  --freeze-mode layer4 \
  --out-dir runs/fold_0_gene_regression
```

### 9) Optional: run the contrastive baseline

```bash
python train_contrastive.py \
  --fold-dir manifest_out/gene_features_448/fold_0 \
  --image-model-name owkin/phikon \
  --epochs 5 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --out-dir runs/fold_0_phikon
```
