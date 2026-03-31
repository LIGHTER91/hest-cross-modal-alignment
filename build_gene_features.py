from __future__ import annotations

import argparse
from pathlib import Path
import json

import anndata as ad
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loso-manifest", type=str, default="manifest_out/manifest_spots_loso_448.csv")
    parser.add_argument("--out-dir", type=str, default="manifest_out/gene_features_448")
    parser.add_argument("--n-hvg", type=int, default=50)
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def to_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def normalize_var_names(adata: ad.AnnData) -> ad.AnnData:
    adata = adata.copy()
    adata.var_names = pd.Index([str(x).strip() for x in adata.var_names])
    adata.var_names_make_unique()
    return adata


def get_common_genes(sample_ids: list[str], sample_to_adata: dict[str, ad.AnnData]) -> list[str]:
    common = None
    for sample_id in sample_ids:
        genes = set(sample_to_adata[sample_id].var_names.astype(str).tolist())
        if common is None:
            common = genes
        else:
            common &= genes

    if not common:
        raise RuntimeError("Aucun gène commun trouvé entre les samples du fold.")

    return sorted(common)


def compute_hvg_from_train(
    train_df: pd.DataFrame,
    candidate_genes: list[str],
    sample_to_adata: dict[str, ad.AnnData],
    n_hvg: int,
):
    matrices = []

    for sample_id, g in train_df.groupby("sample_id"):
        adata = sample_to_adata[sample_id]
        spot_ids = g["spot_id"].astype(str).tolist()
        present = [sid for sid in spot_ids if sid in adata.obs_names]

        if not present:
            continue

        sub = adata[present, candidate_genes].copy()
        X = to_dense(sub.X).astype(np.float32)
        X = np.log1p(X)
        matrices.append(X)

    if not matrices:
        raise RuntimeError("Impossible de calculer les HVG : aucune matrice train valide.")

    X_train = np.vstack(matrices)
    variances = X_train.var(axis=0)

    n_keep = min(n_hvg, len(candidate_genes))
    top_idx = np.argsort(-variances)[:n_keep]
    top_genes = [candidate_genes[i] for i in top_idx]

    return top_genes


def build_features_for_fold(
    fold_df: pd.DataFrame,
    sample_to_adata: dict[str, ad.AnnData],
    hvg_genes: list[str],
):
    rows = []

    for _, row in fold_df.iterrows():
        sample_id = str(row["sample_id"])
        spot_id = str(row["spot_id"])

        adata = sample_to_adata[sample_id]
        if spot_id not in adata.obs_names:
            continue

        x = adata[spot_id, hvg_genes].X
        x = to_dense(x).reshape(-1).astype(np.float32)
        x = np.log1p(x)

        out = {
            "fold_id": row["fold_id"],
            "split": row["split"],
            "sample_id": sample_id,
            "spot_id": spot_id,
            "patch_path_auto": row["patch_path_auto"],
        }

        for i, gene_name in enumerate(hvg_genes):
            out[f"gene_name_{i:03d}"] = gene_name
            out[f"gene_{i:03d}"] = float(x[i])

        rows.append(out)

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    loso_path = Path(args.loso_manifest)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not loso_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {loso_path}")

    df = pd.read_csv(loso_path)

    required = ["fold_id", "split", "sample_id", "spot_id", "h5ad_path", "patch_path_auto"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    df = df[
        df["patch_path_auto"].notna()
        & df["h5ad_path"].notna()
    ].copy()

    sample_h5ad = (
        df[["sample_id", "h5ad_path"]]
        .drop_duplicates()
        .sort_values("sample_id")
    )

    sample_to_adata = {}
    for _, row in sample_h5ad.iterrows():
        sample_id = str(row["sample_id"])
        h5ad_path = Path(str(row["h5ad_path"]))
        if not h5ad_path.exists():
            raise FileNotFoundError(f"h5ad absent : {h5ad_path}")

        print(f"Chargement {sample_id} -> {h5ad_path}")
        adata = ad.read_h5ad(h5ad_path)
        adata = normalize_var_names(adata)
        sample_to_adata[sample_id] = adata

    fold_ids = sorted(df["fold_id"].dropna().astype(str).unique().tolist())
    summary = {}

    for fold_id in fold_ids:
        print(f"\n===== {fold_id} =====")
        fold_df = df[df["fold_id"].astype(str) == fold_id].copy()
        train_df = fold_df[fold_df["split"] == "train"].copy()
        test_df = fold_df[fold_df["split"] == "test"].copy()

        fold_sample_ids = sorted(fold_df["sample_id"].astype(str).unique().tolist())
        candidate_genes = get_common_genes(fold_sample_ids, sample_to_adata)

        hvg_genes = compute_hvg_from_train(
            train_df=train_df,
            candidate_genes=candidate_genes,
            sample_to_adata=sample_to_adata,
            n_hvg=args.n_hvg,
        )

        print(f"{fold_id} | gènes communs fold: {len(candidate_genes)} | HVG retenus: {len(hvg_genes)}")

        train_feat = build_features_for_fold(train_df, sample_to_adata, hvg_genes)
        test_feat = build_features_for_fold(test_df, sample_to_adata, hvg_genes)

        fold_dir = out_dir / fold_id
        ensure_dir(fold_dir)

        train_path = fold_dir / "train_features.csv"
        test_path = fold_dir / "test_features.csv"
        genes_path = fold_dir / "hvg_genes.json"

        train_feat.to_csv(train_path, index=False)
        test_feat.to_csv(test_path, index=False)

        with genes_path.open("w", encoding="utf-8") as f:
            json.dump(hvg_genes, f, indent=2, ensure_ascii=False)

        summary[fold_id] = {
            "n_train": int(len(train_feat)),
            "n_test": int(len(test_feat)),
            "n_hvg": int(len(hvg_genes)),
            "train_path": str(train_path),
            "test_path": str(test_path),
            "genes_path": str(genes_path),
        }

        print(f"{fold_id}: train={len(train_feat)} | test={len(test_feat)} | hvg={len(hvg_genes)}")

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nTerminé. Résumé écrit dans : {summary_path}")


if __name__ == "__main__":
    main()