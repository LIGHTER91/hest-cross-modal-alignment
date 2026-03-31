from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import anndata as ad


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="hest_data")
    parser.add_argument("--manifest", type=str, default="manifest_out/manifest_samples.csv")
    parser.add_argument("--out-dir", type=str, default="manifest_out")
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json_safe(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def find_coord_columns(obs: pd.DataFrame):
    """
    Cherche des colonnes de coordonnées si elles sont déjà dans obs.
    """
    lower_cols = {c.lower(): c for c in obs.columns}

    candidates_x = [
        "x", "pxl_col_in_fullres", "array_col", "imagecol", "col", "x_coord"
    ]
    candidates_y = [
        "y", "pxl_row_in_fullres", "array_row", "imagerow", "row", "y_coord"
    ]

    x_col = None
    y_col = None

    for c in candidates_x:
        if c in lower_cols:
            x_col = lower_cols[c]
            break

    for c in candidates_y:
        if c in lower_cols:
            y_col = lower_cols[c]
            break

    return x_col, y_col


def extract_coords(adata):
    """
    Essaie plusieurs stratégies pour extraire les coordonnées spatiales.
    Retourne un DataFrame indexé comme adata.obs_names avec x/y.
    """
    obs = adata.obs.copy()
    n = obs.shape[0]

    # 1) coordonnées déjà présentes dans obs
    x_col, y_col = find_coord_columns(obs)
    if x_col is not None and y_col is not None:
        return pd.DataFrame(
            {
                "spot_id": obs.index.astype(str),
                "x": obs[x_col].values,
                "y": obs[y_col].values,
                "coord_source": f"obs:{x_col},{y_col}",
            }
        )

    # 2) coordonnées dans obsm["spatial"]
    if "spatial" in adata.obsm:
        spatial = adata.obsm["spatial"]
        if spatial is not None and len(spatial) == n and spatial.shape[1] >= 2:
            return pd.DataFrame(
                {
                    "spot_id": obs.index.astype(str),
                    "x": spatial[:, 0],
                    "y": spatial[:, 1],
                    "coord_source": "obsm:spatial",
                }
            )

    # 3) fallback sans coordonnées
    return pd.DataFrame(
        {
            "spot_id": obs.index.astype(str),
            "x": [None] * n,
            "y": [None] * n,
            "coord_source": ["none"] * n,
        }
    )


def build_patch_index(patches_dir: Path) -> pd.DataFrame:
    """
    Indexe tous les patchs images sous patches_dir.
    Essaye d'extraire sample_id / spot_id depuis le nom de fichier ou le chemin.
    """
    if not patches_dir.exists():
        return pd.DataFrame(
            columns=["sample_id", "spot_id", "patch_path", "patch_filename"]
        )

    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    rows = []

    for p in patches_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in image_exts:
            continue

        rel = str(p.relative_to(patches_dir))
        stem = p.stem
        parts = p.parts

        sample_id = None
        spot_id = None

        # sample_id depuis un dossier TENX...
        for part in parts:
            if part.startswith("TENX"):
                sample_id = part

        # sample_id depuis le nom du fichier
        if sample_id is None:
            for token in stem.replace("-", "_").split("_"):
                if token.startswith("TENX"):
                    sample_id = token
                    break

        # spot_id = nom complet sans extension par défaut
        spot_id = stem

        rows.append(
            {
                "sample_id": sample_id,
                "spot_id": spot_id,
                "patch_path": str(p),
                "patch_filename": p.name,
                "patch_relpath": rel,
            }
        )

    return pd.DataFrame(rows)


def sample_meta_from_json(meta: dict) -> dict:
    """
    Extrait quelques champs utiles si présents.
    """
    keys_of_interest = [
        "id",
        "species",
        "organ",
        "tissue",
        "st_technology",
        "technology",
        "disease_state",
        "oncotree_code",
        "cancer_type",
        "patient_id",
    ]
    out = {}
    for k in keys_of_interest:
        out[k] = meta.get(k, None)
    return out


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)

    ensure_dir(out_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest introuvable: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    print(f"Samples trouvés dans le manifest: {len(manifest_df)}")

    patches_dir = data_dir / "patches_vis"
    patch_index_df = build_patch_index(patches_dir)
    print(f"Patches indexés: {len(patch_index_df)}")

    if len(patch_index_df) > 0:
        patch_index_df.to_csv(out_dir / "patch_index.csv", index=False)

    rows = []

    for _, sample_row in manifest_df.iterrows():
        sample_id = str(sample_row["sample_id"])

        h5ad_rel = sample_row.get("h5ad_path", None)
        wsi_rel = sample_row.get("wsi_path", None)
        json_paths = sample_row.get("json_paths", None)
        parquet_paths = sample_row.get("parquet_paths", None)

        if pd.isna(h5ad_rel):
            print(f"[WARN] pas de h5ad pour {sample_id}, sample ignoré")
            continue

        h5ad_path = data_dir / str(h5ad_rel)
        wsi_path = data_dir / str(wsi_rel) if pd.notna(wsi_rel) else None

        json_path = None
        if pd.notna(json_paths):
            json_path = data_dir / str(str(json_paths).split(" | ")[0])

        parquet_path = None
        if pd.notna(parquet_paths):
            parquet_path = data_dir / str(str(parquet_paths).split(" | ")[0])

        if not h5ad_path.exists():
            print(f"[WARN] h5ad absent pour {sample_id}: {h5ad_path}")
            continue

        print(f"Lecture {sample_id} ...")
        adata = ad.read_h5ad(h5ad_path)

        obs = adata.obs.copy()
        obs["spot_id"] = obs.index.astype(str)

        coord_df = extract_coords(adata)

        meta = load_json_safe(json_path) if json_path else {}
        meta_small = sample_meta_from_json(meta)

        sample_patch_df = patch_index_df.copy()
        if "sample_id" in sample_patch_df.columns:
            sample_patch_df = sample_patch_df[
                (sample_patch_df["sample_id"].isna()) | (sample_patch_df["sample_id"] == sample_id)
            ]

        merged = obs.reset_index(drop=True).merge(coord_df, on="spot_id", how="left")

        # tentative de matching patch par spot_id exact
        if len(sample_patch_df) > 0:
            merged = merged.merge(
                sample_patch_df[["spot_id", "patch_path", "patch_filename", "patch_relpath"]],
                on="spot_id",
                how="left",
            )
        else:
            merged["patch_path"] = None
            merged["patch_filename"] = None
            merged["patch_relpath"] = None

        merged["sample_id"] = sample_id
        merged["h5ad_path"] = str(h5ad_path)
        merged["wsi_path"] = str(wsi_path) if wsi_path else None
        merged["json_path"] = str(json_path) if json_path else None
        merged["parquet_path"] = str(parquet_path) if parquet_path else None
        merged["n_genes"] = adata.n_vars
        merged["n_spots_in_sample"] = adata.n_obs

        for k, v in meta_small.items():
            merged[k] = v

        rows.append(merged)

    if not rows:
        raise RuntimeError("Aucune ligne spot-level construite.")

    spot_manifest_df = pd.concat(rows, axis=0, ignore_index=True)

    out_csv = out_dir / "manifest_spots.csv"
    spot_manifest_df.to_csv(out_csv, index=False)

    print("\nManifest spot-level créé:")
    print(out_csv)

    preview_cols = [
        "sample_id",
        "spot_id",
        "x",
        "y",
        "coord_source",
        "patch_filename",
        "h5ad_path",
        "wsi_path",
        "organ",
        "st_technology",
        "oncotree_code",
    ]
    preview_cols = [c for c in preview_cols if c in spot_manifest_df.columns]

    print("\nAperçu:")
    print(spot_manifest_df[preview_cols].head(20).to_string(index=False))

    matched = spot_manifest_df["patch_path"].notna().sum() if "patch_path" in spot_manifest_df.columns else 0
    total = len(spot_manifest_df)
    print(f"\nSpots totaux: {total}")
    print(f"Spots avec patch matché: {matched}")
    print(f"Taux de match patch: {matched / total:.4f}" if total > 0 else "N/A")


if __name__ == "__main__":
    main()