from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
WSI_EXTS = {".svs", ".tif", ".tiff", ".ndpi"}
ST_EXTS = {".h5ad"}
META_EXTS = {".csv", ".json", ".parquet"}
MASK_EXTS = {".pt", ".npy", ".npz", ".pkl"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="hest_data")
    parser.add_argument("--out-dir", type=str, default="manifest_out")
    parser.add_argument("--filter-dir", type=str, default="hest_filter_output")
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_known_ids(filter_dir: Path) -> list[str]:
    ids = []

    ids_txt = filter_dir / "ids.txt"
    if ids_txt.exists():
        ids.extend([x.strip() for x in ids_txt.read_text(encoding="utf-8").splitlines() if x.strip()])

    filtered_csv = filter_dir / "filtered_metadata.csv"
    if filtered_csv.exists():
        try:
            df = pd.read_csv(filtered_csv)
            if "id" in df.columns:
                ids.extend(df["id"].dropna().astype(str).tolist())
            elif "sample_id" in df.columns:
                ids.extend(df["sample_id"].dropna().astype(str).tolist())
        except Exception:
            pass

    # unique, tri par longueur décroissante pour matcher les IDs longs d'abord
    ids = sorted(set(ids), key=lambda x: (-len(x), x))
    return ids


def guess_sample_id(path_str: str, known_ids: list[str]) -> str | None:
    p = path_str.lower()

    for sid in known_ids:
        if sid.lower() in p:
            return sid

    # fallback heuristique
    patterns = [
        r"\bTENX\d+\b",
        r"\bV\d{2,}\b",
        r"\b[A-Z]{2,}\d{2,}\b",
        r"\b[A-Z0-9]{5,}\b",
    ]

    candidates = []
    for pat in patterns:
        candidates.extend(re.findall(pat, path_str))

    if candidates:
        candidates = sorted(set(candidates), key=lambda x: (-len(x), x))
        return candidates[0]

    return None


def classify_path(path: Path) -> str:
    suffix = path.suffix.lower()
    path_str = str(path).lower()

    if suffix in ST_EXTS:
        return "st_h5ad"

    if suffix in WSI_EXTS:
        return "wsi"

    if suffix in IMAGE_EXTS:
        if "patch" in path_str:
            return "patch_image"
        return "image"

    if suffix == ".json":
        return "json_meta"

    if suffix == ".csv":
        return "csv_meta"

    if suffix == ".parquet":
        return "parquet_meta"

    if suffix in MASK_EXTS:
        if "seg" in path_str or "mask" in path_str:
            return "segmentation_artifact"
        return "tensor_artifact"

    return "other"


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    filter_dir = Path(args.filter_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable: {data_dir}")

    ensure_dir(out_dir)

    known_ids = load_known_ids(filter_dir)
    print(f"IDs connus chargés: {len(known_ids)}")

    all_files = [p for p in data_dir.rglob("*") if p.is_file()]
    print(f"Fichiers trouvés: {len(all_files)}")

    file_rows = []
    per_sample = defaultdict(lambda: {
        "sample_id": None,
        "root_dir": None,
        "h5ad_path": None,
        "wsi_path": None,
        "json_paths": [],
        "csv_paths": [],
        "parquet_paths": [],
        "patch_dirs": set(),
        "seg_dirs": set(),
        "other_image_paths": [],
        "patch_image_count": 0,
        "image_count": 0,
        "wsi_count": 0,
        "h5ad_count": 0,
        "meta_count": 0,
        "artifact_count": 0,
        "all_files_count": 0,
    })

    unmatched = []

    for path in all_files:
        rel = path.relative_to(data_dir)
        path_str = str(rel)

        sample_id = guess_sample_id(path_str, known_ids)
        kind = classify_path(path)

        file_rows.append({
            "sample_id": sample_id,
            "relative_path": str(rel),
            "absolute_path": str(path.resolve()),
            "suffix": path.suffix.lower(),
            "kind": kind,
            "parent_dir": str(rel.parent),
            "filename": path.name,
        })

        if sample_id is None:
            unmatched.append(str(rel))
            continue

        row = per_sample[sample_id]
        row["sample_id"] = sample_id
        row["all_files_count"] += 1

        if row["root_dir"] is None:
            row["root_dir"] = str(rel.parts[0]) if len(rel.parts) > 0 else "."

        if kind == "st_h5ad":
            row["h5ad_count"] += 1
            if row["h5ad_path"] is None:
                row["h5ad_path"] = str(rel)

        elif kind == "wsi":
            row["wsi_count"] += 1
            if row["wsi_path"] is None:
                row["wsi_path"] = str(rel)

        elif kind == "patch_image":
            row["patch_image_count"] += 1
            row["patch_dirs"].add(str(rel.parent))

        elif kind == "image":
            row["image_count"] += 1
            row["other_image_paths"].append(str(rel))

        elif kind == "json_meta":
            row["meta_count"] += 1
            row["json_paths"].append(str(rel))

        elif kind == "csv_meta":
            row["meta_count"] += 1
            row["csv_paths"].append(str(rel))

        elif kind == "parquet_meta":
            row["meta_count"] += 1
            row["parquet_paths"].append(str(rel))

        elif kind in {"segmentation_artifact", "tensor_artifact"}:
            row["artifact_count"] += 1
            row["seg_dirs"].add(str(rel.parent))

    file_index_df = pd.DataFrame(file_rows).sort_values(["sample_id", "kind", "relative_path"], na_position="last")

    sample_rows = []
    for sample_id, row in per_sample.items():
        sample_rows.append({
            "sample_id": row["sample_id"],
            "root_dir": row["root_dir"],
            "h5ad_path": row["h5ad_path"],
            "wsi_path": row["wsi_path"],
            "patch_dir": sorted(row["patch_dirs"])[0] if row["patch_dirs"] else None,
            "seg_dir": sorted(row["seg_dirs"])[0] if row["seg_dirs"] else None,
            "patch_image_count": row["patch_image_count"],
            "image_count": row["image_count"],
            "wsi_count": row["wsi_count"],
            "h5ad_count": row["h5ad_count"],
            "meta_count": row["meta_count"],
            "artifact_count": row["artifact_count"],
            "all_files_count": row["all_files_count"],
            "json_paths": " | ".join(sorted(row["json_paths"])) if row["json_paths"] else None,
            "csv_paths": " | ".join(sorted(row["csv_paths"])) if row["csv_paths"] else None,
            "parquet_paths": " | ".join(sorted(row["parquet_paths"])) if row["parquet_paths"] else None,
        })

    manifest_df = pd.DataFrame(sample_rows).sort_values("sample_id")

    file_index_path = out_dir / "file_index.csv"
    manifest_path = out_dir / "manifest_samples.csv"
    unmatched_path = out_dir / "unmatched_files.txt"

    file_index_df.to_csv(file_index_path, index=False)
    manifest_df.to_csv(manifest_path, index=False)

    with unmatched_path.open("w", encoding="utf-8") as f:
        for x in unmatched[:1000]:
            f.write(x + "\n")

    print("\nManifest créé.")
    print(f"- file index      : {file_index_path}")
    print(f"- sample manifest : {manifest_path}")
    print(f"- unmatched files : {unmatched_path}")

    print("\nAperçu manifest:")
    if len(manifest_df) == 0:
        print("Aucun sample détecté.")
    else:
        cols = [
            "sample_id",
            "h5ad_path",
            "wsi_path",
            "patch_dir",
            "patch_image_count",
            "h5ad_count",
            "wsi_count",
            "all_files_count",
        ]
        cols = [c for c in cols if c in manifest_df.columns]
        print(manifest_df[cols].head(20).to_string(index=False))

    print(f"\nSamples détectés : {len(manifest_df)}")
    print(f"Fichiers sans sample_id détecté : {len(unmatched)}")


if __name__ == "__main__":
    main()