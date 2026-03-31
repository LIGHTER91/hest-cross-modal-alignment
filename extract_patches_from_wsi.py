from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import openslide


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spots-manifest", type=str, default="manifest_out/manifest_spots.csv")
    parser.add_argument("--out-dir", type=str, default="hest_data/patches_auto_448")
    parser.add_argument("--out-manifest", type=str, default="manifest_out/manifest_spots_with_patches_448.csv")
    parser.add_argument("--patch-size", type=int, default=448)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--limit-per-sample", type=int, default=None)
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def safe_read_region(slide, x, y, patch_size, level=0):
    half = patch_size // 2
    w0, h0 = slide.level_dimensions[0]

    x0 = int(round(x)) - half
    y0 = int(round(y)) - half

    x0 = clamp(x0, 0, max(0, w0 - patch_size))
    y0 = clamp(y0, 0, max(0, h0 - patch_size))

    region = slide.read_region((x0, y0), level, (patch_size, patch_size)).convert("RGB")
    return region, x0, y0


def main():
    args = parse_args()

    spots_manifest_path = Path(args.spots_manifest)
    out_dir = Path(args.out_dir)
    out_manifest_path = Path(args.out_manifest)

    ensure_dir(out_dir)
    ensure_dir(out_manifest_path.parent)

    if not spots_manifest_path.exists():
        raise FileNotFoundError(f"Manifest introuvable : {spots_manifest_path}")

    df = pd.read_csv(spots_manifest_path)

    required_cols = ["sample_id", "spot_id", "x", "y", "wsi_path"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le manifest : {missing}")

    df["patch_path_auto"] = None
    df["patch_exists_auto"] = False
    df["patch_x0"] = None
    df["patch_y0"] = None
    df["patch_width"] = None
    df["patch_height"] = None

    work_df = df[df["x"].notna() & df["y"].notna() & df["wsi_path"].notna()].copy()
    print(f"Nombre total de spots exploitables : {len(work_df)}")
    print(f"Taille de patch demandée : {args.patch_size}")

    grouped = work_df.groupby("sample_id", sort=True)

    total_processed = 0
    total_ok = 0

    for sample_id, g in grouped:
        g = g.copy()

        if args.limit_per_sample is not None:
            g = g.head(args.limit_per_sample)

        if len(g) == 0:
            continue

        wsi_path = Path(g["wsi_path"].iloc[0])
        if not wsi_path.exists():
            print(f"[WARN] WSI absente pour {sample_id} : {wsi_path}")
            continue

        print(f"Lecture WSI : {sample_id} -> {wsi_path}")
        slide = openslide.OpenSlide(str(wsi_path))
        sample_out_dir = out_dir / str(sample_id)
        ensure_dir(sample_out_dir)

        for idx, row in tqdm(g.iterrows(), total=len(g), desc=f"Extract {sample_id}"):
            total_processed += 1

            spot_id = str(row["spot_id"])
            x = float(row["x"])
            y = float(row["y"])

            try:
                patch, x0, y0 = safe_read_region(
                    slide,
                    x,
                    y,
                    patch_size=args.patch_size,
                    level=args.level,
                )

                out_path = sample_out_dir / f"{spot_id}.png"
                patch.save(out_path)

                df.at[idx, "patch_path_auto"] = str(out_path)
                df.at[idx, "patch_exists_auto"] = True
                df.at[idx, "patch_x0"] = x0
                df.at[idx, "patch_y0"] = y0
                df.at[idx, "patch_width"] = args.patch_size
                df.at[idx, "patch_height"] = args.patch_size

                total_ok += 1

            except Exception as e:
                print(f"[WARN] Échec patch {sample_id}/{spot_id}: {e}")

        slide.close()

    df.to_csv(out_manifest_path, index=False)

    print("\nExtraction terminée.")
    print(f"Manifest mis à jour : {out_manifest_path}")
    print(f"Spots traités : {total_processed}")
    print(f"Patches extraits : {total_ok}/{total_processed}")

    preview_cols = [
        "sample_id",
        "spot_id",
        "x",
        "y",
        "patch_path_auto",
        "patch_exists_auto",
        "patch_x0",
        "patch_y0",
        "patch_width",
        "patch_height",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]

    print("\nAperçu :")
    print(df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()