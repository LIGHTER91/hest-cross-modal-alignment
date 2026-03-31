from pathlib import Path
import pandas as pd


MANIFEST_PATH = "manifest_out/manifest_spots_with_patches_448.csv"
OUT_PATH = "manifest_out/manifest_spots_loso_448.csv"


def main():
    manifest_path = Path(MANIFEST_PATH)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest introuvable : {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_cols = ["sample_id", "spot_id", "patch_path_auto", "patch_exists_auto", "h5ad_path"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    df = df[
        df["patch_exists_auto"].fillna(False)
        & df["patch_path_auto"].notna()
        & df["h5ad_path"].notna()
    ].copy()

    sample_ids = sorted(df["sample_id"].dropna().astype(str).unique().tolist())

    if len(sample_ids) < 2:
        raise RuntimeError("Pas assez de samples pour faire du LOSO.")

    print("Samples détectés :")
    for s in sample_ids:
        print("-", s)

    df["fold_id"] = None
    df["split"] = None

    fold_rows = []

    for fold_idx, test_sample in enumerate(sample_ids):
        fold_name = f"fold_{fold_idx}"
        mask_test = df["sample_id"].astype(str) == test_sample
        mask_train = ~mask_test

        fold_df = df.copy()
        fold_df["fold_id"] = fold_name
        fold_df.loc[mask_test, "split"] = "test"
        fold_df.loc[mask_train, "split"] = "train"

        fold_rows.append(fold_df)

        n_train = int(mask_train.sum())
        n_test = int(mask_test.sum())
        print(f"{fold_name}: test={test_sample} | train_spots={n_train} | test_spots={n_test}")

    out_df = pd.concat(fold_rows, axis=0, ignore_index=True)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"\nFichier écrit : {OUT_PATH}")
    print("\nAperçu :")
    cols = ["fold_id", "split", "sample_id", "spot_id", "patch_path_auto", "h5ad_path"]
    print(out_df[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()