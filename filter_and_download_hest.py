import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import login, hf_hub_download, snapshot_download
from tqdm import tqdm

REPO_ID = "MahmoodLab/hest"
REPO_TYPE = "dataset"
CSV_FILENAME = "HEST_v1_3_0.csv"


# ---------------------------
# Helpers
# ---------------------------

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def find_col(columns, candidates):
    """
    Trouve une colonne par nom exact puis par sous-chaîne.
    """
    lower_map = {c.lower(): c for c in columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for c in columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c

    return None


def safe_contains(series, pattern: str):
    return series.astype(str).str.contains(pattern, case=False, na=False)


def safe_equals_any(series, values):
    values_lower = {str(v).strip().lower() for v in values}
    return series.astype(str).str.strip().str.lower().isin(values_lower)


def pretty_print_series(title, series, topn=20):
    print(f"\n=== {title} ===")
    if series.empty:
        print("(vide)")
        return
    print(series.head(topn).to_string())


def show_top_values(df, col, topn=20):
    if col is None or col not in df.columns:
        return
    vals = df[col].dropna().astype(str).value_counts()
    pretty_print_series(f"Top valeurs pour '{col}'", vals, topn=topn)


def maybe_unzip_cellvit_seg(local_dir: str) -> None:
    """
    Dézippe les fichiers éventuels dans local_dir/cellvit_seg.
    """
    seg_dir = Path(local_dir) / "cellvit_seg"
    if not seg_dir.exists():
        return

    zip_files = list(seg_dir.glob("*.zip"))
    if not zip_files:
        return

    print(f"\nDécompression de {len(zip_files)} zip(s) dans {seg_dir} ...")
    for zip_path in tqdm(zip_files, desc="Unzip"):
        out_dir = seg_dir / zip_path.stem
        if out_dir.exists():
            continue
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)


def get_token(cli_token: str | None) -> str | None:
    return cli_token or os.environ.get("HF_TOKEN")


def authenticate_if_needed(cli_token: str | None) -> str | None:
    token = get_token(cli_token)
    if token:
        login(token=token, add_to_git_credential=False)
    return token


# ---------------------------
# Hugging Face download
# ---------------------------

def download_metadata_csv(hf_token: str | None = None, cache_dir: str | None = None) -> str:
    """
    Télécharge le CSV de métadonnées localement via hf_hub_download.
    """
    token = authenticate_if_needed(hf_token)

    csv_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=CSV_FILENAME,
        token=token,
        cache_dir=cache_dir,
    )
    return csv_path


def download_subset(ids_to_query, local_dir: str, hf_token: str | None = None) -> None:
    """
    Télécharge uniquement les fichiers liés aux IDs fournis.
    """
    if not ids_to_query:
        print("\n[WARN] Aucun ID à télécharger.")
        return

    token = authenticate_if_needed(hf_token)
    if not token:
        print("\n[ERREUR] Aucun token Hugging Face trouvé.")
        print("Passe --hf-token 'hf_xxx' ou définis la variable d'environnement HF_TOKEN.")
        sys.exit(1)

    ensure_dir(local_dir)

    patterns = [f"*{sample_id}[_.]**" for sample_id in ids_to_query]

    print(f"\nTéléchargement de {len(ids_to_query)} sample(s) dans : {local_dir}")
    for p in patterns:
        print("pattern:", p)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=patterns,
        local_dir=local_dir,
        token=token,
    )

    maybe_unzip_cellvit_seg(local_dir)
    print("\nTéléchargement terminé.")


# ---------------------------
# Metadata logic
# ---------------------------

def detect_columns(df: pd.DataFrame) -> dict:
    columns = list(df.columns)
    return {
        "id_col": find_col(columns, ["id", "sample_id"]),
        "species_col": find_col(columns, ["species"]),
        "tech_col": find_col(columns, ["st_technology", "technology", "tech"]),
        "organ_col": find_col(columns, ["organ", "tissue", "primary_tissue"]),
        "oncotree_col": find_col(columns, ["oncotree_code", "oncotree", "cancer_type"]),
        "disease_col": find_col(columns, ["disease_state", "disease", "condition", "status"]),
        "patient_col": find_col(columns, ["patient_id", "patient", "case_id", "subject_id"]),
    }


def inspect_metadata(df: pd.DataFrame, detected: dict, topn=20) -> None:
    print(f"\nNombre total de lignes : {len(df)}")
    print("\n=== Colonnes détectées ===")
    for k, v in detected.items():
        print(f"{k}: {v}")

    print("\n=== Toutes les colonnes ===")
    for c in df.columns:
        print("-", c)

    for key in ["species_col", "tech_col", "organ_col", "oncotree_col", "disease_col"]:
        col = detected.get(key)
        if col:
            show_top_values(df, col, topn=topn)


def apply_filters(
    df: pd.DataFrame,
    detected: dict,
    species: str | None = None,
    tech: str | None = None,
    organ: str | None = None,
    oncotree: str | None = None,
    cancer_only: bool = False,
) -> pd.DataFrame:
    out = df.copy()

    species_col = detected["species_col"]
    tech_col = detected["tech_col"]
    organ_col = detected["organ_col"]
    oncotree_col = detected["oncotree_col"]
    disease_col = detected["disease_col"]

    if species and species_col:
        out = out[safe_equals_any(out[species_col], [species, species.lower(), species.upper()])]

    if tech and tech_col:
        out = out[safe_contains(out[tech_col], tech)]

    if organ and organ_col:
        out = out[safe_contains(out[organ_col], organ)]

    if oncotree and oncotree_col:
        out = out[out[oncotree_col].astype(str).str.upper() == oncotree.upper()]

    if cancer_only:
        if oncotree_col and oncotree_col in out.columns:
            out = out[out[oncotree_col].notna() & (out[oncotree_col].astype(str).str.strip() != "")]
        elif disease_col and disease_col in out.columns:
            out = out[safe_contains(out[disease_col], r"cancer|tumou|tumor|adenocarcinoma|carcinoma|malignan|neoplas")]
        else:
            print("\n[WARN] Impossible d'appliquer --cancer-only proprement : pas de colonne oncotree/disease détectée.")

    return out


def preview_filtered(df: pd.DataFrame, detected: dict, n=20) -> None:
    cols = []
    for k in ["id_col", "patient_col", "species_col", "tech_col", "organ_col", "oncotree_col", "disease_col"]:
        c = detected.get(k)
        if c and c in df.columns:
            cols.append(c)

    cols = list(dict.fromkeys(cols))

    print(f"\n=== Aperçu filtré ({min(n, len(df))} lignes) ===")
    if not cols:
        print(df.head(n).to_string(index=False))
    else:
        print(df[cols].head(n).to_string(index=False))


def save_outputs(df: pd.DataFrame, detected: dict, output_dir: str) -> list[str]:
    ensure_dir(output_dir)

    id_col = detected["id_col"]
    patient_col = detected["patient_col"]

    filtered_csv = Path(output_dir) / "filtered_metadata.csv"
    ids_txt = Path(output_dir) / "ids.txt"
    summary_json = Path(output_dir) / "summary.json"

    df.to_csv(filtered_csv, index=False)

    ids = []
    if id_col and id_col in df.columns:
        ids = df[id_col].dropna().astype(str).unique().tolist()
        with open(ids_txt, "w", encoding="utf-8") as f:
            for x in ids:
                f.write(x + "\n")

    summary = {
        "n_rows": int(len(df)),
        "n_unique_ids": int(len(ids)),
        "detected_columns": detected,
        "output_files": {
            "filtered_csv": str(filtered_csv),
            "ids_txt": str(ids_txt),
        },
    }

    if patient_col and patient_col in df.columns:
        summary["n_unique_patients"] = int(df[patient_col].dropna().astype(str).nunique())

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return ids


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Filtrer et télécharger un sous-ensemble du dataset HEST.")
    parser.add_argument("--species", type=str, default=None, help='Ex: "Homo sapiens"')
    parser.add_argument("--tech", type=str, default=None, help='Ex: "Visium"')
    parser.add_argument("--organ", type=str, default=None, help='Ex: "pancreas"')
    parser.add_argument("--oncotree", type=str, default=None, help='Ex: "PAAD"')
    parser.add_argument("--cancer-only", action="store_true", help="Garde seulement les cas tumoraux/cancer si possible.")
    parser.add_argument("--limit", type=int, default=10, help="Nombre max d'IDs à garder.")
    parser.add_argument("--topn", type=int, default=20, help="Nombre de valeurs fréquentes à afficher.")
    parser.add_argument("--output-dir", type=str, default="hest_filter_output", help="Dossier de sortie.")
    parser.add_argument("--download", action="store_true", help="Télécharge le subset après filtrage.")
    parser.add_argument("--download-dir", type=str, default="hest_data", help="Dossier de téléchargement.")
    parser.add_argument("--hf-token", type=str, default=None, help="Token Hugging Face, sinon variable HF_TOKEN.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Dossier cache Hugging Face optionnel.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Téléchargement du CSV de métadonnées '{CSV_FILENAME}' depuis Hugging Face...")
    try:
        csv_path = download_metadata_csv(hf_token=args.hf_token, cache_dir=args.cache_dir)
    except Exception as e:
        print(f"\n[ERREUR] Impossible de télécharger le CSV : {e}")
        print("Vérifie que :")
        print("1) tu as demandé l'accès au dataset HEST sur Hugging Face")
        print("2) ton token HF est valide")
        print("3) ce token appartient au même compte qui a l'accès")
        sys.exit(1)

    print(f"CSV local : {csv_path}")

    try:
        meta_df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"\n[ERREUR] Impossible de lire le CSV local : {e}")
        sys.exit(1)

    detected = detect_columns(meta_df)
    inspect_metadata(meta_df, detected, topn=args.topn)

    filtered_df = apply_filters(
        meta_df,
        detected,
        species=args.species,
        tech=args.tech,
        organ=args.organ,
        oncotree=args.oncotree,
        cancer_only=args.cancer_only,
    )

    print(f"\nNombre de lignes après filtrage : {len(filtered_df)}")

    id_col = detected["id_col"]
    if id_col and id_col in filtered_df.columns:
        unique_ids = filtered_df[id_col].dropna().astype(str).unique().tolist()
        unique_ids = unique_ids[:args.limit]
        filtered_df = filtered_df[filtered_df[id_col].astype(str).isin(unique_ids)]
    else:
        unique_ids = []

    preview_filtered(filtered_df, detected, n=20)

    print("\n=== IDs retenus ===")
    for x in unique_ids:
        print(x)

    ids = save_outputs(filtered_df, detected, args.output_dir)

    print(f"\nFichiers sauvegardés dans : {args.output_dir}")
    print(f"Nombre d'IDs sauvegardés : {len(ids)}")

    if args.download:
        ids_to_download = ids[:args.limit]
        download_subset(ids_to_download, args.download_dir, hf_token=args.hf_token)

    print("\nTerminé.")


if __name__ == "__main__":
    main()