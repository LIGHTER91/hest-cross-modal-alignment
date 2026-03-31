import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


def infer_gene_columns(columns):
    return sorted(
        [
            c for c in columns
            if c.startswith("gene_") and not c.startswith("gene_name_")
        ]
    )


class HESTCrossModalDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_transform=None,
        normalize_genes: bool = True,
        gene_mean: np.ndarray | None = None,
        gene_std: np.ndarray | None = None,
    ):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV introuvable : {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        required_cols = ["patch_path_auto", "sample_id", "spot_id"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        self.df = self.df[self.df["patch_path_auto"].notna()].copy()
        self.df["patch_path_auto"] = self.df["patch_path_auto"].astype(str)

        # IMPORTANT : on détecte d'abord les colonnes gènes
        self.gene_cols = infer_gene_columns(self.df.columns)
        if len(self.gene_cols) == 0:
            raise ValueError("Aucune colonne gene_XXX trouvée dans le CSV.")

        # puis on nettoie ces colonnes
        self.df[self.gene_cols] = self.df[self.gene_cols].fillna(0.0).astype(np.float32)

        self.image_transform = image_transform
        self.normalize_genes = normalize_genes

        gene_matrix = self.df[self.gene_cols].values.astype(np.float32)

        if normalize_genes:
            if gene_mean is None or gene_std is None:
                gene_mean = gene_matrix.mean(axis=0)
                gene_std = gene_matrix.std(axis=0)
                gene_std = np.where(gene_std < 1e-8, 1.0, gene_std)

            self.gene_mean = gene_mean.astype(np.float32)
            self.gene_std = gene_std.astype(np.float32)

            if len(self.gene_cols) != len(self.gene_mean):
                raise ValueError(
                    f"Dimension gènes incohérente : dataset={len(self.gene_cols)} vs mean={len(self.gene_mean)}"
                )
        else:
            self.gene_mean = None
            self.gene_std = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = Path(row["patch_path_auto"])
        if not img_path.exists():
            raise FileNotFoundError(f"Image introuvable : {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)

        genes = row[self.gene_cols].values.astype(np.float32)

        if self.normalize_genes:
            genes = (genes - self.gene_mean) / self.gene_std

        genes = torch.tensor(genes, dtype=torch.float32)

        return {
            "image": image,
            "genes": genes,
            "sample_id": str(row["sample_id"]),
            "spot_id": str(row["spot_id"]),
            "patch_path": str(img_path),
        }