import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class PhikonImageEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "owkin/phikon",
        out_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)
        self.freeze_backbone = freeze_backbone

        hidden_dim = int(self.backbone.config.hidden_size)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                outputs = self.backbone(pixel_values=x)
        else:
            outputs = self.backbone(pixel_values=x)

        cls = outputs.last_hidden_state[:, 0, :]
        z = self.proj(cls)
        z = F.normalize(z, dim=-1)
        return z


class GeneEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 50,
        hidden_dim: int = 128,
        out_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        z = F.normalize(z, dim=-1)
        return z


class CrossModalAligner(nn.Module):
    def __init__(
        self,
        gene_dim: int = 50,
        embed_dim: int = 256,
        image_model_name: str = "owkin/phikon",
        freeze_image_backbone: bool = True,
    ):
        super().__init__()

        self.image_encoder = PhikonImageEncoder(
            model_name=image_model_name,
            out_dim=embed_dim,
            freeze_backbone=freeze_image_backbone,
        )
        self.gene_encoder = GeneEncoder(
            in_dim=gene_dim,
            hidden_dim=128,
            out_dim=embed_dim,
            dropout=0.1,
        )

    def forward(self, images, genes):
        z_img = self.image_encoder(images)
        z_gene = self.gene_encoder(genes)
        return z_img, z_gene