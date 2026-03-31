import torch
import torch.nn as nn
from torchvision import models


class ImageBackboneRegressor(nn.Module):
    """
    freeze_mode:
        - "full"   : backbone entièrement gelé
        - "layer4" : backbone gelé sauf layer4
        - "none"   : backbone entièrement entraînable
    """

    def __init__(
        self,
        out_dim: int = 50,
        pretrained: bool = True,
        freeze_mode: str = "layer4",
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        if pretrained:
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = models.resnet18(weights=None)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.freeze_mode = freeze_mode

        self._configure_backbone_trainability()

        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def _configure_backbone_trainability(self):
        if self.freeze_mode not in {"full", "layer4", "none"}:
            raise ValueError(f"freeze_mode invalide: {self.freeze_mode}")

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.freeze_mode == "none":
            for p in self.backbone.parameters():
                p.requires_grad = True

        elif self.freeze_mode == "layer4":
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True

    def _forward_backbone(self, x):
        if self.freeze_mode == "full":
            self.backbone.eval()
            with torch.no_grad():
                feats = self.backbone(x)
            return feats
        return self.backbone(x)

    def forward(self, x):
        feats = self._forward_backbone(x)
        preds = self.head(feats)
        return preds