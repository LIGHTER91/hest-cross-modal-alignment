from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor

from dataset_multimodal import HESTCrossModalDataset


def _resolve_resize_from_processor(processor, default_size=224):
    size = getattr(processor, "size", None)

    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        if "shortest_edge" in size:
            s = int(size["shortest_edge"])
            return s, s

    return default_size, default_size


def build_transforms(model_name: str, train: bool = True):
    processor = AutoImageProcessor.from_pretrained(model_name)

    image_mean = getattr(processor, "image_mean", [0.5, 0.5, 0.5])
    image_std = getattr(processor, "image_std", [0.5, 0.5, 0.5])
    h, w = _resolve_resize_from_processor(processor, default_size=224)

    normalize = transforms.Normalize(mean=image_mean, std=image_std)

    if train:
        return transforms.Compose([
            transforms.Resize((h, w)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            normalize,
        ])


def build_fold_dataloaders(
    fold_dir: str,
    image_model_name: str = "owkin/phikon",
    batch_size: int = 64,
    num_workers: int = 2,
):
    fold_dir = Path(fold_dir)

    train_csv = fold_dir / "train_features.csv"
    test_csv = fold_dir / "test_features.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"train CSV introuvable : {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"test CSV introuvable : {test_csv}")

    train_ds = HESTCrossModalDataset(
        csv_path=str(train_csv),
        image_transform=build_transforms(model_name=image_model_name, train=True),
        normalize_genes=True,
    )

    test_ds = HESTCrossModalDataset(
        csv_path=str(test_csv),
        image_transform=build_transforms(model_name=image_model_name, train=False),
        normalize_genes=True,
        gene_mean=train_ds.gene_mean,
        gene_std=train_ds.gene_std,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_ds, test_ds, train_loader, test_loader