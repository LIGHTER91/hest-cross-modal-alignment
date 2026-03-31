import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_multimodal import HESTCrossModalDataset
from model_regression import ImageBackboneRegressor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", type=str, default="manifest_out/gene_features_448/fold_0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="runs/fold_0_gene_regression")
    parser.add_argument("--freeze-mode", type=str, default="layer4", choices=["full", "layer4", "none"])
    return parser.parse_args()


def build_transforms(train: bool = True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


def build_dataloaders(fold_dir: str, batch_size: int, num_workers: int):
    fold_dir = Path(fold_dir)
    train_csv = fold_dir / "train_features.csv"
    test_csv = fold_dir / "test_features.csv"

    train_ds = HESTCrossModalDataset(
        csv_path=str(train_csv),
        image_transform=build_transforms(train=True),
        normalize_genes=True,
    )

    test_ds = HESTCrossModalDataset(
        csv_path=str(test_csv),
        image_transform=build_transforms(train=False),
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


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def compute_genewise_pearson(preds: torch.Tensor, targets: torch.Tensor):
    preds = preds.float()
    targets = targets.float()

    preds = preds - preds.mean(dim=0, keepdim=True)
    targets = targets - targets.mean(dim=0, keepdim=True)

    num = (preds * targets).sum(dim=0)
    den = torch.sqrt((preds ** 2).sum(dim=0)) * torch.sqrt((targets ** 2).sum(dim=0))
    corr = num / (den + 1e-8)

    return corr.mean().item(), corr.detach().cpu().numpy()


def train_one_epoch(model, loader, optimizer, device, grad_accum_steps=1):
    model.train()

    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["genes"].to(device, non_blocking=True)

        preds = model(images)
        loss = F.mse_loss(preds, targets)

        (loss / grad_accum_steps).backward()

        if (step % grad_accum_steps == 0) or (step == len(loader)):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_targets = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["genes"].to(device, non_blocking=True)

        preds = model(images)
        loss = F.mse_loss(preds, targets)

        total_loss += loss.item()
        n_batches += 1

        all_preds.append(preds)
        all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mse = F.mse_loss(all_preds, all_targets).item()
    mae = F.l1_loss(all_preds, all_targets).item()
    mean_pearson, genewise_corr = compute_genewise_pearson(all_preds, all_targets)

    return {
        "loss": total_loss / max(1, n_batches),
        "mse": mse,
        "mae": mae,
        "mean_pearson": mean_pearson,
        "genewise_corr": genewise_corr,
    }


def main():
    args = parse_args()

    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps doit être >= 1")

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_name = Path(args.fold_dir).name

    print(f"Using device: {device}")

    train_ds, test_ds, train_loader, test_loader = build_dataloaders(
        fold_dir=args.fold_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    gene_dim = len(train_ds.gene_cols)

    model = ImageBackboneRegressor(
        out_dim=gene_dim,
        pretrained=True,
        freeze_mode=args.freeze_mode,
        hidden_dim=256,
        dropout=0.2,
    ).to(device)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    effective_batch_size = args.batch_size * args.grad_accum_steps
    best_score = -1e9
    best_epoch = None
    best_metrics = None
    best_path = out_dir / "best_model.pt"
    history = []

    print(f"Device: {device}")
    print(f"Train size: {len(train_ds)}")
    print(f"Test size : {len(test_ds)}")
    print(f"Gene dim  : {gene_dim}")
    print(f"Freeze mode: {args.freeze_mode}")
    print(f"Num workers: {args.num_workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Grad accum steps: {args.grad_accum_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Trainable params: {count_trainable_params(model):,} / {count_total_params(model):,}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_accum_steps=args.grad_accum_steps,
        )

        metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": metrics["loss"],
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "mean_pearson": metrics["mean_pearson"],
        }
        history.append(row)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} "
            f"test_loss={metrics['loss']:.4f} "
            f"mse={metrics['mse']:.4f} "
            f"mae={metrics['mae']:.4f} "
            f"mean_pearson={metrics['mean_pearson']:.4f}"
        )

        if metrics["mean_pearson"] > best_score:
            best_score = metrics["mean_pearson"]
            best_epoch = epoch
            best_metrics = {
                "loss": float(metrics["loss"]),
                "mse": float(metrics["mse"]),
                "mae": float(metrics["mae"]),
                "mean_pearson": float(metrics["mean_pearson"]),
            }

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "gene_dim": gene_dim,
                    "metrics": metrics,
                    "args": vars(args),
                    "gene_mean": train_ds.gene_mean,
                    "gene_std": train_ds.gene_std,
                    "gene_cols": train_ds.gene_cols,
                },
                best_path,
            )

            np.save(out_dir / "best_genewise_corr.npy", metrics["genewise_corr"])

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "fold_name": fold_name,
                "fold_dir": args.fold_dir,
                "best_epoch": best_epoch,
                "best_mean_pearson": best_score,
                "best_metrics": best_metrics,
                "freeze_mode": args.freeze_mode,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "num_workers": args.num_workers,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nBest model saved to: {best_path}")


if __name__ == "__main__":
    main()