import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from make_dataloaders import build_fold_dataloaders
from model_contrastive import CrossModalAligner
from loss_contrastive import symmetric_info_nce
from eval_retrieval import retrieval_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", type=str, default="manifest_out/gene_features_448/fold_0")
    parser.add_argument("--image-model-name", type=str, default="owkin/phikon")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="runs/fold_0_phikon")
    parser.add_argument("--unfreeze-image-backbone", action="store_true", default=False)
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, device, temperature, grad_accum_steps=1):
    model.train()

    total_loss = 0.0
    n_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        genes = batch["genes"].to(device, non_blocking=True)

        z_img, z_gene = model(images, genes)
        loss, _ = symmetric_info_nce(z_img, z_gene, temperature=temperature)

        (loss / grad_accum_steps).backward()

        if (step % grad_accum_steps == 0) or (step == len(loader)):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, device, temperature):
    model.eval()

    all_img = []
    all_gene = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        genes = batch["genes"].to(device, non_blocking=True)

        z_img, z_gene = model(images, genes)
        loss, _ = symmetric_info_nce(z_img, z_gene, temperature=temperature)

        all_img.append(z_img)
        all_gene.append(z_gene)

        total_loss += loss.item()
        n_batches += 1

    z_img = torch.cat(all_img, dim=0)
    z_gene = torch.cat(all_gene, dim=0)

    sim_matrix = z_img @ z_gene.T
    metrics = retrieval_metrics(sim_matrix, ks=(1, 5, 10))
    metrics["loss"] = total_loss / max(1, n_batches)

    return metrics


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    args = parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds, train_loader, test_loader = build_fold_dataloaders(
        fold_dir=args.fold_dir,
        image_model_name=args.image_model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    gene_dim = len(train_ds.gene_cols)

    model = CrossModalAligner(
        gene_dim=gene_dim,
        embed_dim=256,
        image_model_name=args.image_model_name,
        freeze_image_backbone=not args.unfreeze_image_backbone,
    ).to(device)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_r1 = -1.0
    best_path = out_dir / "best_model.pt"
    effective_batch_size = args.batch_size * args.grad_accum_steps

    print(f"Device: {device}")
    print(f"Image model: {args.image_model_name}")
    print(f"Train size: {len(train_ds)}")
    print(f"Test size : {len(test_ds)}")
    print(f"Gene dim  : {gene_dim}")
    print(f"Freeze image backbone: {not args.unfreeze_image_backbone}")
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
            temperature=args.temperature,
            grad_accum_steps=args.grad_accum_steps,
        )

        metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            temperature=args.temperature,
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} "
            f"test_loss={metrics['loss']:.4f} "
            f"R@1_i2g={metrics['R@1_i2g']:.4f} "
            f"R@5_i2g={metrics['R@5_i2g']:.4f} "
            f"R@10_i2g={metrics['R@10_i2g']:.4f} "
            f"R@1_g2i={metrics['R@1_g2i']:.4f}"
        )

        if metrics["R@1_i2g"] > best_r1:
            best_r1 = metrics["R@1_i2g"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "gene_dim": gene_dim,
                    "metrics": metrics,
                    "args": vars(args),
                },
                best_path,
            )

    print(f"\nBest model saved to: {best_path}")


if __name__ == "__main__":
    main()