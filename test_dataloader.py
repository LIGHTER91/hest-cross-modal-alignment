from make_dataloaders import build_fold_dataloaders

FOLD_DIR = "manifest_out/gene_features/fold_0"

train_ds, test_ds, train_loader, test_loader = build_fold_dataloaders(
    fold_dir=FOLD_DIR,
    batch_size=16,
    num_workers=0,
)

print("Train size:", len(train_ds))
print("Test size :", len(test_ds))

batch = next(iter(train_loader))

print("Image batch shape:", batch["image"].shape)
print("Gene batch shape :", batch["genes"].shape)
print("Sample IDs       :", batch["sample_id"][:3])
print("Spot IDs         :", batch["spot_id"][:3])
print("Patch path       :", batch["patch_path"][:1])