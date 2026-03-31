from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path("hest_data")

img_files = []
for ext in ["*.png", "*.jpg", "*.jpeg"]:
    img_files.extend(ROOT.rglob(ext))

print(f"Nombre d'images trouvées : {len(img_files)}")

to_show = img_files[:6]
if not to_show:
    print("Aucune image patch trouvée.")
else:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for ax, img_path in zip(axes, to_show):
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.set_title(img_path.name[:40], fontsize=8)
        ax.axis("off")

    for ax in axes[len(to_show):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()