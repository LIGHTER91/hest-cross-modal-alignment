import os
from pathlib import Path
from collections import Counter

ROOT = "hest_data"   # change si besoin

root = Path(ROOT)
if not root.exists():
    raise FileNotFoundError(f"Dossier introuvable : {ROOT}")

all_files = [p for p in root.rglob("*") if p.is_file()]
print(f"Nombre total de fichiers : {len(all_files)}")

# Compte les extensions
ext_counter = Counter(p.suffix.lower() for p in all_files)
print("\n=== Extensions trouvées ===")
for ext, count in ext_counter.most_common():
    print(f"{ext or '[no ext]'}: {count}")

# Fichiers potentiellement importants
interesting_exts = {".h5ad", ".svs", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".csv", ".parquet", ".json", ".pt"}
interesting = [p for p in all_files if p.suffix.lower() in interesting_exts]

print("\n=== Fichiers intéressants (max 200) ===")
for p in interesting[:200]:
    print(p)

# Répertoires de premier niveau
print("\n=== Dossiers de premier niveau ===")
for p in sorted(root.iterdir()):
    if p.is_dir():
        print(p.name)