"""Copy Kaggle Dogs-vs-Cats images into dataset/cats and dataset/dogs.

Usage:  python src/prepare_dataset.py PATH_TO_UNZIPPED_KAGGLE_FOLDER [IMAGES_PER_CLASS]
Works whether the images are mixed in one folder (cat.0.jpg, dog.0.jpg) or already in cats/ and dogs/ folders.
"""
import shutil
import sys
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit(__doc__)

source = Path(sys.argv[1])
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
if not source.exists():
    sys.exit(f"Folder not found: {source}")

root = Path(__file__).resolve().parent.parent
targets = {"cat": root / "dataset" / "cats", "dog": root / "dataset" / "dogs"}
for t in targets.values():
    t.mkdir(parents=True, exist_ok=True)

counts = {"cat": 0, "dog": 0}
for p in sorted(source.rglob("*")):
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    hint = (p.name + " " + p.parent.name).lower()
    kind = "cat" if "cat" in hint else "dog" if "dog" in hint else None
    if kind and counts[kind] < limit:
        shutil.copy(p, targets[kind] / p.name)
        counts[kind] += 1

print(f"Copied {counts['cat']} cat images and {counts['dog']} dog images into {root / 'dataset'}")
if min(counts.values()) == 0:
    print("WARNING: one class has no images. Check that file or folder names contain 'cat' / 'dog'.")
