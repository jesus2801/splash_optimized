"""Split a flat YOLO dataset (only ``train/``) into ``train`` / ``valid`` / ``test``.

The Roboflow export sometimes lands everything inside ``dataset/train/`` with
no validation or test split. This script takes that flat folder and shuffles
it into the three standard splits, moving each image's matching label file
along with it.

Defaults to a 70 / 20 / 10 split with seed 0 (reproducible across machines).

Typical usage from the repo root:

    python scripts/split_dataset.py                 # apply 70/20/10 in dataset/
    python scripts/split_dataset.py --dry-run       # preview without moving
    python scripts/split_dataset.py --train 0.8 --val 0.15 --test 0.05

Run it ONCE — afterwards ``train/`` already contains only the 70% remainder,
so re-running would not produce the same result.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "dataset"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split a flat YOLO dataset into train/valid/test in place.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", default=str(DEFAULT_ROOT),
                   help="Dataset root containing train/{images,labels}/.")
    p.add_argument("--train", type=float, default=0.70,
                   help="Fraction of pairs to keep in train/.")
    p.add_argument("--val", type=float, default=0.20,
                   help="Fraction of pairs to move to valid/.")
    p.add_argument("--test", type=float, default=0.10,
                   help="Fraction of pairs to move to test/.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for the shuffle.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen without moving any files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(
            f"--train + --val + --test must sum to 1.0 (got {total:.4f})."
        )

    root = Path(args.root).resolve()
    train_img = root / "train" / "images"
    train_lbl = root / "train" / "labels"

    if not train_img.is_dir() or not train_lbl.is_dir():
        raise SystemExit(
            f"Expected {train_img} and {train_lbl} to exist. "
            f"Is --root pointing at the dataset folder?"
        )

    for split in ("valid", "test"):
        d = root / split / "images"
        if d.is_dir() and any(d.iterdir()):
            raise SystemExit(
                f"{d} already contains files. Refusing to clobber an existing split.\n"
                f"Either delete the existing valid/ and test/ folders or re-export the dataset."
            )

    images = sorted(p for p in train_img.iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images found under {train_img}.")

    pairs: list[tuple[Path, Path]] = []
    orphans = 0
    for img in images:
        lbl = train_lbl / (img.stem + ".txt")
        if lbl.is_file():
            pairs.append((img, lbl))
        else:
            orphans += 1

    if orphans:
        print(f"Warning: {orphans} images had no matching label and will be skipped.")
    print(f"Found {len(pairs)} image+label pairs in {train_img.parent}.")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(round(n * args.train))
    n_val = int(round(n * args.val))
    n_test = n - n_train - n_val
    splits: dict[str, list[tuple[Path, Path]]] = {
        "train": pairs[:n_train],
        "valid": pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }

    print(f"\nPlanned split (seed={args.seed}):")
    for name, items in splits.items():
        pct = (len(items) / n * 100) if n else 0.0
        print(f"  {name:<6} {len(items):>6} pairs  ({pct:.1f}%)")

    if args.dry_run:
        print("\n--dry-run: no files moved.")
        return

    for split_name, items in splits.items():
        if split_name == "train":
            continue
        dest_img = root / split_name / "images"
        dest_lbl = root / split_name / "labels"
        dest_img.mkdir(parents=True, exist_ok=True)
        dest_lbl.mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.move(str(img), str(dest_img / img.name))
            shutil.move(str(lbl), str(dest_lbl / lbl.name))
        print(f"Moved {len(items)} pairs into {split_name}/.")

    remaining = sum(1 for _ in train_img.iterdir())
    print(f"\nDone. train/images now contains {remaining} files.")
    print("If your dataset/data.yaml is not already set up, make sure it has:")
    print("  path: .")
    print("  train: train/images")
    print("  val:   valid/images")
    print("  test:  test/images")


if __name__ == "__main__":
    main()
