"""Inspect a YOLO-format dataset and produce class / bounding-box statistics.

Generates plots and a text report under ``<repo>/results/data_analysis_<name>/``
so you can sanity-check class balance, label hygiene and annotation
distribution *before* committing to a long training run.

Typical usage:

    python src/data_analysis.py                                 # public dataset
    python src/data_analysis.py --data uninorte_dataset/data.yaml --name uninorte
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "dataset" / "data.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute class / bbox statistics on a YOLO dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default=str(DEFAULT_DATA),
                   help="Path to the dataset's data.yaml.")
    p.add_argument("--name", default="public",
                   help="Subfolder name under results/data_analysis_<name>/.")
    return p.parse_args()


def resolve_base_dir(data_yaml: Path, cfg: dict) -> Path:
    """Resolve the dataset root, trying the conventions Ultralytics accepts.

    Tries (in order): absolute ``path``, ``path`` relative to the data.yaml's
    own folder, ``path`` relative to the repo root. Returns the first one
    that actually exists on disk; falls back to the data.yaml's parent.
    """

    raw = cfg.get("path", ".")
    candidates = [
        Path(raw),
        (data_yaml.parent / raw).resolve(),
        (REPO_ROOT / raw).resolve(),
    ]
    for c in candidates:
        if c.is_absolute() and c.is_dir():
            return c
    return data_yaml.parent.resolve()


def split_paths(data_yaml: Path) -> tuple[Path, dict[str, Path]]:
    """Return ``(base_dir, {yaml_key: images_dir})`` for the splits in the YAML.

    The keys are kept as they appear in the YAML (``train`` / ``val`` /
    ``test``) instead of using the folder names, so downstream code can
    match the YAML's vocabulary even when the on-disk folder is named
    ``valid/`` while the YAML says ``val:``.
    """

    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = resolve_base_dir(data_yaml, cfg)
    splits: dict[str, Path] = {}
    for key in ("train", "val", "test"):
        rel = cfg.get(key)
        if not rel:
            continue
        splits[key] = (base_dir / rel).resolve()
    return base_dir, splits


def labels_dir_for(images_dir: Path) -> Path:
    """Resolve the labels/ folder that pairs with an images/ folder."""

    parts = list(images_dir.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return images_dir.parent / "labels"


def count_classes(label_dir: Path, n_classes: int) -> tuple[np.ndarray, int]:
    """Return per-class instance counts and the number of invalid label lines."""

    counts = np.zeros(n_classes, dtype=np.int64)
    invalid = 0
    if not label_dir.is_dir():
        return counts, invalid

    for label_file in label_dir.glob("*.txt"):
        try:
            with label_file.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(parts[0])
                    except ValueError:
                        invalid += 1
                        continue
                    if 0 <= cls_id < n_classes:
                        counts[cls_id] += 1
                    else:
                        invalid += 1
        except OSError:
            invalid += 1
    return counts, invalid


def collect_bboxes(label_dir: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    """Return ``{class_id: [(x, y, w, h), ...]}`` for every box in *label_dir*."""

    boxes: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    if not label_dir.is_dir():
        return boxes
    for label_file in label_dir.glob("*.txt"):
        with label_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    x, y, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    continue
                boxes[cls_id].append((x, y, w, h))
    return boxes


def plot_class_distribution(
    class_names: list[str],
    per_split_counts: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(class_names))
    width = 0.8 / max(len(per_split_counts), 1)
    for i, (split, counts) in enumerate(per_split_counts.items()):
        ax.bar(x + (i - (len(per_split_counts) - 1) / 2) * width, counts,
               width, label=split)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylabel("Number of instances")
    ax.set_title("Class distribution by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_bbox_overview(
    boxes_by_class: dict[int, list[tuple[float, float, float, float]]],
    class_names: list[str],
    out_path: Path,
    sample_cap: int = 8000,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    all_w: list[float] = []
    all_h: list[float] = []
    for items in boxes_by_class.values():
        all_w.extend(b[2] for b in items)
        all_h.extend(b[3] for b in items)

    axes[0].hist([all_w, all_h], bins=40, label=["width", "height"], alpha=0.7)
    axes[0].set_title("Normalized box size distribution")
    axes[0].set_xlabel("Normalized value")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    rng = np.random.default_rng(0)
    for cls_id, items in boxes_by_class.items():
        if not items:
            continue
        if len(items) > sample_cap:
            idx = rng.choice(len(items), size=sample_cap, replace=False)
            sampled = [items[i] for i in idx]
        else:
            sampled = items
        xs = [b[0] for b in sampled]
        ys = [b[1] for b in sampled]
        axes[1].scatter(xs, ys, s=4, alpha=0.3,
                        label=class_names[cls_id] if cls_id < len(class_names) else f"cls {cls_id}")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(1, 0)
    axes[1].set_title("Box centers (normalized image coords)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend(markerscale=3, fontsize=8)

    # 2D histogram instead of seaborn KDE — KDE is O(n^2) on Windows and
    # easily takes minutes on tens of thousands of points.
    if all_w and all_h:
        h, xedges, yedges = np.histogram2d(all_w, all_h, bins=50, range=[[0, 1], [0, 1]])
        im = axes[2].imshow(
            h.T,
            origin="lower",
            extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
            aspect="auto",
            cmap="magma_r",
        )
        fig.colorbar(im, ax=axes[2], shrink=0.8, label="Boxes")
        axes[2].set_title("Width vs height density")
        axes[2].set_xlabel("Normalized width")
        axes[2].set_ylabel("Normalized height")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data).resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    class_names: list[str] = list(cfg.get("names", []))
    n_classes = int(cfg.get("nc", len(class_names)))

    out_dir = REPO_ROOT / "results" / f"data_analysis_{args.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, splits = split_paths(data_yaml)

    per_split_counts: dict[str, np.ndarray] = {}
    total_invalid = 0
    total_images = 0
    train_label_dir: Path | None = None

    for split_key, split_dir in splits.items():
        label_dir = labels_dir_for(split_dir)
        counts, invalid = count_classes(label_dir, n_classes)
        per_split_counts[split_key] = counts
        total_invalid += invalid
        if split_dir.is_dir():
            total_images += sum(1 for _ in split_dir.glob("*.[jpJP][pnPN]*"))
        if split_key == "train":
            train_label_dir = label_dir
        if not label_dir.is_dir():
            print(f"  warning: labels dir missing for split '{split_key}': {label_dir}")

    plot_class_distribution(
        class_names, per_split_counts, out_dir / "class_distribution.png",
    )

    if train_label_dir is not None:
        boxes = collect_bboxes(train_label_dir)
        plot_bbox_overview(boxes, class_names, out_dir / "bbox_overview.png")

    train_counts = per_split_counts.get("train", np.zeros(n_classes, dtype=np.int64))
    train_total = int(train_counts.sum())
    lines = [
        "=" * 60,
        f" Dataset report — {data_yaml}",
        "=" * 60,
        f"Total images across splits: {total_images}",
        f"Invalid label lines       : {total_invalid}",
        "",
        f"{'Class':<25}{'Train':>10}{'Val':>10}{'Test':>10}{'% train':>10}",
    ]
    for i, name in enumerate(class_names):
        pct = (train_counts[i] / train_total * 100) if train_total else 0.0
        row = f"{name:<25}"
        for split in ("train", "val", "test"):
            counts = per_split_counts.get(split, np.zeros(n_classes, dtype=np.int64))
            row += f"{int(counts[i]):>10}"
        row += f"{pct:>9.1f}%"
        lines.append(row)
    report = "\n".join(lines)
    print(report)
    (out_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to: {out_dir}")


if __name__ == "__main__":
    main()
