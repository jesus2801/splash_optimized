"""Evaluate a trained YOLO checkpoint on a YOLO dataset's val/test split.

This is a thin wrapper around ``YOLO.val()`` that prints a tidy per-class
report and writes a CSV / JSON summary, so we can compare runs without having
to dig through ``runs/detect/<name>/`` by hand.

Defaults are tuned for a Colab **A100 40 GB** running at ``imgsz=1280``
with ``--half`` on by default — at this size FP16 validation is ~2x faster
than FP32 with no measurable accuracy delta on this task.

Typical usage:

    python src/model_evaluation.py --weights runs/detect/stage1_public/weights/best.pt
    python src/model_evaluation.py --weights best.pt/best.pt \
        --data uninorte_dataset/data.yaml --split val --imgsz 960   # match older runs
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "dataset" / "data.yaml"


def normalize_data_yaml(data_yaml: Path) -> str:
    """Rewrite ``data_yaml`` with an absolute ``path:`` and return the new path.

    See the docstring in ``src/train.py`` for why this is necessary.
    """

    data_yaml = data_yaml.resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    raw_path = cfg.get("path", ".")
    base = Path(raw_path)
    if not base.is_absolute():
        candidate = (data_yaml.parent / base).resolve()
        base = candidate if candidate.is_dir() else data_yaml.parent
    cfg["path"] = str(base)

    out_path = data_yaml.parent / "data.normalized.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a YOLO checkpoint and dump a summary report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", required=True,
                   help="Path to the .pt checkpoint to evaluate.")
    p.add_argument("--data", default=str(DEFAULT_DATA),
                   help="Path to the dataset's data.yaml.")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Validation image size. Should match the training imgsz "
                        "(1280 by default for the A100 40 GB; pass 960 to match "
                        "older A5000 runs).")
    p.add_argument("--batch", type=int, default=32,
                   help="Validation batch size. 32 fits comfortably on the "
                        "A100 40 GB at imgsz=1280 with --half; push to 64 if you "
                        "drop --imgsz to 960, lower it for imgsz=1536+.")
    p.add_argument("--conf", type=float, default=0.001,
                   help="Low confidence threshold for full PR-curve evaluation.")
    p.add_argument("--iou", type=float, default=0.6,
                   help="IoU threshold for NMS during validation.")
    p.add_argument("--device", default="0")
    p.add_argument("--half", dest="half", action="store_true", default=True,
                   help="Run validation in FP16. ON by default on the A100 — "
                        "use --no-half to compare against an FP32 baseline.")
    p.add_argument("--no-half", dest="half", action="store_false",
                   help="Disable FP16 validation (slower, marginally more accurate).")
    p.add_argument("--out-dir", default=None,
                   help="Where to save report.csv / report.json. "
                        "Defaults to the run's save_dir.")
    return p.parse_args()


def setup_cuda_perf(device: str) -> None:
    """Enable Ampere-class TF32 fast paths.

    See ``src/train.py:setup_cuda_perf`` for the rationale.
    """

    if device in {"cpu", "mps"}:
        return
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")
    setup_cuda_perf(args.device)
    data_path = normalize_data_yaml(Path(args.data))

    model = YOLO(str(weights))
    metrics = model.val(
        data=data_path,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        half=args.half,
        plots=True,
        save_json=True,
    )

    names = model.names
    per_class = []
    map50_per_cls = metrics.box.maps  # mAP50-95 per class
    p_per_cls = getattr(metrics.box, "p", None)
    r_per_cls = getattr(metrics.box, "r", None)

    for cls_id, cls_name in names.items():
        row = {
            "class_id": int(cls_id),
            "class": cls_name,
            "mAP50_95": float(map50_per_cls[cls_id]) if cls_id < len(map50_per_cls) else None,
        }
        if p_per_cls is not None and cls_id < len(p_per_cls):
            row["precision"] = float(p_per_cls[cls_id])
        if r_per_cls is not None and cls_id < len(r_per_cls):
            row["recall"] = float(r_per_cls[cls_id])
        per_class.append(row)

    overall = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "weights": str(weights),
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
    }

    print("\n=== Overall ===")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k:<10} {v:.4f}")
        else:
            print(f"  {k:<10} {v}")

    print("\n=== Per class ===")
    print(f"{'class':<25}{'P':>10}{'R':>10}{'mAP50-95':>12}")
    for row in per_class:
        p = f"{row.get('precision', 0):.3f}" if "precision" in row else "  -  "
        r = f"{row.get('recall', 0):.3f}" if "recall" in row else "  -  "
        m = f"{row.get('mAP50_95', 0):.3f}"
        print(f"{row['class']:<25}{p:>10}{r:>10}{m:>12}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(metrics.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump({"overall": overall, "per_class": per_class}, f, indent=2)
    with (out_dir / "report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "class", "precision", "recall", "mAP50_95"],
        )
        writer.writeheader()
        for row in per_class:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})
    print(f"\nSaved JSON / CSV report to: {out_dir}")


if __name__ == "__main__":
    main()
