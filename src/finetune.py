"""Stage 2 — fine-tune the stage-1 model on the Uninorte pool dataset.

This script takes the ``best.pt`` produced by ``src/train.py`` (stage 1) and
adapts it to the specific environment of our university's training pool. It
uses a low learning rate and a partially frozen backbone so the model keeps
the general "swimmer / drowning / out of water" features it already learned
and only re-tunes the parts that depend on our pool's lighting, lane ropes,
deck color, camera angle, etc.

Run it once your Roboflow export is unpacked into ``uninorte_dataset/`` and
the labels match the same class indices as ``dataset/data.yaml`` (i.e.
``0=Drowning, 1=Person out of water, 2=Swimming``).

Typical usage from the repo root:

    python src/finetune.py --weights runs/detect/stage1_public/weights/best.pt
    python src/finetune.py --weights best.pt/best.pt --epochs 80 --freeze 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "uninorte_dataset" / "data.yaml"
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "detect" / "stage1_public" / "weights" / "best.pt"
DEFAULT_PROJECT = REPO_ROOT / "runs" / "detect"


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
        description="Fine-tune a stage-1 model on the Uninorte pool dataset (stage 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                   help="Path to the stage-1 best.pt to fine-tune from.")
    p.add_argument("--data", default=str(DEFAULT_DATA),
                   help="Path to the Uninorte data.yaml.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=-1)
    p.add_argument("--epochs", type=int, default=60,
                   help="Fewer epochs than stage 1 — we are adapting, not re-learning.")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--freeze", type=int, default=10,
                   help="Number of leading layers to freeze (backbone). "
                        "10 keeps the YOLOv12 backbone fixed; 0 trains everything.")
    p.add_argument("--lr0", type=float, default=1e-4,
                   help="Initial LR. Much smaller than stage 1 — we are not retraining "
                        "from scratch, only adapting features.")
    p.add_argument("--lrf", type=float, default=0.01,
                   help="Final LR multiplier (final_lr = lr0 * lrf).")
    p.add_argument("--name", default="stage2_uninorte")
    p.add_argument("--project", default=str(DEFAULT_PROJECT))
    p.add_argument("--device", default="0")
    p.add_argument("--cache", default="disk", choices=["ram", "disk", "false"])
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def assert_inputs(weights: Path, data: Path) -> None:
    if not weights.exists():
        raise SystemExit(
            f"Stage-1 weights not found at {weights}.\n"
            f"Run `python src/train.py` first, or pass --weights to point at an existing "
            f"checkpoint (e.g. best.pt/best.pt for the upstream HuggingFace weights)."
        )
    if not data.exists():
        raise SystemExit(
            f"Uninorte data.yaml not found at {data}.\n"
            f"Export your labeled set from Roboflow in YOLOv8/YOLOv11 format and unpack it "
            f"into uninorte_dataset/ so the structure matches uninorte_dataset/README.md."
        )


def assert_cuda(device: str) -> None:
    if device in {"cpu", "mps"}:
        return
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Install a CUDA build of PyTorch matching your driver."
        )
    print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(int(device))}")


def main() -> None:
    args = parse_args()
    assert_inputs(Path(args.weights), Path(args.data))
    assert_cuda(args.device)
    data_path = normalize_data_yaml(Path(args.data))

    model = YOLO(args.weights)

    cache_arg: bool | str = False if args.cache == "false" else args.cache

    model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=cache_arg,
        amp=True,
        cos_lr=True,
        close_mosaic=10,
        patience=args.patience,
        freeze=args.freeze,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=5e-4,
        warmup_epochs=1.0,
        # Conservative augmentation: the Uninorte data is already in-domain,
        # we don't want to drown it under aggressive jitter.
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
        degrees=3.0, translate=0.05, scale=0.3, shear=1.0, perspective=0.0,
        fliplr=0.5, flipud=0.0,
        mosaic=0.5, mixup=0.05, copy_paste=0.2,
        erasing=0.2,
        multi_scale=False,
        seed=0,
        deterministic=True,
        project=args.project,
        name=args.name,
        exist_ok=args.resume,
        resume=args.resume,
        plots=True,
    )

    metrics = model.val()
    print(
        f"\nUninorte validation — mAP50={metrics.box.map50:.3f}  "
        f"mAP50-95={metrics.box.map:.3f}  "
        f"P={metrics.box.mp:.3f}  R={metrics.box.mr:.3f}"
    )

    if not args.no_export:
        export_path = model.export(
            format="onnx",
            imgsz=args.imgsz,
            simplify=True,
            dynamic=True,
            opset=17,
        )
        print(f"Exported ONNX model to: {export_path}")
        print(
            "\nFor production inference on the RTX 3050 you can also export to TensorRT:\n"
            "  yolo export model=<path-to-best.pt> format=engine half=True imgsz=640 device=0"
        )


if __name__ == "__main__":
    main()
