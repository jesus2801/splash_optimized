"""Stage 1 — train YOLOv12 on the public-pool dataset.

This is the *base* training step. It produces a general-purpose drowning
detector by fine-tuning a COCO-pretrained YOLOv12 checkpoint on the public
swimming-pool dataset under ``dataset/``. The output weights are then used as
the starting point for the domain-adaptation step in ``src/finetune.py``.

Typical usage from the repo root:

    python src/train.py
    python src/train.py --model yolo12s.pt --epochs 80   # fast baseline
    python src/train.py --model yolo12x.pt               # max-quality (slower)
    python src/train.py --resume                         # resume last run

The defaults are tuned for an RTX A5000 16 GB Laptop GPU at ``imgsz=960``
with the ``yolo12l`` backbone. Compared to the previous RTX 3050 6 GB
defaults this trades a bit of training time per epoch for noticeably better
recall on the small ``Person out of water`` class (whose instances tend to
occupy very few pixels at 640px).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "dataset" / "data.yaml"
DEFAULT_PROJECT = REPO_ROOT / "runs" / "detect"


def normalize_data_yaml(data_yaml: Path) -> str:
    """Rewrite ``data_yaml`` with an absolute ``path:`` and return the new path.

    Ultralytics' ``check_det_dataset`` resolves the YAML's ``path:`` field
    against the current working directory when it's relative (because
    ``Path(".").exists()`` is True, the fallback to ``DATASETS_DIR`` is never
    taken). Roboflow exports always use ``path: .`` and that breaks training
    unless you happen to run ``python`` from inside the dataset folder.

    We dodge the whole problem by writing a sibling ``data.normalized.yaml``
    where ``path:`` is the YAML's own absolute parent directory, which makes
    every relative ``train: / val: / test:`` resolve correctly regardless of
    where the user invoked the script from.
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
        description="Train YOLOv12 on the public pool dataset (stage 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default=str(DEFAULT_DATA), help="Path to data.yaml")
    p.add_argument("--model", default="yolo12l.pt",
                   help="Pretrained checkpoint (auto-downloaded by Ultralytics). "
                        "yolo12l is the default for the A5000 16 GB; use yolo12m "
                        "for faster iteration or yolo12x for max accuracy.")
    p.add_argument("--imgsz", type=int, default=960,
                   help="Training image size. 960 is tuned for the A5000 16 GB; "
                        "drop to 640 for faster epochs or push to 1280 for the "
                        "absolute best small-object recall.")
    p.add_argument("--batch", type=int, default=-1,
                   help="Batch size; -1 lets Ultralytics auto-pick the largest safe batch.")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=25,
                   help="Early-stopping patience in epochs.")
    p.add_argument("--workers", type=int, default=8,
                   help="DataLoader workers. 8 is a good fit for laptop CPUs with "
                        "the A5000 feeding speed; lower it if you see CPU thrash.")
    p.add_argument("--name", default="stage1_public",
                   help="Run name under runs/detect/.")
    p.add_argument("--project", default=str(DEFAULT_PROJECT))
    p.add_argument("--resume", action="store_true",
                   help="Resume the latest run with the same --name.")
    p.add_argument("--device", default="0", help="GPU id, 'cpu', or 'mps'.")
    p.add_argument("--cache", default="disk", choices=["ram", "disk", "false"],
                   help="Image cache strategy. 'disk' is the safest default; "
                        "'ram' is fine on machines with >=32 GB system RAM and "
                        "noticeably reduces dataloader stalls.")
    p.add_argument("--multi-scale", action="store_true",
                   help="Randomly resize training images by +/-50%% per batch. "
                        "Now feasible on the A5000's 16 GB but can OOM the "
                        "auto-batcher at large --imgsz; opt in once you have a "
                        "stable batch size.")
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile for the model. Can give a 5-15%% "
                        "speedup on Ampere+ GPUs but adds 1-2 min of warmup on "
                        "the first epoch and is occasionally flaky on Windows.")
    p.add_argument("--export-format", default="onnx",
                   choices=["onnx", "engine", "torchscript", "none"],
                   help="Export format after training. 'engine' is TensorRT FP16, "
                        "the fastest option for inference on the A5000 itself; "
                        "'onnx' is the most portable; 'none' skips export.")
    p.add_argument("--no-export", action="store_true",
                   help="Skip the export step at the end (alias for --export-format none).")
    return p.parse_args()


def assert_cuda(device: str) -> None:
    if device in {"cpu", "mps"}:
        return
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Install a CUDA build of PyTorch matching your driver, "
            "e.g.: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        )
    print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(int(device))}")


def main() -> None:
    args = parse_args()
    assert_cuda(args.device)
    data_path = normalize_data_yaml(Path(args.data))

    model = YOLO(args.model)

    cache_arg: bool | str
    cache_arg = False if args.cache == "false" else args.cache

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
        close_mosaic=15,
        patience=args.patience,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        # Augmentations tuned for top-down pool footage. Color jitter is
        # moderate (water hue matters), geometric jitter is mild (camera is
        # mounted, not handheld), copy-paste boosts the under-represented
        # "Person out of water" class.
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.5, shear=2.0, perspective=0.0005,
        fliplr=0.5, flipud=0.0,
        mosaic=1.0, mixup=0.15, copy_paste=0.3,
        erasing=0.4,
        multi_scale=args.multi_scale,
        compile=args.compile,
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
        f"\nValidation results — mAP50={metrics.box.map50:.3f}  "
        f"mAP50-95={metrics.box.map:.3f}  "
        f"P={metrics.box.mp:.3f}  R={metrics.box.mr:.3f}"
    )

    export_format = "none" if args.no_export else args.export_format
    if export_format != "none":
        # TensorRT engines are device-specific and don't support dynamic shapes
        # in the same way ONNX does, so we hard-code half=True / dynamic=False
        # for that path. ONNX stays dynamic so the exported model is reusable
        # at any imgsz / batch.
        export_kwargs = {
            "format": export_format,
            "imgsz": args.imgsz,
            "simplify": True,
        }
        if export_format == "engine":
            export_kwargs.update(half=True, dynamic=False, device=args.device)
        elif export_format == "onnx":
            export_kwargs.update(dynamic=True, opset=17)
        export_path = model.export(**export_kwargs)
        print(f"Exported {export_format} model to: {export_path}")


if __name__ == "__main__":
    main()
