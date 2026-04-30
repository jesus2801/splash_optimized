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

Defaults are tuned for an **NVIDIA A100 40 GB** (Google Colab) at
``imgsz=1280`` so the fine-tune resolution matches the stage-1 training
resolution. If you trained stage 1 at a smaller imgsz on a smaller card,
pass ``--imgsz`` here to match — the head's feature-pyramid scales need
to line up across stages or you waste most of stage 1's prior.
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


def batch_arg(value: str) -> int | float:
    """See ``src/train.py:batch_arg`` for rationale (int / float / -1)."""

    try:
        return int(value)
    except ValueError:
        return float(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a stage-1 model on the Uninorte pool dataset (stage 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                   help="Path to the stage-1 best.pt to fine-tune from.")
    p.add_argument("--data", default=str(DEFAULT_DATA),
                   help="Path to the Uninorte data.yaml.")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Fine-tune image size. Should match the stage-1 imgsz "
                        "(also 1280 by default on the A100) so the head "
                        "resolution doesn't change between stages.")
    p.add_argument("--batch", type=batch_arg, default=-1,
                   help="Batch size. -1 = auto (~60%% mem); pass 0.85 for an "
                        "85%% memory budget on a dedicated A100; or an explicit int.")
    p.add_argument("--epochs", type=int, default=60,
                   help="Fewer epochs than stage 1 — we are adapting, not re-learning.")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--workers", type=int, default=12,
                   help="DataLoader workers. 12 matches the Colab A100 host's "
                        "vCPU count; drop to 8 on a 16 GB-card workstation.")
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
    p.add_argument("--cache", default="ram", choices=["ram", "disk", "false"],
                   help="Image cache strategy. 'ram' is the default for Colab "
                        "(host has plenty of RAM and the Uninorte dataset is "
                        "small); drop to 'disk' on a low-RAM workstation.")
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile for the model. ~10-20%% speedup on "
                        "the A100; adds 1-2 min of warmup on the first epoch.")
    p.add_argument("--export-format", default="onnx",
                   choices=["onnx", "engine", "torchscript", "none"],
                   help="Export format after fine-tuning. 'engine' is TensorRT "
                        "FP16 (best for inference on whatever GPU you deploy on); "
                        "'onnx' is the most portable.")
    p.add_argument("--no-export", action="store_true",
                   help="Skip the export step (alias for --export-format none).")
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
    name = torch.cuda.get_device_name(int(device))
    free, total = torch.cuda.mem_get_info(int(device))
    print(
        f"Using NVIDIA GPU: {name}  "
        f"({free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total)"
    )


def setup_cuda_perf(device: str) -> None:
    """Enable Ampere-class fast paths (TF32 matmul, cuDNN benchmark).

    See ``src/train.py:setup_cuda_perf`` for the rationale.
    """

    if device in {"cpu", "mps"}:
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
    assert_inputs(Path(args.weights), Path(args.data))
    assert_cuda(args.device)
    setup_cuda_perf(args.device)
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
        f"\nUninorte validation — mAP50={metrics.box.map50:.3f}  "
        f"mAP50-95={metrics.box.map:.3f}  "
        f"P={metrics.box.mp:.3f}  R={metrics.box.mr:.3f}"
    )

    export_format = "none" if args.no_export else args.export_format
    if export_format != "none":
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
        if export_format != "engine":
            print(
                "\nFor production inference, exporting to TensorRT on the target "
                "GPU is the fastest path (engine files are device-specific):\n"
                f"  yolo export model=<path-to-best.pt> format=engine half=True imgsz={args.imgsz} device=0"
            )


if __name__ == "__main__":
    main()
