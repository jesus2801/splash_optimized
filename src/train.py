"""Stage 1 — train YOLOv12 on the public-pool dataset.

This is the *base* training step. It produces a general-purpose drowning
detector by fine-tuning a COCO-pretrained YOLOv12 checkpoint on the public
swimming-pool dataset under ``dataset/``. The output weights are then used as
the starting point for the domain-adaptation step in ``src/finetune.py``.

Typical usage from the repo root:

    python src/train.py
    python src/train.py --model yolo12l.pt --imgsz 960 --epochs 80   # fast baseline
    python src/train.py --model yolo12x.pt --imgsz 1280              # max-quality (slower)
    python src/train.py --resume                                     # resume last run

The defaults are tuned for an **NVIDIA A100 40 GB** (the GPU exposed by
Google Colab Pro) at ``imgsz=1280`` with the ``yolo12l`` backbone.
Compared to the previous RTX A5000 16 GB defaults (``yolo12l`` @ 960)
we keep the same backbone and just push the resolution up to 1280, which
gives the small ``Person out of water`` class noticeably more pixels to
work with — that class's instances tend to occupy very few pixels even
in 1080p footage. ``yolo12x`` @ 1280 was tempting but OOMs the 40 GB
A100 at any sensible batch size; it only becomes viable on the 80 GB
A100 (Colab Pro+) — see the GPU fallback table in the README.

Per-epoch wall-clock on a Colab A100 40 GB is roughly 2–3 minutes on
~5 k images, so the full 150-epoch run finishes in 5–8 hours and
``patience=25`` early-stopping usually wraps it in 3–5.

If you go back to a smaller card or up to a beefier one, drop in the
matching defaults explicitly:

    # A100 80 GB (Colab Pro+) — fits the bigger model
    python src/train.py --model yolo12x.pt --imgsz 1280 --batch 16
    # RTX A5000 16 GB
    python src/train.py --model yolo12l.pt --imgsz 960  --workers 8 --cache disk
    # RTX 3050 6 GB
    python src/train.py --model yolo12m.pt --imgsz 640  --workers 4 --cache disk --batch 4
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


def batch_arg(value: str) -> int | float:
    """Parse ``--batch`` so the user can pass either an int or a float fraction.

    Ultralytics accepts ``batch=<int>`` (explicit), ``batch=-1`` (auto-batch
    aiming at ~60% memory), or ``batch=<float in (0, 1)>`` (auto-batch aiming
    at that fraction of GPU memory). The float form is genuinely useful on
    the A100 because Colab dedicates the whole GPU to the notebook, so
    pushing utilization above the conservative 60% default is safe.
    """

    try:
        return int(value)
    except ValueError:
        return float(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLOv12 on the public pool dataset (stage 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default=str(DEFAULT_DATA), help="Path to data.yaml")
    p.add_argument("--model", default="yolo12l.pt",
                   help="Pretrained checkpoint (auto-downloaded by Ultralytics). "
                        "yolo12l is the default for the Colab A100 40 GB — the "
                        "right balance of accuracy and VRAM at imgsz=1280. "
                        "yolo12x only fits on an 80 GB A100 (Colab Pro+) at "
                        "this resolution; yolo12m is the fallback for a 16 GB "
                        "card.")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Training image size. 1280 is tuned for the A100 40 GB and "
                        "gives the small 'Person out of water' class a lot more "
                        "pixels to work with; drop to 960 to roughly halve epoch "
                        "time, or 640 for a really fast baseline.")
    p.add_argument("--batch", type=batch_arg, default=-1,
                   help="Batch size. -1 = auto-batch at ~60%% GPU memory (safest). "
                        "Pass a float like 0.85 to auto-batch at 85%% (recommended on "
                        "Colab A100 since nothing else competes for VRAM), or an "
                        "explicit int (e.g. 16 / 24 / 32) to pin it.")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=25,
                   help="Early-stopping patience in epochs.")
    p.add_argument("--workers", type=int, default=12,
                   help="DataLoader workers. 12 matches the Colab A100 host's vCPU "
                        "count; drop to 8 on a 16 GB-card workstation, 4 on a 6 GB "
                        "laptop card to avoid CPU thrash.")
    p.add_argument("--name", default="stage1_public",
                   help="Run name under runs/detect/.")
    p.add_argument("--project", default=str(DEFAULT_PROJECT))
    p.add_argument("--resume", action="store_true",
                   help="Resume the latest run with the same --name. Especially "
                        "useful on Colab where sessions can disconnect.")
    p.add_argument("--device", default="0", help="GPU id, 'cpu', or 'mps'.")
    p.add_argument("--cache", default="ram", choices=["ram", "disk", "false"],
                   help="Image cache strategy. 'ram' is the new default because "
                        "Colab's A100 host typically ships with ~50 GB system RAM "
                        "and dataset/ fits comfortably; drop to 'disk' on a "
                        "low-RAM workstation, 'false' if you're tight on both.")
    p.add_argument("--multi-scale", action="store_true",
                   help="Randomly resize training images by +/-50%% per batch. "
                        "Now safer to enable on the A100 (40 GB headroom) but can "
                        "still OOM the auto-batcher at large --imgsz; opt in once "
                        "you have a stable batch size from a previous run.")
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile for the model. On the A100 this is "
                        "typically a 10-20%% throughput win (vs 5-15%% on the "
                        "A5000) but adds 1-2 min of warmup on the first epoch.")
    p.add_argument("--export-format", default="onnx",
                   choices=["onnx", "engine", "torchscript", "none"],
                   help="Export format after training. 'engine' is TensorRT FP16, "
                        "the fastest option for inference; 'onnx' is the most "
                        "portable (recommended when you'll deploy off-Colab); "
                        "'none' skips export.")
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
    name = torch.cuda.get_device_name(int(device))
    free, total = torch.cuda.mem_get_info(int(device))
    print(
        f"Using NVIDIA GPU: {name}  "
        f"({free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total)"
    )


def setup_cuda_perf(device: str) -> None:
    """Enable Ampere-class fast paths for matmul / convolutions.

    On the A100 (and the A5000) these knobs are essentially free quality-wise
    but give a real throughput boost. We deliberately do NOT touch
    ``cudnn.benchmark`` here: Ultralytics' AutoBatch routine refuses to run
    with benchmark mode on (it needs deterministic kernel selection to
    measure real memory) and falls back to a hard-coded batch=16, which is
    too big for yolo12x @ 1280 on a 40 GB A100 and triggers an OOM cascade.
    Ultralytics flips benchmark on internally once training is past the
    AutoBatch step.
    """

    if device in {"cpu", "mps"}:
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def main() -> None:
    args = parse_args()
    assert_cuda(args.device)
    setup_cuda_perf(args.device)
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
        # "Person out of water" class. With the A100's headroom we can push
        # mixup / copy-paste a bit harder than on the A5000 — both touch the
        # dataloader more than the GPU, but they generate more candidates per
        # batch which matters for the rare-class recall problem.
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.5, shear=2.0, perspective=0.0005,
        fliplr=0.5, flipud=0.0,
        mosaic=1.0, mixup=0.2, copy_paste=0.4,
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
