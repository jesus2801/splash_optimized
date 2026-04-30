"""Run image inference with a trained YOLO checkpoint.

Loops over a folder of images (or a single image) and saves annotated copies
plus a JSON detection report. Intended for quick demos and visual sanity
checks against the Uninorte test images.

Defaults assume the new A100 training defaults — ``imgsz=1280`` and
``--half`` on. If you're loading a checkpoint trained at a different
resolution (e.g. the legacy 960 weights), pass ``--imgsz 960`` so the
inference resolution matches the training resolution.

Typical usage from the repo root:

    python scripts/predict.py --source uninorte/data --weights best.pt/best.pt
    python scripts/predict.py --source path/to/img.jpg --conf 0.5 --no-half
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "best.pt" / "best.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Image inference with a trained YOLO model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                   help="Path to the .pt checkpoint to load.")
    p.add_argument("--source", required=True,
                   help="Path to an image, a folder, or a glob (e.g. uninorte/data).")
    p.add_argument("--output", default=str(REPO_ROOT / "uninorte" / "results"),
                   help="Output folder for annotated images and report.json.")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Inference image size. Should match the value the "
                        "checkpoint was trained at (1280 by default for A100 "
                        "weights; pass 960 for older A5000 weights).")
    p.add_argument("--device", default="0")
    p.add_argument("--half", dest="half", action="store_true", default=True,
                   help="Run inference in FP16. ON by default — on the A100 (and "
                        "A5000) it ~halves inference latency with no measurable "
                        "accuracy loss for this task.")
    p.add_argument("--no-half", dest="half", action="store_false",
                   help="Disable FP16 inference (use FP32 — only useful for "
                        "comparison or on cards without good FP16 throughput).")
    return p.parse_args()


def setup_cuda_perf(device: str) -> None:
    if device in {"cpu", "mps"}:
        return
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_cuda_perf(args.device)

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        verbose=False,
        stream=True,
    )

    summary: list[dict] = []
    for result in results:
        annotated = result.plot()
        src_path = Path(result.path)
        out_path = output_dir / src_path.name
        cv2.imwrite(str(out_path), annotated)

        detections = []
        for box in result.boxes:
            cls = int(box.cls[0])
            detections.append({
                "class_id": cls,
                "class": result.names[cls],
                "confidence": float(box.conf[0]),
                "xyxy": [float(v) for v in box.xyxy[0].tolist()],
            })
        summary.append({
            "image": str(src_path),
            "annotated": str(out_path),
            "detections": detections,
        })
        print(f"{src_path.name}: {len(detections)} detections -> {out_path.name}")

    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
