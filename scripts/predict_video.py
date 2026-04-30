"""Run video inference with object tracking and temporal smoothing.

Single-frame YOLO predictions are noisy: the same swimmer can flicker between
"Swimming" and "Drowning" from one frame to the next, which makes raw output
useless for any kind of alerting. This script runs Ultralytics' built-in
ByteTrack/BoT-SORT tracker to assign persistent IDs across frames and then
applies a sliding-window majority vote per ID before deciding what to draw
or alert on.

Typical usage from the repo root:

    python scripts/predict_video.py --source uninorte/videos/drowning.mp4
    python scripts/predict_video.py --source rtsp://camera --weights best.pt/best.pt --half
"""

from __future__ import annotations

import argparse
from collections import Counter, deque, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "best.pt" / "best.pt"

DROWNING_CLASS_NAME = "Drowning"
ALERT_COLOR = (0, 0, 255)        # red
SWIMMING_COLOR = (0, 200, 0)     # green
OUT_OF_WATER_COLOR = (255, 200, 0)  # cyan-ish
DEFAULT_BOX_COLOR = (200, 200, 200)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Video inference with tracking and temporal smoothing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--source", required=True,
                   help="Path to a video file, folder of videos, or stream URL.")
    p.add_argument("--output", default=str(REPO_ROOT / "uninorte" / "results"))
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=960,
                   help="Inference image size. Should match the value the "
                        "checkpoint was trained at (960 by default).")
    p.add_argument("--device", default="0")
    p.add_argument("--half", action="store_true",
                   help="Run inference in FP16. Strongly recommended on the "
                        "A5000 — it nearly halves inference latency with no "
                        "measurable accuracy loss for this task.")
    p.add_argument("--vid-stride", type=int, default=2,
                   help="Process every Nth frame (1 = every frame). Pool footage at "
                        "30 fps does not need every frame for drowning detection. "
                        "On the A5000 you can comfortably set this to 1 if you "
                        "want every-frame coverage.")
    p.add_argument("--smooth-window", type=int, default=10,
                   help="Per-track sliding window length (in processed frames).")
    p.add_argument("--drowning-threshold", type=int, default=5,
                   help="How many frames in the smoothing window must be 'Drowning' "
                        "before raising an alert.")
    p.add_argument("--tracker", default="botsort.yaml",
                   choices=["botsort.yaml", "bytetrack.yaml"])
    p.add_argument("--show", action="store_true",
                   help="Display annotated frames in a window while processing.")
    return p.parse_args()


def color_for(class_name: str, alert: bool) -> tuple[int, int, int]:
    if alert:
        return ALERT_COLOR
    if class_name == "Swimming":
        return SWIMMING_COLOR
    if class_name == "Person out of water":
        return OUT_OF_WATER_COLOR
    if class_name == "Drowning":
        return ALERT_COLOR
    return DEFAULT_BOX_COLOR


def smoothed_class(history: deque[int], names: dict[int, str]) -> tuple[int, str, int]:
    """Return ``(class_id, class_name, votes)`` from a per-track class history."""

    counts = Counter(history)
    cls_id, votes = counts.most_common(1)[0]
    return cls_id, names[cls_id], votes


def draw_box(
    frame: np.ndarray,
    xyxy: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    thick: int = 2,
) -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - text_h - 8)),
                  (x1 + text_w + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")
    source = args.source
    if not (source.startswith("rtsp://") or source.startswith("http")):
        if not Path(source).exists():
            raise SystemExit(f"Source not found: {source}")

    model = YOLO(str(weights))
    drowning_id = next(
        (cid for cid, name in model.names.items() if name == DROWNING_CLASS_NAME),
        None,
    )
    if drowning_id is None:
        print(f"Warning: model has no '{DROWNING_CLASS_NAME}' class; alerts disabled.")

    class_history: dict[int, deque[int]] = defaultdict(
        lambda: deque(maxlen=args.smooth_window)
    )
    alert_counts: dict[int, int] = defaultdict(int)

    out_writer: cv2.VideoWriter | None = None
    out_path = output_dir / (Path(source).stem + "_labeled.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    results = model.track(
        source=source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        vid_stride=args.vid_stride,
        stream=True,
        persist=True,
        tracker=args.tracker,
        verbose=False,
    )

    frame_idx = 0
    alerts_raised = 0
    for result in results:
        frame = result.orig_img.copy()

        if out_writer is None:
            h, w = frame.shape[:2]
            out_writer = cv2.VideoWriter(str(out_path), fourcc, 25.0, (w, h))

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            xyxys = boxes.xyxy.cpu().int().tolist()

            for tid, cls, conf, xyxy in zip(ids, classes, confs, xyxys):
                class_history[tid].append(cls)
                smoothed_id, smoothed_name, votes = smoothed_class(
                    class_history[tid], result.names,
                )
                drowning_votes = (
                    sum(1 for c in class_history[tid] if c == drowning_id)
                    if drowning_id is not None else 0
                )
                alert = drowning_votes >= args.drowning_threshold

                if alert and alert_counts[tid] == 0:
                    alerts_raised += 1
                    print(f"[frame {frame_idx}] DROWNING ALERT — track #{tid} "
                          f"({drowning_votes}/{len(class_history[tid])} votes)")
                alert_counts[tid] = drowning_votes

                label = f"#{tid} {smoothed_name} {conf:.2f}"
                if alert:
                    label = f"!! ALERT #{tid} {smoothed_name}"
                draw_box(frame, tuple(xyxy), label, color_for(smoothed_name, alert))

        cv2.putText(
            frame,
            f"frame {frame_idx}  alerts {alerts_raised}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_writer.write(frame)
        if args.show:
            cv2.imshow("splash", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames, {alerts_raised} alerts raised...")

    if out_writer is not None:
        out_writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"\nDone. {frame_idx} frames processed, {alerts_raised} alerts.")
    print(f"Annotated video saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
