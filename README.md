# Splash — drowning detection for the Uninorte pool

YOLOv12-based detector for three classes inside our university's
training/semi-Olympic pool:

| index | class                |
|-------|----------------------|
| 0     | Drowning             |
| 1     | Person out of water  |
| 2     | Swimming             |

This repo is a fork of the original
[H20Saver](https://github.com/EsonH/H20Saver) project by Eason Huang, fully
refactored for:

- **YOLOv12** (with the new attention-centric backbone) instead of YOLOv11
- **NVIDIA Ampere hardware**, with defaults tuned for an
  **A100 40 GB on Google Colab Pro** running `yolo12l` at `imgsz=1280`
  (the previous A5000 default was `yolo12l` @ 960 — same backbone,
  bigger images, free quality bump from the A100's headroom). `yolo12x`
  @ 1280 needs the 80 GB A100 slice on Pro+. The RTX A5000 16 GB and
  RTX 3050 6 GB recipes are still one CLI flag away — see the fallback
  table below
- **Two-stage training**: a generic public-pool model first, then a
  domain-adaptation fine-tune on our actual Uninorte data
- **Real-time inference** with object tracking + temporal smoothing, so
  single-frame flicker does not trigger false drowning alerts

## Repository layout

```
splash/
├── src/
│   ├── train.py              # stage 1: train YOLOv12 on the public dataset
│   ├── finetune.py           # stage 2: fine-tune on the Uninorte dataset
│   ├── data_analysis.py      # class / bbox stats for any YOLO dataset
│   └── model_evaluation.py   # per-class metrics report on val/test
├── scripts/
│   ├── predict.py            # image inference (folder or single image)
│   └── predict_video.py      # video inference with tracking + alerting
├── dataset/                  # public-pool dataset (gitignored)
│   └── data.yaml
├── uninorte_dataset/         # Uninorte dataset (gitignored, see its README)
│   ├── data.yaml
│   └── README.md
├── runs/                     # all training / inference outputs (gitignored)
├── best.pt/                  # legacy upstream HF weights (kept for now)
├── requirements.txt
├── setup.py
└── README.md
```

## 1. Install

The training stack expects an NVIDIA GPU with a CUDA build of PyTorch.

### A. Google Colab (recommended for training)

Pick **Runtime → Change runtime type → GPU → A100** (Pro/Pro+ tier required).
Then in a notebook cell:

```bash
# 1. Verify the A100 actually got attached
!nvidia-smi | head -n 20

# 2. Mount Drive so weights and datasets persist across sessions
from google.colab import drive
drive.mount("/content/drive")

# 3. Clone the repo into Drive (one-time) and cd into it
%cd /content/drive/MyDrive
!git clone https://github.com/<your-fork>/splash.git || echo "already cloned"
%cd splash

# 4. Install Python deps. Colab already ships a CUDA PyTorch matching its
#    driver, so we don't reinstall it — just the rest:
!pip install -q -r requirements.txt
```

Verify CUDA + the A100 are visible to PyTorch:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
# expected: True NVIDIA A100-SXM4-40GB
```

If `nvidia-smi` shows you got a T4 / V100 instead, change the runtime and
restart — the defaults in this repo will OOM on a 16 GB card.

### B. Local workstation / laptop

```bash
# 1. Create a clean environment
python -m venv .venv
.\.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate       # Linux / macOS

# 2. Install PyTorch with CUDA support FIRST (use the variant matching your
#    installed driver — pick one of:
#      cu121  for CUDA 12.1
#      cu124  for CUDA 12.4
#      cu126  for CUDA 12.6
#    Check yours with `nvidia-smi`. Most A100 / A5000 hosts run cu121+.)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install the rest
pip install -r requirements.txt
```

For the fastest deployment-time inference you'll also want NVIDIA TensorRT
installed so `yolo export ... format=engine` works. It's optional for
training; install it only when you want to convert a `.pt` checkpoint into a
`.engine` file:

```bash
pip install tensorrt
```

## 2. Workflow

The two-stage workflow is the most important thing to internalize: stage 1
gives the model a good prior for what swimmers vs drowning vs out-of-water
look like in *general*, stage 2 specializes that prior to *our* pool.

### Stage 1 — train on the public pool dataset

Place the public-pool dataset under `dataset/` (it is gitignored). On
Colab, the easiest path is to upload the Roboflow ZIP to Drive once and
then symlink it under `dataset/` so re-cloning the repo doesn't drag the
images along. Then:

```bash
python src/data_analysis.py --name public                        # sanity-check
python src/train.py                                              # default: yolo12l @ 1280 (Colab A100 40 GB)
```

Outputs land in `runs/detect/stage1_public/`. The best checkpoint is
`runs/detect/stage1_public/weights/best.pt`.

Useful options:

```bash
python src/train.py --imgsz 960  --batch 32 --epochs 80 --name stage1_fast        # faster baseline
python src/train.py --batch 16                                                    # explicit batch (skip auto-batch)
python src/train.py --cache disk                                                  # safer than ram on retry-prone runs
python src/train.py --multi-scale                                                 # +/-50% imgsz jitter
python src/train.py --compile                                                     # torch.compile (10-20% speedup)
python src/train.py --export-format engine                                        # TensorRT FP16 after training
python src/train.py --resume                                                      # resume after interruption
```

On a Colab A100 40 GB, `yolo12l` at `imgsz=1280` typically trains at
2–3 minutes per epoch on ~5 k training images, so a full 150-epoch run
is roughly 5–8 hours. With `patience=25` early-stopping it usually wraps
in 3–5. **Keep the tab open** — Colab silently times sessions out after
~12 h of idle UI; if it disconnects mid-run, just re-run with `--resume`
to pick up at the last checkpoint.

> **Heads up on `--batch -1` (auto-batch).** Ultralytics' AutoBatch
> probes VRAM with `cudnn.benchmark = False`. If you've enabled benchmark
> mode somewhere upstream, AutoBatch silently falls back to a hard-coded
> `batch=16`, which OOMs the bigger models at imgsz=1280. The repo's
> `setup_cuda_perf()` deliberately leaves benchmark alone for this reason.
> If AutoBatch still misbehaves on your machine, just pass `--batch 16`
> (or whatever you measured) explicitly.

#### GPU-fallback cheatsheet

If you're not on a Colab A100, override the defaults explicitly:

| GPU                        | Recommended override                                                            |
|----------------------------|---------------------------------------------------------------------------------|
| **A100 40 GB** (default)   | *(no flags needed — repo defaults)*                                             |
| A100 80 GB (Colab Pro+)    | `--model yolo12x.pt --batch 16` (the extra VRAM unlocks the bigger model)       |
| RTX A5000 16 GB            | `--imgsz 960 --workers 8 --cache disk --batch 8`                                |
| RTX A6000 / 4090 24 GB     | `--batch 12` (or `--model yolo12x.pt --imgsz 1024`)                             |
| RTX 3050 6 GB              | `--model yolo12m.pt --imgsz 640 --workers 4 --cache disk --batch 4`             |

### Stage 2 — fine-tune on the Uninorte dataset

Once your Roboflow export is unpacked into `uninorte_dataset/` (see
[`uninorte_dataset/README.md`](uninorte_dataset/README.md) for the exact
layout and class-order requirement):

```bash
python src/data_analysis.py --data uninorte_dataset/data.yaml --name uninorte
python src/finetune.py --weights runs/detect/stage1_public/weights/best.pt
```

Outputs land in `runs/detect/stage2_uninorte/`. This is the model you should
deploy.

The fine-tune defaults to the same `imgsz=1280` as stage 1 — keep them
matched, or stage 1's head-resolution prior is mostly thrown away.

Why fine-tune instead of training from scratch on the merged data?

1. **You don't have enough Uninorte data yet.** Stage 1 gives you a strong
   prior from ~7 k public images, stage 2 adapts it with whatever volume you
   collect from our pool.
2. **You want fast iteration on the Uninorte data.** Stage 2 trains in
   ~30–90 min on the A100 with `--freeze 10`, so every time you label
   more frames you can ship a new model the same hour.
3. **It is the recipe with the highest expected accuracy.** Domain
   adaptation from a strong general detector to a small in-domain dataset
   beats training from scratch on small data.

### Evaluating a checkpoint

```bash
python src/model_evaluation.py \
    --weights runs/detect/stage2_uninorte/weights/best.pt \
    --data uninorte_dataset/data.yaml \
    --split test
```

This prints per-class P / R / mAP and writes `report.json` and `report.csv`
next to the run's plots. FP16 evaluation is on by default on the A100 —
pass `--no-half` if you want to compare against an FP32 baseline.

## 3. Inference / demos

The scripts under `scripts/` are intended for quick demos. They are not the
final deployment surface.

### Image inference

```bash
python scripts/predict.py --source uninorte/data --weights best.pt/best.pt
```

`--half` is on by default; pass `--no-half` to disable. Pass `--imgsz 960`
if you're running an older A5000-trained checkpoint.

### Video inference with tracking and drowning alerts

```bash
python scripts/predict_video.py \
    --source uninorte/videos/drowning.mp4 \
    --weights runs/detect/stage2_uninorte/weights/best.pt \
    --drowning-threshold 5
```

Defaults: `--imgsz 1280 --half --vid-stride 1`. The A100 chews through
1080p every-frame easily; if you're deploying on a smaller card or
running multiple streams, drop `--vid-stride` to 2.

The `--drowning-threshold N` flag is the one to tune for production: the
script only fires an alert if a tracked person was classified as
`Drowning` in at least N frames out of the last `--smooth-window`
processed frames. With defaults `5 / 10`, that is roughly 0.3 s of
sustained signal at 30 fps with `--vid-stride 1` — short enough to react,
long enough to filter single-frame noise. Bump the threshold to `7-8` if
you see false positives in busy multi-swimmer scenes.

## 4. Tips for the NVIDIA A100 (40 GB)

- **Auto-batch with a fraction.** `--batch 0.85` tells Ultralytics to fill
  ~85% of VRAM instead of the conservative 60% default. Safe on Colab
  because nothing else competes for the GPU; not recommended on a shared
  workstation. The default (`--batch -1`) leaves headroom and is the
  right call for the very first run. **Caveat**: AutoBatch can fall back
  to a hard-coded `batch=16` if `cudnn.benchmark` is true; if you see
  that warning, pass `--batch <int>` explicitly.
- **`--cache ram` is the new default.** Colab A100 hosts ship with ~80 GB
  of system RAM and a 5 k-image dataset at imgsz=1280 needs ~25 GB; epoch
  starts are noticeably faster than `--cache disk`. **However**: if your
  first training attempt OOMs and Ultralytics retries with a smaller
  batch, the previous RAM cache isn't always released — successive
  retries chew through host RAM until caching silently disables itself.
  If you hit that pattern, switch to `--cache disk`; the on-disk cache
  is per-image and survives retries cleanly.
- **`--workers 12`** matches the Colab A100 host's vCPU count. On a
  laptop or shared workstation with fewer cores, drop it to 8 or you
  starve the dataloader.
- **TF32 is enabled automatically.** The training/eval/inference scripts
  flip `torch.backends.cuda.matmul.allow_tf32 = True` at startup so
  Ampere Tensor Cores get used even for the FP32-cast paths inside the
  model. Free 1.5–2× speedup on matmul-heavy ops.
- **`--multi-scale` is now affordable.** With 40 GB of headroom the
  +/-50% imgsz jitter rarely OOMs the auto-batcher; opt in once you have
  a stable batch size from a previous run for an extra +1–2 mAP50 on
  small classes.
- **`--compile` is worth it.** torch.compile gives a 10–20% throughput
  win on the A100 (vs 5–15% on the A5000). Costs 1–2 min of warmup on
  the first epoch. Disable it if you hit the rare Ultralytics op that
  isn't compile-safe.
- **For deployment** (not training), export to TensorRT FP16 on whatever
  GPU you'll actually deploy on — engine files are device-specific:

  ```bash
  yolo export model=runs/detect/stage2_uninorte/weights/best.pt \
              format=engine half=True imgsz=1280 device=0
  ```

  Or just pass `--export-format engine` to `train.py` / `finetune.py` so
  it happens automatically at the end of training. Expect roughly 2–3×
  the FPS of plain PyTorch inference, which is the difference between
  "demo" and "deployable monitoring".

### Colab gotchas

- Sessions disconnect after a few hours of inactive UI even on Pro+.
  Use `--resume` to pick up training from the last checkpoint.
- Don't write outputs straight to `/content/` — that's wiped on every
  session restart. Always train into a folder under
  `/content/drive/MyDrive/...`.
- `cv2.imshow` doesn't work in Colab; the `--show` flag in
  `scripts/predict_video.py` is a no-op there. The annotated `.mp4` is
  always written to `--output` regardless.
- If the A100 you got reports < 40 GB free in `nvidia-smi`, you have a
  partial-MIG slice. Restart the runtime and try again, or pass
  `--batch 8 --imgsz 960` to fit a smaller slice.

## 5. License

This project is distributed under the same license as the upstream H20Saver
fork — see [LICENSE](LICENSE).
