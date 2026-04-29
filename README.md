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
- **NVIDIA RTX hardware**, specifically tuned defaults for an RTX 3050 6 GB
  Laptop GPU at `imgsz=640`
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
#    Check yours with `nvidia-smi`. The RTX 3050 Laptop typically runs cu121+.)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install the rest
pip install -r requirements.txt
```

Verify CUDA is visible to PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True NVIDIA GeForce RTX 3050 Laptop GPU
```

## 2. Workflow

The two-stage workflow is the most important thing to internalize: stage 1
gives the model a good prior for what swimmers vs drowning vs out-of-water
look like in *general*, stage 2 specializes that prior to *our* pool.

### Stage 1 — train on the public pool dataset

Place the public-pool dataset under `dataset/` (it is gitignored). Then:

```bash
python src/data_analysis.py --name public                        # sanity-check
python src/train.py                                              # default: yolo12m
```

Outputs land in `runs/detect/stage1_public/`. The best checkpoint is
`runs/detect/stage1_public/weights/best.pt`.

Useful options:

```bash
python src/train.py --model yolo12s.pt --epochs 80 --name stage1_fast   # fast baseline
python src/train.py --resume                                            # resume after interruption
python src/train.py --batch 8                                           # if auto-batch picks too high
```

On an RTX 3050 6 GB, `yolo12m` at `imgsz=640` typically trains at 4–9 minutes
per epoch on ~5k training images, so a full 150-epoch run is 10–22 hours.
Plan for an overnight run.

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

Why fine-tune instead of training from scratch on the merged data?

1. **You don't have enough Uninorte data yet.** Stage 1 gives you a strong
   prior from ~7k public images, stage 2 adapts it with whatever volume you
   collect from our pool.
2. **You want fast iteration on the Uninorte data.** Stage 2 trains in 1–3 h
   with `--freeze 10`, so every time you label more frames you can ship a
   new model the same day.
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
next to the run's plots.

## 3. Inference / demos

The scripts under `scripts/` are intended for quick demos. They are not the
final deployment surface.

### Image inference

```bash
python scripts/predict.py --source uninorte/data --weights best.pt/best.pt --half
```

### Video inference with tracking and drowning alerts

```bash
python scripts/predict_video.py \
    --source uninorte/videos/drowning.mp4 \
    --weights runs/detect/stage2_uninorte/weights/best.pt \
    --half --vid-stride 2 --drowning-threshold 5
```

The `--drowning-threshold N` flag is the one to tune for production: the
script only fires an alert if a tracked person was classified as
`Drowning` in at least N frames out of the last `--smooth-window`
processed frames. With defaults `5 / 10`, that is roughly 1.5 seconds of
sustained signal at 30 fps with `--vid-stride 2` — short enough to react,
long enough to filter single-frame noise.

## 4. Tips for the RTX 3050 (6 GB)

- Always train with `--batch -1` (auto-batch). Manually picking a batch
  size on a 6 GB card is a great way to OOM 40 minutes into training.
- Use `--cache disk`. The 32 GB system RAM is plenty for `disk` caching but
  `ram` caching can race with the CUDA allocator on Windows.
- `--workers 4` (or 6) is the sweet spot. More workers compete with the
  GPU host loop and can actually slow you down.
- For *inference* on the RTX 3050, after you have a final model, export it
  to TensorRT FP16:

  ```bash
  yolo export model=runs/detect/stage2_uninorte/weights/best.pt \
              format=engine half=True imgsz=640 device=0
  ```

  Expect roughly 2× the FPS of plain PyTorch inference, which is the
  difference between "demo" and "deployable monitoring".

## 5. License

This project is distributed under the same license as the upstream H20Saver
fork — see [LICENSE](LICENSE).
