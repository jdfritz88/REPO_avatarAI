#!/usr/bin/env bash
# Setup MuseTalk V1.5 for lip-sync animation (replaces SadTalker)
# Run from the project root:  bash scripts/setup_musetalk.sh
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
MODELS_DIR="$BACKEND_DIR/models"
MUSETALK_DIR="$MODELS_DIR/MuseTalk"
VENV_PYTHON="$BACKEND_DIR/venv/bin/python"
SENTINEL="$PROJECT_ROOT/.musetalk_ready"

if [ ! -f "$VENV_PYTHON" ]; then
  VENV_PYTHON="$BACKEND_DIR/venv/Scripts/python.exe"
fi
if [ ! -f "$VENV_PYTHON" ]; then
  VENV_PYTHON="python3"
fi

echo "=== MuseTalk V1.5 Setup ==="
echo "Project : $PROJECT_ROOT"
echo "Backend : $BACKEND_DIR"
echo "Python  : $VENV_PYTHON"
echo ""

# ── 1. Clone MuseTalk ────────────────────────────────────────────────────────
if [ -d "$MUSETALK_DIR/.git" ]; then
  echo "[1/5] MuseTalk already cloned — pulling latest..."
  git -C "$MUSETALK_DIR" pull --ff-only
else
  echo "[1/5] Cloning MuseTalk..."
  mkdir -p "$MODELS_DIR"
  git clone https://github.com/TMElyralab/MuseTalk.git "$MUSETALK_DIR"
fi

# ── 2. Install Python dependencies ──────────────────────────────────────────
echo ""
echo "[2/5] Installing MuseTalk requirements..."

# Install requirements, skipping packages incompatible with Python 3.12
# (mmpose/mmcv have no 3.12 wheels; tensorflow is optional for our use case)
"$VENV_PYTHON" -m pip install -q \
  "accelerate==0.28.0" \
  "opencv-python==4.9.0.80" \
  "soundfile==0.12.1" \
  "librosa==0.11.0" \
  "einops==0.8.1" \
  "omegaconf" \
  "pyyaml" \
  "imageio" \
  "imageio[ffmpeg]" \
  "ffmpeg-python" \
  "moviepy<2" \
  "mediapipe" \
  "face-alignment" \
  "safetensors" \
  "timm"

# ── 3. Replace preprocessing.py with CPU-compatible version ─────────────────
echo ""
echo "[3/5] Writing CPU-compatible preprocessing.py (replaces mmpose with face_alignment)..."

PREPROCESS="$MUSETALK_DIR/musetalk/utils/preprocessing.py"
# Always overwrite — idempotent, works on first run and re-runs
cp "$PREPROCESS" "${PREPROCESS}.orig" 2>/dev/null || true
cat > "$PREPROCESS" << 'PYEOF'
# preprocessing.py — rewritten for Python 3.12 / CPU-only compatibility
# Uses face_alignment (pip install face-alignment) instead of mmpose/mmcv.
#
# NOTE: this module is imported by the persistent musetalk_worker.py, whose
# protocol reserves stdout exclusively for "READY" and per-job JSON replies
# (see musetalk_worker.py's docstring). Every print() below therefore writes
# to sys.stderr, not stdout — a stray stdout line here gets read by the
# parent (app/services/animator.py) as if it were a job's JSON reply,
# desyncing the protocol for the rest of the worker's lifetime.
import os
import sys
import json
import pickle
import numpy as np
import cv2
import torch
from tqdm import tqdm
import face_alignment

# Initialise face alignment on CPU or GPU automatically
_device = "cuda" if torch.cuda.is_available() else "cpu"
_fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    device=_device,
)

# Sentinel value used by callers when no face is detected in a frame
coord_placeholder = (0.0, 0.0, 0.0, 0.0)


def resize_landmark(landmark, w, h, new_w, new_h):
    landmark_norm = landmark / np.array([w, h])
    return landmark_norm * np.array([new_w, new_h])


def read_imgs(img_list):
    frames = []
    print("reading images...", file=sys.stderr)
    for img_path in tqdm(img_list, file=sys.stderr):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def _detect_face_bbox(frame_bgr, bbox_shift=0):
    """Detect face bounding box using 68-point landmarks. Returns (x1,y1,x2,y2) or None."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    preds = _fa.get_landmarks(frame_rgb)
    if preds is None or len(preds) == 0:
        return None
    lm = preds[0]  # (68, 2)
    x1, y1 = int(lm[:, 0].min()), int(lm[:, 1].min())
    x2, y2 = int(lm[:, 0].max()), int(lm[:, 1].max())
    y1 = max(0, y1 + bbox_shift)
    if x2 <= x1 or y2 <= y1 or x1 < 0:
        return None
    return (x1, y1, x2, y2)


def get_landmark_and_bbox(img_list, upperbondrange=0):
    """
    Detect face bounding boxes for all images.
    Returns (coords_list, frames).
    """
    frames = read_imgs(img_list)
    coords_list = []
    print(f"Getting face bounding boxes (bbox_shift={upperbondrange})..." if upperbondrange != 0
          else "Getting face bounding boxes...", file=sys.stderr)
    for frame in tqdm(frames, file=sys.stderr):
        bbox = _detect_face_bbox(frame, bbox_shift=upperbondrange)
        coords_list.append(bbox if bbox is not None else coord_placeholder)
    return coords_list, frames


def get_bbox_range(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    deltas = []
    for frame in tqdm(frames, file=sys.stderr):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = _fa.get_landmarks(frame_rgb)
        if preds is None or len(preds) == 0:
            continue
        lm = preds[0]
        deltas.append(int(lm[:, 1].max()) - int(lm[:, 1].mean()))
    if not deltas:
        return "No faces detected"
    avg = int(sum(deltas) / len(deltas))
    return f"Total frame: {len(frames)}  Adjust range: [-{avg}~{avg}]  current value: {upperbondrange}"
PYEOF
echo "  preprocessing.py replaced ✓"

# ── 3b. Install our custom persistent worker ────────────────────────────────
# musetalk_worker.py is OUR driver (loads the models once and serves inference
# jobs over stdin/stdout), not part of upstream MuseTalk — so the git clone
# above doesn't include it. Copy the tracked copy from the backend into the
# clone's scripts/ dir. The backend can also run it from its tracked location,
# but copying keeps everything self-contained under the MuseTalk repo.
echo ""
echo "[3b/5] Installing persistent MuseTalk worker (musetalk_worker.py)..."
WORKER_SRC="$BACKEND_DIR/musetalk_worker.py"
WORKER_DST="$MUSETALK_DIR/scripts/musetalk_worker.py"
if [ -f "$WORKER_SRC" ]; then
  cp "$WORKER_SRC" "$WORKER_DST"
  echo "  musetalk_worker.py installed ✓"
else
  echo "  ERROR: $WORKER_SRC not found — is the repo checkout complete?" >&2
  exit 1
fi

# ── 3c. Patch resnet.py for PyTorch 2.6's torch.load default change ────────
# PyTorch 2.6 flipped torch.load's weights_only default to True, which can't
# parse the ResNet18 backbone checkpoint's legacy .tar serialization format
# (downloaded step 4 below, straight from download.pytorch.org — a trusted
# source, so weights_only=False is safe). Without this the worker crashes on
# startup and the animator silently falls back to no-lip-sync mode.
echo ""
echo "[3c/5] Patching resnet.py for PyTorch 2.6 torch.load compatibility..."
RESNET_PY="$MUSETALK_DIR/musetalk/utils/face_parsing/resnet.py"
if [ -f "$RESNET_PY" ] && ! grep -q "weights_only=False" "$RESNET_PY"; then
  sed -i 's/torch\.load(model_path)/torch.load(model_path, weights_only=False)/' "$RESNET_PY"
  echo "  resnet.py patched ✓"
else
  echo "  resnet.py already patched or not found — skipping"
fi

# ── 3d. Patch load_all_model()'s stray stdout print ─────────────────────────
# musetalk_worker.py's protocol reserves stdout exclusively for "READY" and
# per-job JSON replies (see its own docstring). utils.py's load_all_model()
# print()s a diagnostic line to stdout during startup, which the parent
# (app/services/animator.py) reads as the READY handshake line instead —
# it kills the worker thinking it failed to start, every single time.
echo ""
echo "[3d/5] Patching load_all_model()'s stray stdout print..."
UTILS_PY="$MUSETALK_DIR/musetalk/utils/utils.py"
if [ -f "$UTILS_PY" ] && grep -q 'print(f"load unet model from' "$UTILS_PY"; then
  sed -i 's/print(f"load unet model from {unet_model_path}")/import sys as _sys; _sys.stderr.write(f"load unet model from {unet_model_path}\\n")/' "$UTILS_PY"
  echo "  utils.py patched ✓"
else
  echo "  utils.py already patched or not found — skipping"
fi

# ── 4. Download model weights from HuggingFace ──────────────────────────────
echo ""
echo "[4/5] Downloading MuseTalk model weights (~8.8 GB total)..."

# gdown is required for the Google Drive face parsing model
"$VENV_PYTHON" -m pip install -q gdown

"$VENV_PYTHON" - "$MUSETALK_DIR" << 'PYEOF'
from huggingface_hub import snapshot_download
import os, sys, urllib.request, subprocess

musetalk_dir = sys.argv[1]
models_target = os.path.join(musetalk_dir, "models")
os.makedirs(models_target, exist_ok=True)

# ── MuseTalk weights (unet + VAE) ────────────────────────────────────────────
print("  Downloading TMElyralab/MuseTalk weights (~7 GB)...")
snapshot_download(
    repo_id="TMElyralab/MuseTalk",
    local_dir=models_target,
    ignore_patterns=["*.md", "*.txt", "*.gitattributes"],
)
print("  MuseTalk weights done.")

# ── Whisper-tiny (audio feature extractor, ~150 MB) ─────────────────────────
whisper_target = os.path.join(models_target, "whisper")
if not os.path.isdir(whisper_target) or not os.listdir(whisper_target):
    print("  Downloading openai/whisper-tiny (~150 MB)...")
    snapshot_download(
        repo_id="openai/whisper-tiny",
        local_dir=whisper_target,
        ignore_patterns=["*.md", "*.gitattributes", "flax_model*", "tf_model*", "rust_model*"],
    )
    print("  Whisper-tiny done.")
else:
    print("  Whisper-tiny already present — skipping.")

# ── SD-VAE (stabilityai/sd-vae-ft-mse, ~335 MB) ─────────────────────────────
vae_target = os.path.join(models_target, "sd-vae")
if not os.path.isdir(vae_target) or not os.listdir(vae_target):
    print("  Downloading stabilityai/sd-vae-ft-mse (~335 MB)...")
    snapshot_download(
        repo_id="stabilityai/sd-vae-ft-mse",
        local_dir=vae_target,
        ignore_patterns=["*.md", "*.gitattributes"],
    )
    print("  SD-VAE done.")
else:
    print("  SD-VAE already present — skipping.")

# ── face-parse-bisent (BiSeNet face segmentation) ────────────────────────────
bisent_dir = os.path.join(models_target, "face-parse-bisent")
os.makedirs(bisent_dir, exist_ok=True)

resnet_path = os.path.join(bisent_dir, "resnet18-5c106cde.pth")
if not os.path.isfile(resnet_path):
    print("  Downloading ResNet18 backbone (~45 MB)...")
    urllib.request.urlretrieve(
        "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        resnet_path,
    )
    print("  ResNet18 done.")
else:
    print("  resnet18-5c106cde.pth already present — skipping.")

bisenet_path = os.path.join(bisent_dir, "79999_iter.pth")
if not os.path.isfile(bisenet_path):
    print("  Downloading BiSeNet face parser via gdown (~53 MB)...")
    import gdown
    gdown.download(id="154JgKpzCPW82qINcVieuPH3fZ2e0P812", output=bisenet_path, quiet=False)
    print("  BiSeNet done.")
else:
    print("  79999_iter.pth already present — skipping.")

print("  All model downloads complete.")
PYEOF

# ── 5. Verify and write sentinel ─────────────────────────────────────────────
echo ""
echo "[5/5] Verifying installation..."

MISSING=0

if [ ! -f "$MUSETALK_DIR/scripts/inference.py" ]; then
  echo "  MISSING: scripts/inference.py"
  MISSING=1
else
  echo "  scripts/inference.py ✓"
fi

if [ ! -f "$MUSETALK_DIR/scripts/musetalk_worker.py" ]; then
  echo "  MISSING: scripts/musetalk_worker.py"
  MISSING=1
else
  echo "  scripts/musetalk_worker.py ✓"
fi

if [ ! -f "$MUSETALK_DIR/models/musetalkV15/unet.pth" ]; then
  echo "  MISSING: models/musetalkV15/unet.pth"
  MISSING=1
else
  echo "  models/musetalkV15/unet.pth ✓"
fi

if [ ! -d "$MUSETALK_DIR/models/whisper" ]; then
  echo "  MISSING: models/whisper/"
  MISSING=1
else
  echo "  models/whisper/ ✓"
fi

if [ ! -f "$MUSETALK_DIR/models/sd-vae/config.json" ]; then
  echo "  MISSING: models/sd-vae/config.json"
  MISSING=1
else
  echo "  models/sd-vae/ ✓"
fi

if [ ! -f "$MUSETALK_DIR/models/face-parse-bisent/resnet18-5c106cde.pth" ]; then
  echo "  MISSING: models/face-parse-bisent/resnet18-5c106cde.pth"
  MISSING=1
elif [ ! -f "$MUSETALK_DIR/models/face-parse-bisent/79999_iter.pth" ]; then
  echo "  MISSING: models/face-parse-bisent/79999_iter.pth"
  MISSING=1
else
  echo "  models/face-parse-bisent/ ✓"
fi

if [ "$MISSING" -eq 0 ]; then
  touch "$SENTINEL"
  echo ""
  echo "  Sentinel written: $SENTINEL"
  echo ""
  echo "=== MuseTalk setup complete! ==="
  echo "Set AVATAR_ENGINE=musetalk in .env and restart the backend."
else
  echo ""
  echo "Some checks failed — see above. Fix and re-run."
  exit 1
fi
