"""
Post-processing blink compositing for rendered avatar video.

MuseTalk (and Wav2Lip, and every other mouth-crop lip-sync model in this
class — confirmed against MuseTalk's own BBOX_SHIFT.md docs) never touches
anything above the mouth: the mask boundary is pinned around landmark
points ~28-30, well below the eyes, and that boundary is not adjustable
past that point. The source photo's eyes are pasted into every output
frame completely unchanged, so these models cannot be made to blink no
matter how the mask parameters are tuned.

This module adds real blinking as a separate, deterministic compositing
pass over the ALREADY-RENDERED video, independent of whichever engine
produced it (MuseTalk, the simple static-image fallback, or anything
added later) — because the eye region is pixel-identical in every frame,
there's nothing to re-detect per frame:

  1. Detect precise eye landmarks on the avatar's source photo (once).
  2. Sample a real skin patch from just above each eye (brow/lid skin,
     already free of eyelashes) from that same photo.
  3. On a human-realistic schedule (~15-20 blinks/min, ~150ms each), slide
     that skin patch down over the eye for a few consecutive frames with a
     feathered blend, then remove it — using the person's own skin texture
     rather than a generic vertical squash of the eyeball (which is what
     makes naive blink filters look obviously fake).

Never raises: a missing blink is far better than a broken video, so any
failure (no face detected, corrupt video, ffmpeg mux failure) falls back
to returning the input untouched.
"""
import logging
import random
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_fa = None  # lazy — FaceAlignment loads its own model weights on first use


def _get_fa():
    global _fa
    if _fa is None:
        import face_alignment

        _fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    return _fa


# Standard 68-point scheme (dlib-compatible, what face_alignment returns):
_RIGHT_EYE = list(range(36, 42))
_LEFT_EYE = list(range(42, 48))


def _eye_regions(image_bgr: np.ndarray) -> list:
    """Return [{'eye_box':..., 'patch_box':...}, ...] for both eyes, or [] if no face found."""
    fa = _get_fa()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    preds = fa.get_landmarks(rgb)
    if not preds:
        return []

    landmarks = preds[0]
    h, w = image_bgr.shape[:2]
    regions = []
    for eye_idx in (_RIGHT_EYE, _LEFT_EYE):
        pts = landmarks[eye_idx]
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        eye_w, eye_h = x2 - x1, y2 - y1
        if eye_w <= 0 or eye_h <= 0:
            continue
        # Pad generously — the lid needs to cover brow-to-lash, not just the
        # tight landmark box (which hugs the visible eyeball only).
        pad_x = eye_w * 0.35
        pad_y_top = eye_h * 1.7  # reaches up into the brow for the skin patch
        pad_y_bot = eye_h * 0.55
        ex1 = max(0, int(x1 - pad_x))
        ex2 = min(w, int(x2 + pad_x))
        ey1 = max(0, int(y1 - pad_y_top))
        ey2 = min(h, int(y2 + pad_y_bot))
        # Skin patch: same width, sampled from directly above the eye box
        # (brow skin) — this is what gets slid down to simulate the lid.
        patch_h = ey2 - ey1
        py1 = max(0, ey1 - patch_h)
        py2 = ey1
        if py2 - py1 != ey2 - ey1 or ex2 - ex1 <= 0:
            continue  # too close to the image edge to get a clean patch
        regions.append({"eye_box": (ex1, ey1, ex2, ey2), "patch_box": (ex1, py1, ex2, py2)})
    return regions


def _apply_blink_frame(frame: np.ndarray, regions: list, closedness: float) -> np.ndarray:
    """closedness in [0, 1]: 0 = untouched, 1 = eye fully covered by the lid patch."""
    if closedness <= 0:
        return frame
    out = frame
    for r in regions:
        ex1, ey1, ex2, ey2 = r["eye_box"]
        px1, py1, px2, py2 = r["patch_box"]
        eye_h = ey2 - ey1
        cover_h = max(1, int(round(eye_h * closedness)))
        cover_h = min(cover_h, py2 - py1, ey2 - ey1)

        # Bottom `cover_h` px of the skin patch (closest to the lash line),
        # slid down to cover the top `cover_h` px of the eye box.
        patch = frame[py2 - cover_h : py2, px1:px2]
        if patch.shape[0] != cover_h or patch.shape[1] != (ex2 - ex1):
            continue  # degenerate crop at a frame edge — skip, don't distort

        target_y1, target_y2 = ey1, ey1 + cover_h
        feather = max(1, cover_h // 4)
        alpha = np.ones((cover_h, 1), dtype=np.float32)
        if feather < cover_h:
            alpha[-feather:, 0] = np.linspace(1.0, 0.15, feather)
        alpha = np.repeat(alpha, patch.shape[1], axis=1)[:, :, None]

        region = out[target_y1:target_y2, ex1:ex2].astype(np.float32)
        blended = patch.astype(np.float32) * alpha + region * (1 - alpha)
        out[target_y1:target_y2, ex1:ex2] = blended.astype(np.uint8)
    return out


def _blink_schedule(total_frames: int, fps: float) -> dict:
    """Map frame_index -> closedness (0..1) for a natural blink cadence."""
    schedule: dict = {}
    rng = random.Random(0)  # deterministic per render, still looks natural
    t = rng.uniform(0.4, 1.2)  # first blink shouldn't always land at frame 0
    duration_s = total_frames / fps
    # ~150ms blink at 25fps ≈ 4 frames: down, closed, closed, opening, open
    shape = [0.4, 1.0, 1.0, 0.5, 0.15]
    while t < duration_s:
        start = int(t * fps)
        for i, c in enumerate(shape):
            idx = start + i
            if idx < total_frames:
                schedule[idx] = c
        t += rng.uniform(2.6, 4.2)  # ~15-20 blinks/min while attentive/still
    return schedule


def apply_blinks(video_path: str, avatar_image_path: str, output_path: str) -> str:
    """
    Composite natural eye blinks onto an already-rendered avatar video.
    Returns output_path on success, or the original video_path unchanged
    if blinking couldn't be applied for any reason (never raises).
    """
    try:
        src_image = cv2.imread(avatar_image_path)
        if src_image is None:
            logger.warning(f"Blink: could not read avatar image {avatar_image_path}")
            return video_path

        regions = _eye_regions(src_image)
        if not regions:
            logger.warning("Blink: no face landmarks found on avatar image — skipping")
            return video_path

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Blink: could not open {video_path} — skipping")
            return video_path

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        schedule = _blink_schedule(total_frames, fps)
        logger.info(f"Blink: {len(schedule)} composited frames across {total_frames} total")

        tmp_video = str(Path(output_path).with_suffix("")) + "_blinkvideo.mp4"
        writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            closedness = schedule.get(idx, 0.0)
            if closedness > 0:
                frame = _apply_blink_frame(frame, regions, closedness)
            writer.write(frame)
            idx += 1
        cap.release()
        writer.release()

        # Re-attach the original audio (OpenCV's VideoWriter drops it).
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "warning",
                "-i", tmp_video,
                "-i", video_path,
                "-c:v", "copy", "-c:a", "aac",
                "-map", "0:v:0", "-map", "1:a:0?",
                "-shortest",
                output_path,
            ],
            capture_output=True,
        )
        Path(tmp_video).unlink(missing_ok=True)
        if result.returncode != 0:
            logger.warning(
                f"Blink: ffmpeg mux failed, using un-blinked video: "
                f"{result.stderr.decode(errors='replace')[:500]}"
            )
            return video_path

        return output_path

    except Exception as e:
        logger.error(f"Blink compositing failed, using un-blinked video: {e}")
        return video_path
