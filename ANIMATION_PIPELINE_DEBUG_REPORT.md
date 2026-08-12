# MuseTalk Animation/Lip-Sync Pipeline — Debug Report

**Date:** 2026-08-11
**Repo:** `F:\Apps\freedom_system\REPO_punithVT` (fork of `PunithVT/ai-avatar-system`, pushed to `https://github.com/jdfritz88/REPO_punithVT`)
**Goal:** Verify that MuseTalk-based lip-sync animation actually works end-to-end using a real avatar photo (a Kindroid participant named "Silva"), and that only the active speaker gets animated (not idle/listening participants).
**Status: NOT YET CONFIRMED WORKING.** Testing was stopped by the user mid-run after 5 real bugs were found and fixed. The pipeline got further than ever before (past model loading, into actual GPU inference) but never produced a confirmed successful video output before the test was halted. One fix (ffmpeg PATH) has **mixed/inconclusive evidence** — see Bug #5.

---

## Test harness

A standalone script, `backend/_test_animation.py` (not part of the app, written for this debugging session — still present on disk), was used to drive a real end-to-end test:

1. `POST /api/v1/sessions/create` with `avatar_id` = Silva's uploaded avatar (`00000000-0000-0000-0000-000000000000`)
2. `PATCH /api/v1/sessions/{id}/settings` with `{"participant_ids": ["kindroid-silva"], "turn_mode": "round_robin", "addressing_enabled": true}`
3. Open a real WebSocket to `/ws/session/{id}`, send `{"type": "text", "text": "Hi Silva, quick animation test — say hello!"}`
4. Listen for WS events, specifically waiting for a `video_chunk` event (proof the animation pipeline produced real output) and asserting the resulting video file exists on disk and is non-trivially sized.

Silva's avatar was created by uploading a real photo via `POST /api/v1/avatars/upload` — `face_detected: true` was confirmed in the response, so the source image itself is not the problem.

Every bug below was found by reading **actual logs** (`app/services/animator.py`'s `worker_stderr.log`, written to `backend/models/MuseTalk/worker_stderr.log`, plus the live backend stdout log) — not guessed. Multiple bugs were layered on top of each other, so each fix revealed the *next* failure underneath it.

---

## Bug #1 — PyTorch 2.6 `torch.load` default change breaks ResNet18 checkpoint loading

**Symptom:** MuseTalk worker subprocess crashed on startup every time, before ever reaching "READY". Traceback ended in:

```
File "...\musetalk\utils\face_parsing\resnet.py", line 83, in init_weight
    state_dict = torch.load(model_path) #modelzoo.load_url(resnet18_url)
File "...\torch\serialization.py", line 1487, in load
    return _legacy_load(...)
RuntimeError: Cannot use ``weights_only=True`` with files saved in the legacy .tar format.
In PyTorch 2.6, we changed the default value of the `weights_only` argument
in `torch.load` from `False` to `True`.
```

**Root cause:** PyTorch 2.6 (required by this project's `chatterbox-tts` dependency — see the earlier dependency-conflict resolution in this same session) flipped `torch.load`'s default `weights_only` argument from `False` to `True`. The ResNet18 backbone checkpoint (`resnet18-5c106cde.pth`, downloaded from `download.pytorch.org` during `scripts/setup_musetalk.sh`) uses the old legacy `.tar` serialization format, which the new strict-by-default loader refuses to parse.

**Fix applied** — `backend/models/MuseTalk/musetalk/utils/face_parsing/resnet.py` line 83:
```python
# before
state_dict = torch.load(model_path) #modelzoo.load_url(resnet18_url)
# after
state_dict = torch.load(model_path, weights_only=False) #modelzoo.load_url(resnet18_url)
```
`weights_only=False` is safe here because the checkpoint is a well-known, official file from a trusted source (PyTorch's own model zoo), not user-supplied.

Also patched into `scripts/setup_musetalk.sh` (new step 3c) as an idempotent `sed` so a fresh clone/setup gets the fix automatically.

**Verification:** Confirmed fixed — after this change, the worker no longer crashes at this step; `worker_stderr.log` shows `INFO: float16 enabled` with no further traceback, repeatedly, across many subsequent runs.

---

## Bug #2 — Stray `print()` in `load_all_model()` corrupts the worker's stdout protocol

**Symptom:** After Bug #1 was fixed, the worker still failed to be recognized as ready. Backend log:
```
ERROR - Animation failed (musetalk): Worker failed to start. stderr (tail): ...
```
even though the freshly-captured `worker_stderr.log` showed **no traceback at all** — just `INFO: float16 enabled`.

**Root cause:** `musetalk_worker.py`'s protocol (documented in its own docstring) reserves **stdout** exclusively for a `"READY\n"` handshake line, followed by one JSON line per job result. The parent process (`app/services/animator.py`, `_ensure_worker()`) does:
```python
ready_line = await asyncio.wait_for(proc.stdout.readline(), timeout=model_load_timeout)
if not ready_line.decode().strip().startswith("READY"):
    proc.kill()
    raise RuntimeError(f"Worker failed to start. stderr (tail):\n{stderr_out}")
```
`backend/models/MuseTalk/musetalk/utils/utils.py`'s `load_all_model()` (called during worker startup, before "READY" is printed) had:
```python
print(f"load unet model from {unet_model_path}")
```
`print()` defaults to **stdout**. This line got read by the parent as the READY handshake line instead of the real "READY" — so the parent always concluded the worker "failed to start" even when it hadn't.

**Fix applied** — `backend/models/MuseTalk/musetalk/utils/utils.py`:
```python
# before
print(f"load unet model from {unet_model_path}")
# after
sys.stderr.write(f"load unet model from {unet_model_path}\n")
```
(plus `import sys` added to the top of the file). Also patched into `scripts/setup_musetalk.sh` (new step 3d, idempotent `sed`) for fresh installs.

**Verification:** Confirmed fixed — after this change, `worker_stderr.log` shows `load unet model from ...` correctly landing in stderr, and the worker successfully reaches `"MuseTalk worker ready — models loaded"` in the backend log.

---

## Bug #3 — Stray `print()` calls in OUR OWN `preprocessing.py` rewrite

**Symptom:** After Bug #2's fix, the worker reached READY, accepted a real job, and started real inference — but then failed with inconsistent errors across different job attempts:
```
Animation failed (musetalk): Expecting value: line 1 column 1 (char 0). Falling back to simple.
Animation failed (musetalk): cannot unpack non-iterable NoneType object. Falling back to simple.
```
Different error each time — classic symptom of a **desynced line-oriented protocol** (once one stray line shifts the read offset, every subsequent read is misaligned).

**Root cause:** `backend/models/MuseTalk/musetalk/utils/preprocessing.py` is **our own custom file**, written earlier in this same session (via a heredoc in `scripts/setup_musetalk.sh`) to replace the upstream mmpose/mmcv-based face detection with a `face_alignment`-based CPU/Python-3.12-compatible version. It contained two more stdout `print()` calls, both in the **per-job path** (called on every single inference, not just at startup):
```python
def read_imgs(img_list):
    frames = []
    print("reading images...")          # <- stdout, called every job
    for img_path in tqdm(img_list):
        ...

def get_landmark_and_bbox(img_list, upperbondrange=0):
    ...
    print(f"Getting face bounding boxes...")   # <- stdout, called every job
```
Every single job run polluted stdout again, corrupting the per-job JSON reply protocol the same way Bug #2 corrupted the READY handshake.

**Fix applied** — `backend/models/MuseTalk/musetalk/utils/preprocessing.py` (and the matching heredoc in `scripts/setup_musetalk.sh`):
- Added `import sys`
- `print("reading images...")` → `print("reading images...", file=sys.stderr)`
- `print(f"Getting face bounding boxes...")` → same pattern, `file=sys.stderr`
- All `tqdm(...)` calls given explicit `file=sys.stderr` (tqdm defaults to stderr already, but made explicit for safety)

**Verification:** Confirmed fixed at the protocol level — the worker now reliably reaches actual UNet/VAE inference (visible via tqdm progress bars in `worker_stderr.log` for VAE encode, audio feature extraction, etc.) instead of erroring immediately after the first job. **Whether inference actually *completes* successfully was not yet confirmed** — see Bug #4 and #5 below, and the Final Status section.

---

## Bug #4 — Inference timeout (60s) too short for a cold worker's first-ever GPU inference

**Symptom:** After Bug #3's fix, the worker reliably loads and starts real inference, but repeatedly failed with:
```
ERROR - Animation failed (musetalk): MuseTalk inference timed out after 60s. Falling back to simple.
```
Crucially, `app/services/animator.py`'s timeout handler **kills the worker process** on any inference timeout and resets `self._worker_proc = None`, forcing a full respawn (re-pay full model-load cost) on the next attempt:
```python
except asyncio.TimeoutError:
    proc.kill()
    self._worker_proc = None
    raise RuntimeError(f"MuseTalk inference timed out after {infer_timeout}s")
```
This created a loop where the worker could never survive long enough to become "warm" — every attempt paid full cold-start cost (CUDA context creation, cuDNN algorithm search, kernel JIT compilation) and then got killed at the 60s mark before finishing, repeating forever.

**Root cause:** `app/services/animator.py` line ~224 hardcoded:
```python
infer_timeout = 60 if self.device == "cuda" else 300
```
This 60s budget only accounts for *steady-state* per-sentence inference (per the code's own comment: "GPU: expect ~5-15s per sentence"), not the one-time warmup cost of a **freshly spawned process's first-ever** CUDA/cuDNN call. The code had already made an analogous allowance for *model loading* (bumped from 120s→300s, with a comment citing measured cold-start cost on this exact machine) but never applied the same reasoning to the inference call itself.

**Fix applied** — `backend/app/services/animator.py`:
```python
# before
infer_timeout = 60 if self.device == "cuda" else 300
# after
infer_timeout = 180 if self.device == "cuda" else 300
```
(with an expanded comment explaining the cold-start reasoning and the self-defeating kill-on-timeout interaction).

**Verification: NOT YET CONFIRMED.** The extended timeout was in place for the final test run, which was manually stopped by the user before any single inference attempt completed or timed out under the new 180s budget. It's possible 180s is enough; it's also possible the real bottleneck is something else entirely (e.g. Bug #5 below causing a hang, or genuinely slow hardware/thermal throttling). **This needs a full uninterrupted run to confirm.**

---

## Bug #5 — `ffmpeg` not installed / not on PATH (UNRESOLVED — mixed evidence)

**Symptom:** `worker_stderr.log` repeatedly shows:
```
'ffmpeg' is not recognized as an internal or external command,
operable program or batch file.
```
This is presumably from MuseTalk's internal video-muxing step (combining animated frames + audio into the final `.mp4`), and/or from the "simple" fallback path in `app/services/animator.py`, which separately failed with:
```
Chunk 0 failed [...]: [WinError 2] The system cannot find the file specified
```
(`WinError 2` = the classic Windows "file not found" error for a subprocess call — consistent with `ffmpeg` not being found).

**Root cause diagnosis:** Confirmed via direct check that no system `ffmpeg` is installed or on `PATH` on this machine (`which ffmpeg` → not found). However, `imageio-ffmpeg` (already a project dependency) bundles a real, working ffmpeg binary at:
```
backend/venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win64-v4.2.2.exe
```

**Fix applied** — `backend/main.py`, added near the top (before other imports that might spawn subprocesses):
```python
import imageio_ffmpeg
_ffmpeg_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
if _ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
```
Reasoning: `app/services/animator.py` builds the MuseTalk worker subprocess's environment via `os.environ.copy()`, so fixing `PATH` once in the main process at startup should propagate to the worker subprocess automatically, without needing to patch every ffmpeg-invoking call site individually.

**Verification: INCONCLUSIVE / LIKELY STILL BROKEN.** After this fix was applied and the backend fully restarted, the **same** `'ffmpeg' is not recognized...` error was still observed in `worker_stderr.log` on at least one subsequent attempt — interleaved with *other* attempts in the same log that did *not* show the error. This inconsistency was not root-caused before testing was stopped. Open hypotheses for whoever picks this up:
- The `_worker_env` dict in `AvatarAnimator` may be built/cached once (comment: "Build env once") at a point in the animator's lazy `initialize()` call — need to confirm this happens *after* `main.py`'s PATH fix has actually executed, not before.
- There could be a stale/orphaned worker subprocess from before the fix that didn't get killed on backend restart, and something is still routing jobs to it. Worth checking `tasklist`/Task Manager for lingering `python.exe` processes tied to `musetalk_worker.py` after a "clean" restart.
- The specific ffmpeg-calling code path (upstream MuseTalk code vs. `app/services/animator.py`'s own "simple" fallback) was not identified precisely — `grep -rn "ffmpeg" backend/models/MuseTalk` and `grep -rn "ffmpeg" backend/app/services/animator.py` would narrow this down quickly.

---

## Final status when testing was stopped

The user issued a stop command while the test was mid-run, on this state:
- Session `f81336b1-f9b3-43e8-a61e-5e836e219933`
- Backend log's last relevant lines:
  ```
  2026-08-11 19:44:56 - app.services.animator - INFO - Animating [musetalk] image=...\avatars\1f68e7bf-...\image.jpg audio=...\8c1a010624df_audio.wav
  2026-08-11 19:44:56 - app.services.animator - INFO - Starting persistent MuseTalk worker (loading models once)…
  2026-08-11 19:45:05 - app.services.animator - INFO - MuseTalk worker ready — models loaded
  2026-08-11 19:45:43 - main - INFO - Client disconnected from session f81336b1-... (test script stopped by user)
  ```
- The worker had successfully loaded and was presumably in the middle of actual inference (within the new 180s budget) when the connection was closed — **no confirmed success or failure signal was ever received for a complete run.**

**No video file was ever confirmed to be produced and played back successfully in this debugging session.** The pipeline has demonstrably gotten much further than at the start (5 real, verified-root-caused bugs fixed along the way), but end-to-end success is still unconfirmed.

## Recommended next steps for whoever continues this

1. Restart the backend cleanly (`cd backend && ./venv/Scripts/python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000`), confirm no orphaned `musetalk_worker.py` processes are still running from before.
2. Re-run `backend/_test_animation.py` uninterrupted (it already exists on disk) and let it run to completion — it will either report `ALL CHECKS PASSED` with a real video file size, or a clear assertion failure.
3. If it still fails on the ffmpeg error, run `Get-Process | Where-Object {$_.Path -like "*musetalk_worker*"}` (PowerShell) to check for stale workers, and trace exactly which code path invokes `ffmpeg` by grepping `backend/models/MuseTalk` and `backend/app/services/animator.py`.
4. If it times out again even at 180s, capture a fresh, complete `worker_stderr.log` for a single isolated run and inspect exactly how far inference gets (which tqdm bar is the last one to update) to see if it's truly hung vs. just slow.
5. Once a `video_chunk` event is confirmed with a real non-trivial video file, additionally verify (per the original request) that **only the actively-speaking participant is animated** — this is already true by construction in the round-robin orchestration loop (`app/websocket.py` `_run_round` awaits each participant's full turn sequentially before starting the next), but should be visually confirmed once video actually renders.

## Files changed in this session (animation-pipeline-specific)

- `backend/models/MuseTalk/musetalk/utils/face_parsing/resnet.py` (Bug #1)
- `backend/models/MuseTalk/musetalk/utils/utils.py` (Bug #2)
- `backend/models/MuseTalk/musetalk/utils/preprocessing.py` (Bug #3)
- `scripts/setup_musetalk.sh` (Bugs #1, #2, #3 — patched for fresh installs)
- `backend/app/services/animator.py` (Bug #4)
- `backend/main.py` (Bug #5, unresolved)
- `backend/_test_animation.py` (test harness, still present on disk)
- `.env` — added `KINDROID_SILVA_AVATAR_ID=00000000-0000-0000-0000-000000000000` (not a bug fix, just wiring Silva's uploaded avatar to her Kindroid participant config)

None of these changes have been committed to git yet.
