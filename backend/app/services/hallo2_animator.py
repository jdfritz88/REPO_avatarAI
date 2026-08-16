"""
Hallo2 idle-loop generation service.

Hallo2 is a genuinely different, full-face-generating model — unlike
MuseTalk (backend/app/services/animator.py), which only ever repaints the
mouth region and structurally cannot blink (confirmed against MuseTalk's
own masking code and docs), Hallo2 drives the whole face and produces real
blinking. Verified tonight in an isolated experiment
(backend/model_experiments/hallo2/) against three candidate models
(SadTalker, EchoMimic, Hallo2) — Hallo2 won on community/academic
quality-vs-complaint ratio and license.

It runs in its OWN venv (backend/model_experiments/hallo2/venv/) because
its dependency stack (specific onnxruntime/xformers/transformers/diffusers
versions) genuinely conflicts with the main backend venv — this service
therefore spawns the Hallo2 venv's own python.exe as a subprocess, exactly
like AvatarAnimator does for the MuseTalk worker, just pointed at a
different interpreter.

A single Hallo2 render is far too slow to be a live per-message response —
even at reduced (20) inference steps, a single ~5s clip measured at
270-330s in this session's own testing. So this service does NOT plug into
the live chat pipeline at all; it only generates a fixed IDLE PLAYLIST per
avatar (see generate_idle_playlist): up to 6 short segments, generated
once (like Kindroid's own "render once, reuse forever" idle-animation
model), each independently trimmed to loop back near the avatar's own
resting pose — so any segment can follow any other in random order without
a visible jump-cut (see find_loop_point in the worker script for why this
works: every segment starts from the same static source photo).
"""
import asyncio
import json
import logging
import sys
import tempfile
import uuid
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import settings
from app.services.gpu_lock import gpu_lock

TMPDIR = Path(tempfile.gettempdir())
logger = logging.getLogger(__name__)

# Everything lives under backend/model_experiments/hallo2/ tonight — this
# mirrors how MuseTalk's path is configurable via settings, but Hallo2 was
# built as an experiment first, not through the setup-script pipeline, so
# it's a fixed relative path for now rather than a settings.HALLO2_PATH.
_HALLO2_DIR = Path(__file__).resolve().parent.parent.parent / "model_experiments" / "hallo2"
_HALLO2_VENV_PYTHON = _HALLO2_DIR / "venv" / "Scripts" / "python.exe"
_HALLO2_WORKER_SCRIPT = _HALLO2_DIR / "scripts" / "hallo2_worker.py"
_HALLO2_CONFIG = "configs/inference/long_fast.yaml"  # 20 inference steps — see long_fast.yaml

# Empirically measured tonight: ~270-330s per ~5s segment at 20 steps on
# this machine (RTX 4080 12GB), fairly consistent across repeated calls
# once the worker is warm. Generous ceiling, not a tight budget — a job
# that's genuinely stuck should still be caught well before the user's
# 5-minute-per-render rule would be violated by a NEW job starting late.
_JOB_TIMEOUT_S = 480
_MODEL_LOAD_TIMEOUT_S = 120

MAX_PLAYLIST_SEGMENTS = 6


class Hallo2AnimatorError(RuntimeError):
    pass


class Hallo2Animator:
    """Persistent-worker wrapper around the isolated Hallo2 venv, for idle-playlist generation only."""

    def __init__(self):
        self._worker_proc: Optional[asyncio.subprocess.Process] = None
        self._worker_lock = asyncio.Lock()
        self._stderr_path = _HALLO2_DIR / "worker_stderr.log"
        self._available = _HALLO2_VENV_PYTHON.exists() and _HALLO2_WORKER_SCRIPT.exists()
        if not self._available:
            logger.warning(
                f"Hallo2 not available (missing {_HALLO2_VENV_PYTHON} or {_HALLO2_WORKER_SCRIPT}) "
                "— idle-playlist generation will fail until it's set up."
            )

    async def _ensure_worker(self) -> asyncio.subprocess.Process:
        if self._worker_proc is not None and self._worker_proc.returncode is None:
            return self._worker_proc

        if not self._available:
            raise Hallo2AnimatorError("Hallo2 is not installed/configured (see model_experiments/hallo2/)")

        logger.info("Starting persistent Hallo2 worker (loading models once)…")
        # Same deadlock-avoidance as the MuseTalk worker: stderr -> a FILE,
        # never a pipe nobody drains. Hallo2's vendored code (and this
        # worker's own sys.stdout = sys.stderr redirect, needed because
        # that vendored code prints stray diagnostic lines that would
        # otherwise corrupt the JSON-line protocol on real stdout) makes
        # this an even higher-volume stream than MuseTalk's.
        stderr_file = open(self._stderr_path, "ab")
        proc = await asyncio.create_subprocess_exec(
            str(_HALLO2_VENV_PYTHON),
            str(_HALLO2_WORKER_SCRIPT),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr_file,
            cwd=str(_HALLO2_DIR),
        )
        stderr_file.close()

        init_msg = json.dumps({"config": _HALLO2_CONFIG}) + "\n"
        proc.stdin.write(init_msg.encode())
        await proc.stdin.drain()

        try:
            ready_line = await asyncio.wait_for(proc.stdout.readline(), timeout=_MODEL_LOAD_TIMEOUT_S)
        except asyncio.TimeoutError:
            proc.kill()
            raise Hallo2AnimatorError(f"Hallo2 worker timed out loading models after {_MODEL_LOAD_TIMEOUT_S}s")

        if not ready_line.decode().strip().startswith("READY"):
            proc.kill()
            tail = self._stderr_path.read_text(errors="replace")[-4000:] if self._stderr_path.exists() else ""
            raise Hallo2AnimatorError(f"Hallo2 worker failed to start. stderr (tail):\n{tail}")

        logger.info("Hallo2 worker ready — models loaded")
        self._worker_proc = proc
        return proc

    async def _run_job(self, source_image: str, driving_audio: str, output_path: str, seed: int) -> str:
        async with self._worker_lock:
            proc = await self._ensure_worker()

            job = json.dumps({
                "source_image": source_image,
                "driving_audio": driving_audio,
                "output": output_path,
                "seed": seed,
            }) + "\n"

            # gpu_lock (not just _worker_lock above) is what actually keeps
            # this off the GPU at the same time as a live MuseTalk chat
            # render — see app/services/gpu_lock.py for why. Held only
            # around the write+wait below, not the worker spawn/health
            # check above, so a live chat request queued behind this one
            # segment waits at most one segment's remaining time, not this
            # whole call's setup overhead too.
            async with gpu_lock:
                try:
                    proc.stdin.write(job.encode())
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError, OSError) as e:
                    proc.kill()
                    self._worker_proc = None
                    raise Hallo2AnimatorError(f"Hallo2 worker pipe is dead: {e}") from e

                try:
                    result_line = await asyncio.wait_for(proc.stdout.readline(), timeout=_JOB_TIMEOUT_S)
                except asyncio.TimeoutError:
                    proc.kill()
                    self._worker_proc = None
                    raise Hallo2AnimatorError(f"Hallo2 render timed out after {_JOB_TIMEOUT_S}s")

                if not result_line:
                    proc.kill()
                    self._worker_proc = None
                    raise Hallo2AnimatorError("Hallo2 worker exited before returning a result")

            result = json.loads(result_line.decode().strip())
            if result["status"] != "ok":
                raise Hallo2AnimatorError(result.get("msg", "Unknown Hallo2 worker error"))

            return result["output"]

    @staticmethod
    def _generate_near_silent_wav(path: Path, duration_s: int = 5, sample_rate: int = 16000, seed: int = 0) -> None:
        """
        Hallo2's audio_separator step hard-rejects TRUE digital silence
        (`if not np.any(mix): raise ValueError(...)` — confirmed by reading
        audio_separator/separator/common_separator.py directly tonight).
        Very low-amplitude noise is perceptually silent but passes that
        check, so idle segments are driven by this instead of pure zeros.

        A flat noise floor for the WHOLE clip drives Hallo2's audio-reactive
        pose module continuously, producing nonstop small head jitter for
        5 straight seconds — confirmed live in the browser and flagged by
        the user as unnatural ("a person waiting... might move their head
        to adjust but not constantly non stop... may once per segment").
        Real waiting behavior is mostly still with one small adjustment.

        First attempt kept a low-amplitude (±20) noise floor under a single
        energy pulse, reasoning the pulse would stand out against a quiet
        base — measured wrong: frame-to-frame pixel diff across the
        rendered test clip stayed in the same 3.7-11.8 range on EVERY
        frame transition, no quiet-then-spike pattern, because that ±20
        floor was itself still enough to drive continuous motion for the
        full 5s (never reduced, only added to). Fixed by making the
        resting samples LITERAL zero — `np.any(mix)` in audio_separator
        only needs ONE nonzero sample anywhere in the whole array, not a
        nonzero floor throughout, so true silence everywhere except the
        pulse window satisfies that check while giving the model genuinely
        nothing to react to outside the pulse.
        """
        rng = np.random.default_rng(seed if seed else None)
        n = sample_rate * duration_s

        pulse_len = int(sample_rate * 0.6)
        earliest = int(sample_rate * 1.0)
        latest = n - pulse_len - int(sample_rate * 1.0)
        pulse_start = rng.integers(earliest, max(earliest + 1, latest))
        # Raised-cosine envelope so the pulse fades in/out smoothly rather
        # than clicking in — an abrupt step reads as a jump-cut, not a
        # natural adjustment.
        envelope = (1 - np.cos(np.linspace(0, 2 * np.pi, pulse_len))) / 2
        pulse = rng.standard_normal(pulse_len) * 900 * envelope

        samples = np.zeros(n)
        samples[pulse_start:pulse_start + pulse_len] = pulse
        samples = samples.astype(np.int16)

        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(samples.tobytes())

    async def generate_one_idle_segment(self, source_image_path: str, output_path: str, seed: int = 42) -> str:
        """
        Render a single idle segment from an arbitrary source image — e.g.
        a LivePortrait expression-edited still (see
        app/services/expression_editor.py) rather than the avatar's default
        photo. Same near-silent+pulse driving audio and loop-trimming as
        generate_idle_playlist, just for one on-demand render instead of a
        batch, for the "pick an expression, render it as a new idle
        segment" UI flow.
        """
        silent_wav = TMPDIR / f"hallo2_idle_silence_{uuid.uuid4().hex[:12]}.wav"
        try:
            self._generate_near_silent_wav(silent_wav, seed=1000 + seed)
            return await self._run_job(source_image_path, str(silent_wav), output_path, seed)
        finally:
            for attempt in range(5):
                try:
                    silent_wav.unlink(missing_ok=True)
                    break
                except OSError:
                    if attempt == 4:
                        logger.warning(f"Could not delete temp file {silent_wav} (left for OS cleanup)")
                    else:
                        await asyncio.sleep(0.5)

    async def generate_idle_playlist(
        self, avatar_image_path: str, output_dir: str, num_segments: int = MAX_PLAYLIST_SEGMENTS,
        seed_offset: int = 0,
    ) -> list[str]:
        """
        Generate up to `num_segments` independent idle clips for one avatar.
        Each uses a different random seed (so each looks like a distinct
        idle moment — different blink timing/head sway), and each is
        independently loop-trimmed by the worker back toward the avatar's
        resting pose, so any segment can follow any other in random
        playback order without a visible jump-cut.

        `seed_offset` shifts every seed used here — pass the count of
        segments a caller already has (see idle_playlist.py's
        ensure_idle_playlist) when TOPPING UP an existing playlist, so the
        newly generated segments don't reuse the same seeds — and
        therefore the same near-duplicate motion — as the ones already
        kept.

        Returns the list of successfully generated local file paths. A
        failure partway through returns whatever succeeded so far rather
        than discarding earlier work — a 3-segment playlist beats none.
        """
        num_segments = min(num_segments, MAX_PLAYLIST_SEGMENTS)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results: list[str] = []
        for i in range(num_segments):
            silent_wav = TMPDIR / f"hallo2_idle_silence_{uuid.uuid4().hex[:12]}.wav"
            try:
                self._generate_near_silent_wav(silent_wav, seed=1000 + seed_offset + i)
                output_path = str(out_dir / f"segment_{i}.mp4")
                seed = 42 + seed_offset + i  # deterministic but distinct per segment
                result = await self._run_job(avatar_image_path, str(silent_wav), output_path, seed)
                results.append(result)
                logger.info(f"Hallo2 idle segment {i + 1}/{num_segments} done: {result}")
            except Hallo2AnimatorError as e:
                logger.error(f"Hallo2 idle segment {i + 1}/{num_segments} failed: {e}")
                break
            finally:
                # Windows-specific race: the worker subprocess (or a library
                # it calls into, e.g. librosa/audio_separator) can hold this
                # file open for a moment after the job's JSON response comes
                # back, since file-handle release isn't guaranteed to happen
                # synchronously with "done". unlink() on Windows raises
                # WinError 32 ("used by another process") instead of just
                # silently succeeding the way POSIX unlink would — and
                # letting that propagate out of a `finally` here would
                # discard an already-successful segment's result. A brief
                # retry clears it in practice; if it still won't release,
                # this is just a leftover temp file, not a real failure —
                # log and move on rather than losing real rendered work.
                for attempt in range(5):
                    try:
                        silent_wav.unlink(missing_ok=True)
                        break
                    except OSError:
                        if attempt == 4:
                            logger.warning(f"Could not delete temp file {silent_wav} (left for OS cleanup)")
                        else:
                            await asyncio.sleep(0.5)

        return results


hallo2_animator = Hallo2Animator()
