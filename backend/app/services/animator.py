import asyncio
import datetime
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import uuid
import wave
from pathlib import Path
from typing import Optional

import torch

from app.config import settings
from app.services.gpu_lock import gpu_lock
from app.services import win_process_tree as _win_job

TMPDIR = Path(tempfile.gettempdir())

# Centralized log folder at the repo root (backend/app/services/animator.py ->
# services -> app -> backend -> REPO_avatarAI). All per-launch worker logs go
# here, each with a unique timestamped name, so nothing ever gets overwritten
# and every run's evidence is preserved permanently in one place.
LOGS_DIR = Path(__file__).resolve().parents[3] / "Logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


def _kill_worker_tree(proc: "asyncio.subprocess.Process", job: Optional[int] = None) -> None:
    """
    Kill a worker subprocess AND its descendants.

    On this machine the venv's python.exe was created against the Microsoft
    Store Python 3.10 distribution, which is a launcher stub that re-execs
    the real interpreter as a CHILD process (confirmed via `wmic process
    get ParentProcessId`: the worker PID we spawn is the parent of a second
    PID actually running the model). `proc.kill()` (Windows TerminateProcess)
    only reaches the stub — the real interpreter, the one actually holding
    MuseTalk's weights in GPU memory, survives as an orphan.

    This was FIRST fixed with `taskkill /PID <pid> /T /F`, which kills a
    process tree by walking it from the given PID at the moment it's
    called. That fix had a real, confirmed gap: if the stub has already
    exited on its own by the time a LATER timeout fires the kill (a very
    real scenario — a live overnight session left an orphaned worker
    pair that survived multiple subsequent timeout-triggered kill
    attempts), `taskkill` can't resolve the now-dead stub PID at all and
    never reaches the grandchild — confirmed directly with a controlled
    test: `taskkill /PID <already-exited-stub> /T /F` returns
    "ERROR: The process ... not found" and the grandchild survives.

    `job` (a Windows Job Object handle, created and the worker assigned to
    it immediately at spawn time in _ensure_worker — see win_process_tree.py
    for why that ordering is what makes this reliable even after the stub
    exits) is the primary mechanism now: closing/terminating the job kills
    every process ever assigned to it regardless of whether the stub is
    still alive. `taskkill /T` is kept as a secondary attempt too — cheap,
    harmless if redundant, and still useful as a fallback on the rare
    chance job-object creation/assignment failed for this particular
    worker (see win_process_tree.py's own failure logging for that case).
    """
    if os.name != "nt":
        if proc.pid is not None:
            proc.kill()
        return

    if job:
        _win_job.kill_job(job, "musetalk_worker")

    if proc.pid is not None:
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
        )


class AvatarAnimator:
    """
    Avatar Animation Service.
    Supported engines (set AVATAR_ENGINE in .env):
      - musetalk : MuseTalk V1.5 — persistent worker (models loaded once)
      - simple   : ffmpeg static image + audio, no lip-sync
    """

    def __init__(self):
        self.engine = settings.AVATAR_ENGINE
        self.resolution = settings.AVATAR_RESOLUTION
        self.fps = settings.AVATAR_FPS
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_float16 = self.device == "cuda"  # float16 on GPU = ~2× faster via Tensor Cores
        self._initialised = False
        self._musetalk_dir: Optional[Path] = None

        # Persistent worker handles
        self._worker_proc: Optional[asyncio.subprocess.Process] = None
        self._worker_job: Optional[int] = None  # Windows Job Object handle — see win_process_tree.py
        self._worker_lock = asyncio.Lock()
        self._worker_env: dict = {}
        self._worker_stderr_path: Optional[Path] = None

        # Set by _worker_infer whenever the MuseTalk worker's CUDA-graph
        # watchdog reports something the end user should actually be told
        # about (a rendering glitch it recovered from, or a permanent
        # fallback to slower rendering) — websocket.py checks and clears
        # this after each animate() call to turn it into a real WS event
        # the frontend can show, rather than it just sitting in a log file.
        self.pending_watchdog_alert: Optional[dict] = None
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(
                f"AvatarAnimator: engine={self.engine}, device=cuda "
                f"({gpu_name}, {vram_gb:.1f} GB VRAM), float16={self.use_float16}"
            )
        else:
            logger.info(
                f"AvatarAnimator: engine={self.engine}, device=cpu "
                f"(no GPU — consider AWS g5/g6 instance for real-time performance)"
            )

    # ── initialisation ────────────────────────────────────────────────────────

    async def initialize(self):
        if self._initialised:
            return

        if self.engine == "musetalk":
            self._musetalk_dir = self._find_dir(settings.MUSETALK_PATH, "scripts/inference.py")
            if self._musetalk_dir is None:
                logger.warning(
                    "MuseTalk not found at '%s'. "
                    "Run scripts/setup_musetalk.sh to install it. "
                    "Falling back to simple animation.",
                    settings.MUSETALK_PATH,
                )
                self.engine = "simple"
            else:
                logger.info(f"MuseTalk found at: {self._musetalk_dir}")
                # Build env once
                existing = os.environ.get("PYTHONPATH", "")
                self._worker_env = os.environ.copy()
                self._worker_env["PYTHONPATH"] = str(self._musetalk_dir) + (
                    os.pathsep + existing if existing else ""
                )
                # DIAGNOSTIC (temporary — Bug #5 investigation): confirm
                # whether main.py's ffmpeg PATH fix has actually landed in
                # os.environ by the time this env snapshot is taken.
                logger.info(
                    f"[DIAG] worker env PATH (first 500 chars): "
                    f"{self._worker_env.get('PATH', '<MISSING>')[:500]}"
                )
                logger.info(
                    f"[DIAG] ffmpeg dir present in worker PATH: "
                    f"{'imageio_ffmpeg' in self._worker_env.get('PATH', '')}"
                )

        elif self.engine not in ("simple",):
            logger.warning(f"Unknown engine '{self.engine}', using simple animation.")
            self.engine = "simple"

        self._initialised = True

    def _resolve_worker_script(self, musetalk_dir: Path) -> Path:
        """
        Locate the persistent-worker script.

        `musetalk_worker.py` is OUR custom driver, not part of the upstream
        MuseTalk repo that setup_musetalk.sh clones — so a fresh clone won't
        have it under models/MuseTalk/scripts/. We ship a tracked copy at
        backend/musetalk_worker.py and prefer whichever exists, so MuseTalk
        works even if setup hasn't copied the file into the clone yet.
        The process still runs with cwd=musetalk_dir + PYTHONPATH set to the
        clone, so its `from musetalk.utils …` imports resolve regardless of
        where the script file physically lives.
        """
        in_clone = musetalk_dir / "scripts" / "musetalk_worker.py"
        if in_clone.exists():
            return in_clone
        # backend/app/services/animator.py → backend/musetalk_worker.py
        tracked = Path(__file__).resolve().parent.parent.parent / "musetalk_worker.py"
        if tracked.exists():
            logger.info(f"Using tracked MuseTalk worker at {tracked}")
            return tracked
        raise FileNotFoundError(
            f"musetalk_worker.py not found in {in_clone} or {tracked}. "
            "Re-run scripts/setup_musetalk.sh."
        )

    def _find_dir(self, config_path: str, marker_file: str) -> Optional[Path]:
        candidates = [
            Path(config_path),
            Path(__file__).resolve().parent.parent.parent / config_path,
        ]
        for p in candidates:
            if (p / marker_file).exists():
                return p.resolve()
        return None

    # ── persistent worker management ─────────────────────────────────────────

    async def _ensure_worker(self) -> asyncio.subprocess.Process:
        """Start the persistent worker if not already running."""
        if self._worker_proc is not None and self._worker_proc.returncode is None:
            return self._worker_proc

        musetalk_dir: Path = self._musetalk_dir  # type: ignore[assignment]
        worker_script = self._resolve_worker_script(musetalk_dir)

        logger.info("Starting persistent MuseTalk worker (loading models once)…")
        # stderr -> a file, NOT a pipe. The worker's model loading (tqdm progress,
        # HF "Loading weights" bars, library warnings) writes a lot of stderr
        # output; with stderr=PIPE and nobody ever draining it, the OS pipe
        # buffer (~64KB on Windows) fills up and the worker process BLOCKS on
        # its next stderr write — indefinitely, since nothing reads it. That
        # looked like "loading is just slow" but was actually a silent deadlock
        # that always ended in a timeout. File writes never block on a reader.
        # One log file PER WORKER LAUNCH, timestamped, never reused --
        # not a rotation scheme layered on top of a single growing file.
        # Real incident tonight: manually swapping this file's contents
        # aside between diagnostic runs meant a specific run's evidence
        # (in that case, tied to a "bonsai" test) was only one accidental
        # overwrite away from being gone for good, and ephemeral scratch
        # storage isn't a safe place to rely on for anything that matters
        # later. A fresh, permanently-dated file per launch means no
        # future run can ever silently erase a past one's evidence.
        launch_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._worker_stderr_path = LOGS_DIR / f"worker_stderr_{launch_stamp}.log"
        stderr_file = open(self._worker_stderr_path, "ab")
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(worker_script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr_file,
            cwd=str(musetalk_dir),
            env=self._worker_env,
        )
        stderr_file.close()  # child has its own fd now; safe to close our handle

        # Assign to a kill-on-close Job Object AS THE VERY NEXT STEP after
        # spawn — measured directly against this real worker script: the
        # stub/grandchild split (the venv python.exe launcher handing off to
        # the real interpreter) takes ~0.3s to appear, while this assignment
        # itself completes in ~0.02s, a comfortable real margin (not
        # assumed — timed with a controlled test before relying on it). Any
        # further child the stub spawns after this point inherits job
        # membership automatically, so the real interpreter gets captured
        # even though it doesn't exist yet at the moment of assignment.
        self._worker_job = _win_job.create_kill_on_close_job("musetalk_worker")
        if self._worker_job:
            _win_job.assign_pid_to_job(self._worker_job, proc.pid, "musetalk_worker")

        # Send init config — include float16 flag so worker can optimise for GPU
        init_msg = (
            json.dumps(
                {
                    "unet_model_path": str(musetalk_dir / "models" / "musetalkV15" / "unet.pth"),
                    "unet_config": str(musetalk_dir / "models" / "musetalkV15" / "musetalk.json"),
                    "whisper_dir": str(musetalk_dir / "models" / "whisper"),
                    "vae_type": str(musetalk_dir / "models" / "sd-vae"),
                    "use_float16": self.use_float16,
                }
            )
            + "\n"
        )
        proc.stdin.write(init_msg.encode())
        await proc.stdin.drain()

        # Fail fast at 30s instead of hanging up to 5-10 minutes: when the GPU
        # is contended by another process (e.g. a separate TTS server sharing
        # this machine's single card), model loading stalls indefinitely
        # rather than genuinely taking longer — better to give up quickly and
        # fall back to simple animation than block the whole reply on it.
        model_load_timeout = 30
        logger.info(f"Waiting for worker to finish loading models (timeout={model_load_timeout}s)…")
        try:
            ready_line = await asyncio.wait_for(proc.stdout.readline(), timeout=model_load_timeout)
        except asyncio.TimeoutError:
            _kill_worker_tree(proc, self._worker_job)
            self._worker_job = None
            raise RuntimeError("MuseTalk worker timed out while loading models")

        if not ready_line.decode().strip().startswith("READY"):
            _kill_worker_tree(proc, self._worker_job)
            self._worker_job = None
            stderr_out = self._worker_stderr_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(
                f"Worker failed to start. stderr (tail):\n{stderr_out}"
            )

        logger.info("MuseTalk worker ready — models loaded")
        self._worker_proc = proc
        return proc

    async def _worker_infer(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        coord_cache: Optional[str],
        infer_timeout: int = 30,
    ) -> str:
        """Send one job to the persistent worker and await its result."""
        async with self._worker_lock:
            proc = await self._ensure_worker()

            job = (
                json.dumps(
                    {
                        "image": str(Path(image_path).resolve()),
                        "audio": str(Path(audio_path).resolve()),
                        "output": str(Path(output_path).resolve()),
                        "coord_cache": coord_cache,
                    }
                )
                + "\n"
            )

            # gpu_lock keeps this off the GPU at the same moment as a
            # background Hallo2 idle-playlist segment — see
            # app/services/gpu_lock.py. Chat is the interactive path, so
            # this acquire can make a reply wait behind an in-progress
            # Hallo2 segment (up to that segment's remaining ~4-5 min) —
            # accepted lag per explicit direction, not a bug. infer_timeout
            # below only bounds the GPU call itself once the lock is held,
            # so a long wait for the lock never gets misread as MuseTalk
            # itself being stuck.
            async with gpu_lock:
                # If the worker died (OOM/segfault) its stdin is closed; writing
                # raises BrokenPipeError. Reset the handle so the NEXT job respawns
                # a fresh worker instead of repeatedly failing against a dead pipe.
                try:
                    proc.stdin.write(job.encode())
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError, OSError) as e:
                    _kill_worker_tree(proc, self._worker_job)
                    self._worker_proc = None
                    self._worker_job = None
                    raise RuntimeError(f"MuseTalk worker pipe is dead: {e}") from e

                # GPU steady-state: ~5-15s per sentence, first inference on a
                # freshly-spawned worker pays extra one-time CUDA/cuDNN warm-up
                # cost on top of that. Previously budgeted up to 300s/600s for
                # this, but under GPU contention from another process (e.g. a
                # separate TTS server sharing this machine's one card) that just
                # means the whole reply hangs for minutes before falling back —
                # worse than failing fast. Flat 30s default: still enough for a
                # normal warm/cold GPU inference on a short chat sentence, but
                # gives up quickly when the GPU is genuinely unavailable instead
                # of blocking the user's reply. Callers rendering much longer,
                # non-time-critical clips (e.g. a 60s idle loop) pass a bigger
                # budget explicitly — see generate_idle_loop.
                try:
                    result_line = await asyncio.wait_for(proc.stdout.readline(), timeout=infer_timeout)
                except asyncio.TimeoutError:
                    _kill_worker_tree(proc, self._worker_job)
                    self._worker_proc = None
                    self._worker_job = None
                    raise RuntimeError(f"MuseTalk inference timed out after {infer_timeout}s")

                # Empty read == worker exited mid-job (EOF on stdout). Reset so the
                # next call respawns instead of erroring on a half-dead process.
                if not result_line:
                    _kill_worker_tree(proc, self._worker_job)
                    self._worker_proc = None
                    self._worker_job = None
                    raise RuntimeError("MuseTalk worker exited before returning a result")

            result = json.loads(result_line.decode().strip())
            if result["status"] != "ok":
                raise RuntimeError(result.get("msg", "Unknown worker error"))

            watchdog_alert = result.get("watchdog_alert")
            if watchdog_alert:
                logger.warning(f"MuseTalk CUDA-graph watchdog: {watchdog_alert.get('message')}")
                self.pending_watchdog_alert = watchdog_alert

            return output_path

    # ── public API ────────────────────────────────────────────────────────────

    async def animate(
        self,
        avatar_image_path: str,
        audio_path: str,
        output_path: str,
        cache_key: Optional[str] = None,
        infer_timeout: Optional[int] = None,
    ) -> str:
        """
        Animate avatar with audio. Returns path to the generated video.
        Falls back to simple (static image + audio) on any engine failure.

        `infer_timeout` bounds MuseTalk inference for THIS call. Left as
        None (the default), it's derived from the audio's own duration
        using the same max(30, duration_s * 15) "generous ceiling, not a
        tight budget" formula generate_idle_loop already uses below — a
        flat 30s regardless of sentence length was confirmed live tonight
        to be too tight for anything past the first (deliberately short)
        reply chunk: chunk 1 of a real Kindroid reply finished in 28s,
        and every longer sentence after it in the same reply timed out at
        the flat 30s cap and fell back to non-lip-synced "simple"
        animation — this was the actual mechanism behind the reported
        "disconnect between chat and animation," not (only) the token-
        pacing issue fixed separately in websocket.py. Callers that know
        their own budget (generate_idle_loop) can still override.
        """
        if not self._initialised:
            await self.initialize()

        if infer_timeout is None:
            infer_timeout = max(30, int(self._audio_duration_s(audio_path) * 15))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Animating [{self.engine}] image={avatar_image_path} audio={audio_path}")

        try:
            if self.engine == "musetalk":
                return await self._animate_musetalk(
                    avatar_image_path, audio_path, output_path, infer_timeout=infer_timeout
                )
            else:
                return await self._animate_simple(avatar_image_path, audio_path, output_path)
        except Exception as e:
            logger.error(f"Animation failed ({self.engine}): {e}. Falling back to simple.")
            return await self._animate_simple(avatar_image_path, audio_path, output_path)

    @staticmethod
    def _audio_duration_s(audio_path: str) -> float:
        """
        Read a WAV's duration directly from its header — no decode needed.

        Deliberately NOT using the stdlib `wave` module: it only understands
        integer PCM (format tag 1) and raises `wave.Error: unknown format`
        on IEEE-float WAVs (format tag 3) — which is exactly what
        `torchaudio.save(path, wav, sr)` writes for Chatterbox's raw float32
        model output (see app/services/tts.py). Confirmed live tonight: with
        `wave.open`, EVERY TTS chunk hit this method's except-branch and
        silently fell back to the flat 2.0s default, which defeated the
        whole point of scaling the MuseTalk timeout to the chunk's real
        length — the very first long sentence tested still timed out at a
        flat 30s despite the scaling code being in place. This instead
        parses the RIFF header directly (format-agnostic: reads byte rate
        and data-chunk size, works the same for PCM or float samples).
        """
        try:
            with open(audio_path, "rb") as f:
                riff = f.read(12)
                if riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
                    raise ValueError("not a RIFF/WAVE file")
                byte_rate = None
                while True:
                    header = f.read(8)
                    if len(header) < 8:
                        break
                    chunk_id = header[:4]
                    chunk_size = int.from_bytes(header[4:8], "little")
                    if chunk_id == b"fmt ":
                        fmt = f.read(chunk_size)
                        byte_rate = int.from_bytes(fmt[8:12], "little")
                    elif chunk_id == b"data":
                        if byte_rate:
                            return chunk_size / float(byte_rate)
                        f.seek(chunk_size, 1)
                    else:
                        f.seek(chunk_size, 1)
            raise ValueError("no data chunk found")
        except Exception:
            # Any read failure (missing file, non-WAV) falls back to the old
            # flat-30s assumption rather than blocking the call on it.
            return 2.0

    async def generate_idle_loop(
        self,
        avatar_image_path: str,
        output_path: str,
        duration_s: int = 60,
    ) -> str:
        """
        Render a silent idle loop for an avatar: a `duration_s`-second clip
        with no speech driving it, meant to be played on <video loop> while
        nothing is being said (mirrors Kindroid's pre-rendered idle clip —
        rendered once per avatar, then just looped client-side, not
        regenerated per idle moment).

        Not time-critical (nothing blocks on it — see the /idle-video
        endpoint), so this uses a much larger inference budget than live
        chat sentences: a 60s clip is roughly 10-20x the frame count of a
        typical spoken sentence, so it needs proportionally longer to render
        even on an uncontended GPU.
        """
        silent_wav = TMPDIR / f"idle_silence_{uuid.uuid4().hex[:12]}.wav"
        try:
            with wave.open(str(silent_wav), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)  # 16-bit
                w.setframerate(24000)
                w.writeframes(b"\x00\x00" * 24000 * duration_s)

            return await self.animate(
                avatar_image_path,
                str(silent_wav),
                output_path,
                infer_timeout=max(900, duration_s * 15),
            )
        finally:
            silent_wav.unlink(missing_ok=True)

    # ── MuseTalk ──────────────────────────────────────────────────────────────

    async def _animate_musetalk(
        self,
        avatar_path: str,
        audio_path: str,
        output_path: str,
        infer_timeout: int = 30,
    ) -> str:
        """Run MuseTalk via persistent worker (models stay loaded between calls)."""
        musetalk_dir: Path = self._musetalk_dir  # type: ignore[assignment]

        # Per-avatar face-coordinate cache (saves face-detection on repeat calls)
        avatar_id = hashlib.md5(str(Path(avatar_path).resolve()).encode()).hexdigest()
        coord_cache = str(musetalk_dir / "results" / "coords" / f"{avatar_id}.pkl")
        os.makedirs(os.path.dirname(coord_cache), exist_ok=True)

        await self._worker_infer(
            avatar_path, audio_path, output_path, coord_cache, infer_timeout=infer_timeout
        )

        logger.info(f"MuseTalk animation done: {output_path}")
        return output_path

    # ── Simple ffmpeg fallback ────────────────────────────────────────────────

    async def _animate_simple(
        self,
        avatar_path: str,
        audio_path: str,
        output_path: str,
    ) -> str:
        """Combine static image + audio with FFmpeg. No lip-sync."""
        logger.info("Using simple animation (static image + audio, no lip-sync)")

        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(avatar_path),
            "-i",
            str(audio_path),
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            "-vf",
            (
                f"fps={self.fps},"
                f"scale={self.resolution}:{self.resolution}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={self.resolution}:{self.resolution}:(ow-iw)/2:(oh-ih)/2"
            ),
            output_path,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode(errors="replace")
            logger.error(f"FFmpeg error:\n{err}")
            raise RuntimeError("Simple animation (ffmpeg) failed")

        logger.info(f"Simple animation done: {output_path}")
        return output_path

    # ── helpers ───────────────────────────────────────────────────────────────

    def generate_cache_key(self, text: str, avatar_id: str) -> str:
        return hashlib.md5(f"{avatar_id}:{text}".encode()).hexdigest()


# Global instance
avatar_animator = AvatarAnimator()
