"""
Expression-editing service — identity-preserving still-photo expression
edits via LivePortrait (ComfyUI-AdvancedLivePortrait's Expression Editor
node), used to give Hallo2 idle segments visual variety it can't produce
on its own. Confirmed earlier tonight, by reading Hallo2's own source
code: Hallo2 has NO expression/emotion input at all — the resting
expression in every clip it generates comes entirely from whatever's
already in the source photo. This service is how we get MORE than one
resting expression to hand it.

This is genuinely a different shape than the other two GPU services in
this app (animator.py's MuseTalk worker, hallo2_animator.py's Hallo2
worker), which both spawn a persistent Python subprocess and talk to it
over a stdin/stdout JSON-line protocol. LivePortrait/ComfyUI instead runs
as a real HTTP server (ComfyUI's own REST API) — so this service's job is
to own that server subprocess's lifecycle (start it once, health-check it,
restart if it dies) and delegate the actual generation call to
model_experiments/comfyui/expression_editor_client.py, a pure-stdlib HTTP
client built and verified against it earlier tonight. That client has no
heavy dependencies of its own (no torch import), so it's imported directly
into this MAIN backend process rather than needing its own venv subprocess
— only the ComfyUI SERVER needs the isolated comfyui/venv/ (torch,
diffusers, etc.).
"""
import asyncio
import importlib.util
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from app.services.gpu_lock import gpu_lock

logger = logging.getLogger(__name__)

_COMFYUI_DIR = Path(__file__).resolve().parent.parent.parent / "model_experiments" / "comfyui"
_COMFYUI_VENV_PYTHON = _COMFYUI_DIR / "venv" / "Scripts" / "python.exe"
_COMFYUI_MAIN = _COMFYUI_DIR / "main.py"
_CLIENT_MODULE_PATH = _COMFYUI_DIR / "expression_editor_client.py"

_SERVER_URL = "http://127.0.0.1:8188"
_STARTUP_TIMEOUT_S = 90  # ComfyUI + custom-node import, cold start
_GENERATE_TIMEOUT_S = 120  # matches expression_editor_client's own default


def _load_client_module():
    """
    expression_editor_client.py lives under model_experiments/, outside
    this package — imported by file path (like the sitecustomize.py /
    isolated-venv scripts elsewhere in this app) rather than added as a
    package dependency, since it's explicitly a standalone script tonight's
    setup work produced, not a library this repo owns yet.
    """
    spec = importlib.util.spec_from_file_location("expression_editor_client", _CLIENT_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExpressionEditorServiceError(RuntimeError):
    pass


class ExpressionEditorService:
    def __init__(self):
        self._server_proc: Optional[asyncio.subprocess.Process] = None
        self._server_lock = asyncio.Lock()
        self._stderr_path = _COMFYUI_DIR / "server_stderr.log"
        self._available = _COMFYUI_VENV_PYTHON.exists() and _COMFYUI_MAIN.exists() and _CLIENT_MODULE_PATH.exists()
        self._client = None
        if not self._available:
            logger.warning(
                f"LivePortrait/ComfyUI not available (missing {_COMFYUI_VENV_PYTHON}, "
                f"{_COMFYUI_MAIN}, or {_CLIENT_MODULE_PATH}) — expression editing will fail until set up."
            )
        else:
            self._client = _load_client_module()

    @property
    def presets(self) -> dict:
        if not self._client:
            return {}
        return self._client.PRESETS

    def _server_responding(self) -> bool:
        try:
            urllib.request.urlopen(f"{_SERVER_URL}/system_stats", timeout=2)
            return True
        except (urllib.error.URLError, OSError):
            return False

    async def _ensure_server(self) -> None:
        async with self._server_lock:
            if self._server_proc is not None and self._server_proc.returncode is None:
                if await asyncio.to_thread(self._server_responding):
                    return
                # Process alive but not answering — treat as dead, restart below.
                self._server_proc.kill()
                self._server_proc = None

            if not self._available:
                raise ExpressionEditorServiceError(
                    "LivePortrait is not installed/configured (see model_experiments/comfyui/)"
                )

            # A server might already be running from a previous backend
            # process that wasn't cleanly stopped — reuse it rather than
            # spawning a second one on the same port.
            if await asyncio.to_thread(self._server_responding):
                logger.info("Reusing already-running ComfyUI server")
                return

            logger.info("Starting ComfyUI server for LivePortrait expression editing…")
            stderr_file = open(self._stderr_path, "ab")
            proc = await asyncio.create_subprocess_exec(
                str(_COMFYUI_VENV_PYTHON), str(_COMFYUI_MAIN),
                "--port", "8188", "--disable-auto-launch",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=stderr_file,
                cwd=str(_COMFYUI_DIR),
            )
            stderr_file.close()
            self._server_proc = proc

            deadline = asyncio.get_event_loop().time() + _STARTUP_TIMEOUT_S
            while asyncio.get_event_loop().time() < deadline:
                if proc.returncode is not None:
                    tail = self._stderr_path.read_text(errors="replace")[-4000:] if self._stderr_path.exists() else ""
                    raise ExpressionEditorServiceError(f"ComfyUI server exited during startup. stderr (tail):\n{tail}")
                if await asyncio.to_thread(self._server_responding):
                    logger.info("ComfyUI server ready")
                    return
                await asyncio.sleep(1)

            proc.kill()
            self._server_proc = None
            raise ExpressionEditorServiceError(f"ComfyUI server did not respond within {_STARTUP_TIMEOUT_S}s")

    async def generate(
        self, source_image_path: str, output_path: str, *,
        preset: Optional[str] = None, params: Optional[dict] = None,
    ) -> str:
        if not self._available:
            raise ExpressionEditorServiceError(
                "LivePortrait is not installed/configured (see model_experiments/comfyui/)"
            )

        await self._ensure_server()

        # gpu_lock: this shares the one GPU with MuseTalk (live chat) and
        # Hallo2 (idle-playlist) — see app/services/gpu_lock.py. A single
        # expression edit is a few seconds of GPU work once warm, so this
        # is a short hold, not the multi-minute one Hallo2 segments take.
        async with gpu_lock:
            try:
                return await asyncio.to_thread(
                    self._client.generate_expression,
                    source_image_path, output_path,
                    preset=preset, params=params,
                    server=_SERVER_URL, timeout_s=_GENERATE_TIMEOUT_S,
                )
            except self._client.ExpressionEditorError as e:
                raise ExpressionEditorServiceError(str(e)) from e


expression_editor_service = ExpressionEditorService()
