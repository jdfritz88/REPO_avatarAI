"""
Ordered, cancellable job queue for the GPU-bound avatar-media pipelines
(expression-photo generation + idle-segment rendering).

Before this: each trigger point (auto-setup on first chat, manual
idle-playlist/reroll endpoints) independently fired off its own
`asyncio.create_task`, and the only thing keeping two of them from
colliding on the GPU was `gpu_lock` (app/services/gpu_lock.py) — that
prevented corruption, but gave no visibility into what was pending and no
way to cancel a job before it started. If you deleted an avatar whose
auto-setup hadn't started yet, its task would still run to completion
against a since-deleted avatar.

This module is the single place all of that goes through now: one
ordered list of jobs, one background worker processing them strictly
one at a time (gpu_lock is still what actually serializes the real GPU
work — this queue adds ordering, visibility, and cancellation on top of
it, not a second lock). A job can only be cancelled while it's still
QUEUED; once it's RUNNING it completes normally — cancelling never
interrupts a render already in progress, cancelling only ever removes
work that hasn't started yet.
"""
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

logger = logging.getLogger(__name__)

JobKind = Literal["full_setup", "topup", "reroll"]
JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]

# Completed/failed/cancelled jobs older than this stay in the snapshot list
# for the UI to see briefly, then get pruned so it doesn't grow forever.
_MAX_FINISHED_KEPT = 20


@dataclass
class RenderJob:
    id: str
    avatar_id: str
    avatar_name: str
    kind: JobKind
    status: JobStatus = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None
    _done_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False, compare=False)


class RenderQueue:
    def __init__(self):
        self._jobs: list[RenderJob] = []
        self._wakeup = asyncio.Event()
        self._worker_task: Optional[asyncio.Task] = None

    def start(self):
        """Idempotent — safe to call from FastAPI's startup event even if it somehow runs twice."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("Render queue worker started")

    def enqueue(self, avatar_id: str, avatar_name: str, kind: JobKind) -> str:
        for j in self._jobs:
            if j.avatar_id == avatar_id and j.status in ("queued", "running"):
                logger.info(
                    f"Avatar {avatar_id} already has a {j.status} {j.kind} job ({j.id}), not re-queuing {kind}"
                )
                return j.id
        job = RenderJob(id=str(uuid.uuid4()), avatar_id=avatar_id, avatar_name=avatar_name, kind=kind)
        self._jobs.append(job)
        self._wakeup.set()
        logger.info(f"Enqueued {kind} job {job.id} for avatar {avatar_id} ({avatar_name})")
        return job.id

    def cancel(self, job_id: str) -> bool:
        """True if a QUEUED (not yet started) job was removed. False if it's already running/done/missing — never interrupts a running job."""
        for j in self._jobs:
            if j.id == job_id and j.status == "queued":
                j.status = "cancelled"
                j._done_event.set()
                self._jobs.remove(j)
                logger.info(f"Cancelled queued job {job_id} for avatar {j.avatar_id}")
                return True
        return False

    def cancel_by_avatar(self, avatar_id: str) -> bool:
        """Same as cancel(), but by avatar — used when an avatar is deleted so any still-queued work for it is dropped."""
        for j in self._jobs:
            if j.avatar_id == avatar_id and j.status == "queued":
                return self.cancel(j.id)
        return False

    def snapshot(self) -> list[dict]:
        return [
            {
                "id": j.id, "avatar_id": j.avatar_id, "avatar_name": j.avatar_name,
                "kind": j.kind, "status": j.status, "created_at": j.created_at, "error": j.error,
            }
            for j in self._jobs
        ]

    async def wait_for(self, job_id: str, timeout: Optional[float] = None) -> Optional[RenderJob]:
        """Block until `job_id` leaves the queue/running state. Used by endpoints that still want to hand back the finished result synchronously, without duplicating the actual render logic."""
        job = next((j for j in self._jobs if j.id == job_id), None)
        if job is None:
            return None
        try:
            await asyncio.wait_for(job._done_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        return job

    async def _worker_loop(self):
        from app.services.idle_playlist import ensure_idle_playlist, generate_full_idle_set

        while True:
            queued = [j for j in self._jobs if j.status == "queued"]
            if not queued:
                self._wakeup.clear()
                await self._wakeup.wait()
                continue

            job = queued[0]
            job.status = "running"
            logger.info(f"Render queue: starting {job.kind} job {job.id} for avatar {job.avatar_id}")
            try:
                if job.kind in ("full_setup", "reroll"):
                    await generate_full_idle_set(job.avatar_id)
                else:
                    await ensure_idle_playlist(job.avatar_id)
                job.status = "done"
                logger.info(f"Render queue: finished job {job.id} for avatar {job.avatar_id}")
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                logger.error(f"Render queue: job {job.id} ({job.kind}) failed for avatar {job.avatar_id}: {e}")
            finally:
                job._done_event.set()
                finished = [j for j in self._jobs if j.status not in ("queued", "running")]
                if len(finished) > _MAX_FINISHED_KEPT:
                    for stale in finished[:-_MAX_FINISHED_KEPT]:
                        self._jobs.remove(stale)


render_queue = RenderQueue()
