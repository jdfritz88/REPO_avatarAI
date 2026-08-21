# Session Report — 2026-08-20 — MuseTalk GPU stall investigation

## Bottom line

The recurring MuseTalk slowdown (batches that should take ~0.1-0.3s taking
anywhere from ~2s to 70s+, sometimes never completing before the job-level
timeout) is **not** caused by CUDA graphs, memory fragmentation, allocator
retries, process-local state, GPU priority, or thermal/power throttling.
Every one of those was tested tonight with direct, reproducible evidence and
ruled out. What's left, and matches every measurement, is GPU-level
scheduling/preemption below PyTorch's visibility (Windows WDDM) — most
likely from another process using the GPU, possibly compounded by
Optimus/hybrid-graphics frame-copy overhead and Alienware/NVIDIA Dynamic
Boost power-budget cycling. This needs either a system-level change (HAGS)
or an application-level workaround (below), not more code changes to the
inference path itself.

## Decisive evidence: the live Megumi test (01:20:50–01:28:40)

Real conversation, real logs, GPU telemetry sampled at 0.3s resolution the
whole time.

- Every response hit the slowdown. Two chunks hung completely (zero
  progress after CUDA graph capture, killed by the 30s job timeout,
  fell back to static-image/no-lipsync). One chunk didn't hang but
  crawled: batch 1 took 70.48s, then 10 more batches settled into a
  dead-steady ~7s each (should be ~0.1-0.3s).
- GPU telemetry during **every** stall window — hung or crawling —
  showed the identical signature: **100% GPU utilization, ~0% memory
  utilization, clocks pinned near max (2445MHz, unthrottled), power
  draw ~50W of a 175W budget.** That combination is the fingerprint of
  a stuck wait, not real compute, not memory pressure, not thermal
  throttling.
- The watchdog never tripped despite 11 consecutive ~7s batches,
  because its baseline EMA gets seeded from the *first* post-warmup
  batch — if that one's already bad, everything after it looks
  "normal" by comparison relative to a multiplicative threshold. Real,
  confirmed detection gap, not yet fixed (needs a design that doesn't
  purely self-normalize, e.g. an absolute ceiling that isn't
  overridable by a bad EMA).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (added earlier
  tonight as a fix for a real, separate memory-fragmentation issue) is
  silently a no-op — worker log shows
  `expandable_segments not supported on this platform`. Whatever
  benefit was measured earlier came from the paired
  `torch.cuda.empty_cache()` call, not this setting.

## Direct tests run tonight, with results

| Test | Result |
|---|---|
| Fresh OS process (new PID, new CUDA context) vs. old process | **Identical** steady-state speed. Rules out any restart-based recovery pattern (liveness-probe-and-restart, draining+migrate) for this specific phenomenon — a brand-new worker starts already in whatever state the GPU is in. |
| Windows process priority boost (`HIGH_PRIORITY_CLASS`) + CUDA stream priority boost | Made it **worse** — steady ~5.0s/call vs. ~2.05s/call baseline (2.4x slower). `SetPriorityClass` itself silently failed to apply; the stream-priority change alone caused the regression. |
| CUDA graphs disabled entirely (`DISABLE_CUDA_GRAPH=1`, forced eager mode) | **Same stall** — batch 1 at 27.94s, batch 2 at 8.64s, identical 100%-util/~0%-mem/low-power GPU signature. Disproves the "CUDA graph replay internal sync is hanging" hypothesis directly. |
| Pageable vs. pinned GPU↔CPU memory transfer | Ruled out earlier tonight as the deciding factor — both drift with the same external time-varying pattern independently. |
| Thermal throttling | Ruled out — temps stayed 55-73°C throughout, well under limits. |
| SW power-cap / Dynamic Boost throttle flags | Present in ~10% of samples in one earlier run, but doesn't explain the magnitude/consistency of tonight's stalls on its own — flag stayed at `0x0` (no throttle) during essentially the entire live-test stall windows. |
| Memory allocator retries (`num_alloc_retries`) | Stayed 0 throughout even during 100s+ jobs — ruled out. |
| Driver crash / TDR events | Zero logged in Windows Event Log during any test window tonight. |
| OVRService / vorpX / Duet / Nonary Games stopped | Inconclusive on their own — stopping them didn't visibly change the pattern, but WDDM preemption from a background compositor/overlay wouldn't necessarily show up as a named event either. |

## What's confirmed fixed

- **Memory fragmentation** (`torch_reserved` climbing unboundedly across
  many jobs in one long-lived worker): fixed via
  `torch.cuda.empty_cache()` after every job. Real, measured, holds up
  over multiple jobs. (The paired `expandable_segments:True` setting is
  inert on Windows — see above — so this fix is carried entirely by
  `empty_cache()`.)
- **Logging**: every worker launch now gets its own permanently-dated
  log file (`Logs/worker_stderr_YYYY-MM-DD_HHMMSS.log`), and the
  watchdog incident log moved into the same centralized `Logs/` folder
  at the repo root. Nothing gets overwritten. Verified live during
  tonight's test — new files were created correctly on each worker
  spawn.

## Still open

1. **Root cause is one layer below what this codebase controls.** Every
   test that changes *how* we call the GPU (graphs vs. eager, priority,
   pinned memory, process identity) produced the identical signature.
   The next real lever is either a system setting or working around the
   symptom rather than removing the cause:
   - **Enable Hardware-accelerated GPU Scheduling (HAGS)** in Windows
     graphics settings — this is a system setting change that needs to
     be done manually (Settings → System → Display → Graphics →
     Default graphics settings). Not yet tested tonight since it
     requires your action. Directly relevant given the preemption
     hypothesis.
   - **Adaptive degradation under detected contention** — when the
     watchdog (once its EMA-seeding bug is fixed) detects a sustained
     slow patch, temporarily drop to a cheaper render path (lower FPS,
     smaller batch, or the existing "simple" fallback) rather than
     waiting out a long timeout on every single reply.
2. **Watchdog EMA-seeding bug** — needs a fix that isn't purely
   self-normalizing (see above). Not fixed tonight; flagged with real
   evidence (the 11-batch run that should have tripped it and didn't).

## Files touched tonight

- `backend/app/services/animator.py` — centralized `Logs/` folder,
  per-launch timestamped worker_stderr logs.
- `backend/models/MuseTalk/scripts/musetalk_worker.py` — watchdog
  centralized log path moved to `Logs/`; added `DISABLE_CUDA_GRAPH=1`
  env toggle (real, tested, kept as a permanent diagnostic option — not
  a fix, since eager mode showed the identical stall).
- `Logs/` — new centralized folder at repo root; all prior scattered
  worker/watchdog logs moved in, none overwritten.
