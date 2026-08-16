# REPO_punithVT Fix Log

## 2026-08-14 — MuseTalk animation hangs up to 5-10 minutes per sentence under GPU contention

**Symptom:** Chatting with an avatar (e.g. Silva via the multi-agent
participant picker) played a few words of audio, then went silent for a long
stretch before continuing — repeatedly, one stall per sentence chunk.

**Root cause (confirmed via live log tracing, not guessed):**
`backend/app/services/animator.py`'s persistent MuseTalk worker path
processes each sentence strictly sequentially and blocking:
TTS → `avatar_animator.animate()` → upload → send to client
(`backend/app/websocket.py:907-1059`, `_animate_from_queue`). Nothing reaches
the browser until animation for that sentence finishes or times out.

The GPU (single RTX 4080, 12GB) was simultaneously being used by a second,
unrelated process — the AllTalk TTS server (port 7851, from
`REPO_claude_code_voice_mode`) that speaks Claude Code's own responses. Under
that contention, MuseTalk's worker couldn't reliably get a clean GPU slot, so
both its model-load step and its per-sentence inference step were timing out
against their old generous budgets:
- model load: 300s (cuda) / 600s (cpu)
- inference: 300s first-call / 60s steady-state (cuda), 600s/300s (cpu)

Each of those timeouts had to fully elapse before the code fell back to
simple (non-lip-synced) animation and finally released that sentence's audio
— which is what produced the multi-minute stalls between spoken chunks.
(AllTalk itself also crashed under the same contention — WinError 64 on its
asyncio accept loop, port 7851 — a separate, already-diagnosed issue in the
voice-mode repo, not fixed here.)

**Fix applied:** Flattened both timeouts to a flat 30s in
`backend/app/services/animator.py`:
- `_ensure_worker`: `model_load_timeout = 30` (was `300 if cuda else 600`)
- `_worker_infer`: `infer_timeout = 30` (was `60/300` cuda, `300/600` cpu)

This doesn't fix GPU contention itself — it makes the app fail fast into the
existing simple-animation fallback instead of hanging, which is the
user-facing symptom. Also removed the now-dead `_worker_warmed_up` state
(three sites) since the warm/cold timeout distinction it existed for no
longer applies with a flat timeout.

**Do NOT re-apply as a "fix" without reading this entry:** reverting to
longer timeouts (300s/600s) "to give MuseTalk more time" will reintroduce the
multi-minute stall symptom whenever the GPU is contended — the real fix for
*that* is coordinating GPU access between AllTalk and this backend (not yet
implemented; discussed as "option 3" with the user, still open).

**Verification status:** Syntax-checked (`ast.parse`), not yet re-tested
live in a Kindroid chat session after this change — do that before closing
this out.
