"""
Shared mutual-exclusion lock so the two GPU-bound render engines in this
app — MuseTalk (live chat lip-sync, app/services/animator.py) and Hallo2
(background idle-playlist generation, app/services/hallo2_animator.py) —
never run CUDA inference at the same moment on this machine's one GPU.

Confirmed twice in the same session (2026-08-16) that running both
concurrently doesn't just slow things down, it breaks them outright:
MuseTalk model-load timeouts and Hallo2 segment renders blowing past
their own timeout, in both cases while the other engine was mid-inference
on the same card. A lock forces strict alternation instead of contention.

Both engines acquire this immediately before sending a job to their
persistent worker subprocess and release it right after that job's result
comes back — held only around the actual GPU call, not around request
setup, file I/O, or queuing — so whichever engine asks first simply goes
first, and the other waits its turn rather than colliding with it.
"""
import asyncio

gpu_lock = asyncio.Lock()
