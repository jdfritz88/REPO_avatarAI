# Live Debugging Session Protocol

Reusable instructions for any live debugging/monitoring session on this app (e.g. the MuseTalk GPU stall investigation). Paste this as the prompt when starting a new session.

---

No guessing on errors or bugs. Every claim about root cause must be backed by something you actually read or observed — real code, real logs, a real reproduction — never inferred or assumed. If you're not sure, say so and go verify; do not present a guess as a finding. Second-guess every conclusion before reporting it.

No shortcuts, no workarounds, no dummies, no placeholders, no stubs, no nubs. Fixes must address the real root cause, not paper over the symptom.

**Logging**: write findings to the repo's `Logs/` folder. Use one log file per session/day (or every few hours) — do not create a new file for every individual error or observation. Append to the current session's log as things happen; never overwrite an existing log file.

**Process monitoring**: while reproducing or watching the issue, continuously watch GPU telemetry (utilization, memory, power draw, clock speed, temperature) AND a Task-Manager-equivalent view of per-process CPU usage. Every single time an issue shows up in animation rendering (a stall, an error, a timeout), take a snapshot of both and record it in the running session log — not just the GPU side.

**Verify externally**: before concluding a root cause, check it against outside sources (official docs, upstream GitHub issues, known community reports) rather than relying only on internal reasoning — "run it by the internet."

**Live conversation testing**: when testing an avatar (e.g. Rae), have real conversations through the actual browser UI — click into the app, send a message, read the avatar's actual reply, then compose a genuine contextual response to what it actually said. Never use a scripted/timed message sequence that fires messages regardless of the avatar's replies — that is not a conversation. Do five separate sessions of roughly five minutes of real back-and-forth each. If clicks keep misfiring, reopen a new browser tab rather than continuing to retry blindly in the same one.

Watch the animation rendering live in the browser while this happens, not just from logs after the fact.
