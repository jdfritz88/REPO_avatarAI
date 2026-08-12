# Session Report — 2026-08-12 (night session)

Branch: `branch02_fixes` (REPO_punithVT), clean working tree — no code changes
made in this repo tonight, investigation only. One commit was made in the
voice-mode repo (details below).

---

## 1. Voice mode — fixed and committed

**Problem:** Claude was rarely speaking; you'd made fixes in
`REPO_claude_code_voice_mode` but weren't sure they were active.

**Findings:**
- Global Stop hook at `C:\Users\jespe\.claude\settings.json` points to
  `REPO_claude_code_voice_mode/.claude/hooks/speak_on_stop.py`. Confirmed
  active — it strips code/markdown from every response and speaks the
  remaining prose automatically via `speak_text()`, no manual tool call
  needed.
- The repo had ~858 lines of uncommitted fixes (device persistence, VAD
  config, push-to-talk mute fix, streaming TTS improvements) plus the new
  Stop-hook file itself, untracked.

**Action taken:** Committed as `6e0e72d` on
`Branch09_claude_code_voice_mode-start_VAD_install_and_update_push_to_talk`.
Excluded `logs/`, a stray `nul` file (Windows redirect artifact), and
`CLAUDE.md.lnk` (shortcut file) as non-source. **Not pushed to remote.**

---

## 2. AvatarAI stack — brought up locally

Started manually (not via `start_punitVT.bat`, so output could be piped to
and monitored by Claude directly instead of opening separate windows):

| Service | How | Port | Status at session end |
|---|---|---|---|
| Redis | `redis-portable/redis-server.exe redis.conf` | 6379 | Up (bg task `bt0lvgvcs`) |
| Postgres | pre-existing Windows service `postgresql-x64-15` | 5432 | Already running |
| Backend (FastAPI/uvicorn) | `backend/venv/Scripts/python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000` | 8000 | Up (bg task `bwpw5wexx`) |
| Frontend (Next.js/Turbopack) | `npm run dev` in `frontend/` | 3000 | Up (bg task `bupr40p13`) |

**Important:** these are background processes tied to tonight's Claude Code
session. They will very likely need to be restarted next session — either
re-run `start_punitVT.bat` yourself, or ask Claude to launch them again the
same way so output stays watchable.

A persistent error-watching Monitor (task `bd01nfk05`) was also running
across all three logs — that's what caught the Kindroid/Anthropic bug below.

---

## 3. "Why is there a login screen?" — resolved, not a concern

Traced end-to-end:
- Register/login POSTs to `http://localhost:8000/api/v1/users/register|login`
  — your own local FastAPI process.
- Backend writes to `postgresql://avatar_user:***@localhost:5432/avatar_db`
  — your own local Postgres service.
- Password is bcrypt-hashed before storage (`backend/app/api/v1/users.py`).
- **Nothing leaves the machine.** "AvatarAI" is just the name of the
  open-source template this project (`PunithVT/ai-avatar-system`) was built
  from — not a third-party hosted service.
- Auth exists because the upstream project is designed as a multi-tenant web
  service (every avatar/conversation/voice profile is scoped to a
  `user_id`); several past commits (`1da1eb1`, `3518c67`) closed real
  cross-tenant data-leak bugs that the ownership checks now prevent.
- **You don't need an account.** "Continue as Guest" already exists
  (`AuthModal.tsx:109-113`) — it sets a synthetic `token: 'guest'`, which
  `frontend/lib/api.ts:13-17` treats as "send no Authorization header,"
  which makes the backend fall back to a seeded `demo-user`
  (`backend/main.py:99-116`, gated on `DEBUG=True`, which is set).

**Open question (not yet implemented):** whether to strip the login modal
entirely and auto-route straight to the demo-user guest path. Confirmed
technically trivial (~5 line change, no backend changes) since the guest
path already does exactly this. **Decision: not yet — you said "not yet,"
we moved on to testing guest mode instead.**

---

## 4. Live-traced a "Continue as Guest" click

Watched backend logs in real time as you clicked through:
- Confirmed the click itself is client-only (no network call) — it just
  sets local Zustand state.
- First real request: `GET` avatars list, `WHERE avatars.user_id = 'demo-user'`,
  empty result. This repeats every 5s — traced to
  `frontend/components/AvatarList.tsx:45` (`refetchInterval: 5000`), a
  React Query poll, not user activity. **Not a bug**, just normal polling.
- You then selected an existing avatar (id `00000000-0000-0000-0000-000000000000`,
  name "Silva") and started a chat — this created a new row in `sessions`
  and `conversations`, and inserted your message "Hi Silva" into `messages`.

---

## 5. BUG FOUND: chatting with "Silva" silently falls back to Anthropic (empty key) instead of Kindroid

**Symptom:** Sending "Hi Silva" produced no reply. Backend logged:
```
app.services.llm - ERROR - anthropic_stream_failed
app.websocket - ERROR - LLM producer error [...]: "Could not resolve
authentication method. Expected one of api_key, auth_token, or credentials
to be set..."
```

**Root cause (confirmed by reading code, not guessed):**

There are **two separate chat pipelines** in `backend/app/websocket.py`:

1. **Legacy single-agent pipeline** (`_handle_text_input_inner_legacy`) —
   always uses one hardcoded global `llm_service` instance
   (`app/services/llm.py`), built from `settings.LLM_PROVIDER` (`.env:39`
   = `anthropic`) and `settings.ANTHROPIC_API_KEY` (`.env:19` = **empty**).
2. **Multi-agent pipeline** (`_handle_multi_agent_turn` → `_run_round`) —
   builds a per-participant client via `build_llm_client(participant)`,
   correctly resolving Kindroid kins (incl. Silva) from
   `KINDROID_SILVA_API_KEY` / `KINDROID_SILVA_AI_ID` in
   `app/services/participants.py`.

The dispatcher (`_handle_text_input_inner`, `websocket.py:643-654`) picks
pipeline 2 **only if** `session.settings["participant_ids"]` was set via
`PATCH /api/v1/sessions/{id}/settings` or a `set_participants` WS message —
i.e., only through the multi-agent **participant picker** UI.

Clicking an avatar card in the normal single-agent Chat screen never sets
`participant_ids`, so it always falls to pipeline 1 (global Anthropic
client), **regardless of which avatar/kin you clicked**. The fact that
`KINDROID_SILVA_AVATAR_ID` in `.env` happens to equal Silva's `Avatar.id`
is coincidental context, not a connection any code path actually uses in
the single-agent flow — that ID linkage is only consumed by the multi-agent
path.

**Your Kindroid/Mistral keys are fine and correctly configured** — this is
purely a routing bug, not a credentials problem.

**Two fix options discussed, neither implemented yet:**
- (a) Use the existing multi-agent/participant-picker UI (from commit
  `6e069fc`) and select Silva there as a participant.
- (b) Patch the legacy single-agent path so selecting an avatar with a
  matching `KINDROID_*_AVATAR_ID` auto-routes to that kin's client instead
  of the global Anthropic one.

**Decision: open — pick up here next session.**

---

## Next session — pick up here

1. Decide fix (a) vs (b) for the Silva/Kindroid routing bug above, or try
   the participant-picker UI first to confirm it actually works end-to-end.
2. Decide whether to strip the login modal in favor of always-guest mode
   (§3), still open.
3. Restart the three services (Redis/backend/frontend) — they were only
   running for tonight's session.
4. `git push` the voice-mode commit (`6e0e72d`) if you want it on the
   remote — it wasn't pushed tonight.
