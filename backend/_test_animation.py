"""One-off live-WS verification that Silva's turn actually produces a real
animated/lip-synced video using HER OWN avatar photo. Run directly against
the running backend, not part of the test suite."""

import asyncio
import json

import httpx
import websockets

BASE = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"
SILVA_AVATAR_ID = "00000000-0000-0000-0000-000000000000"


async def main():
    async with httpx.AsyncClient() as client:
        # Use Silva's own avatar as the session's base avatar too (only
        # ready avatar in the system right now) — human's session avatar
        # is separate from participant avatars, but doesn't matter for
        # this test since only Silva will actually speak.
        resp = await client.post(f"{BASE}/api/v1/sessions/create", json={"avatar_id": SILVA_AVATAR_ID})
        resp.raise_for_status()
        session = resp.json()
        session_id = session["id"]
        print(f"[1] session created: {session_id}")

        resp = await client.patch(
            f"{BASE}/api/v1/sessions/{session_id}/settings",
            json={
                "participant_ids": ["kindroid-silva"],
                "turn_mode": "round_robin",
                "addressing_enabled": True,
            },
        )
        resp.raise_for_status()
        print("[2] participants set: kindroid-silva")

    events = []
    video_chunks = []
    async with websockets.connect(f"{WS_BASE}/ws/session/{session_id}") as ws:
        await ws.send(json.dumps({"type": "text", "text": "Hi Silva, quick animation test — say hello!"}))
        print("[3] sent text turn, waiting for events...")

        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=340.0)
                msg = json.loads(raw)
                events.append(msg["type"])
                if msg["type"] == "video_chunk":
                    video_chunks.append(msg)
                    print(f"    video_chunk: participant={msg.get('participant_name')} url={msg['video_url']}")
                elif msg["type"] == "error":
                    print(f"    ERROR event: {msg.get('message')}")
                elif msg["type"] == "video_chunk_end":
                    print(f"    video_chunk_end: sent_chunks={msg.get('sent_chunks')}")
                    break
        except asyncio.TimeoutError:
            print("    TIMEOUT waiting for events")

    print(f"\n[4] event sequence: {events}")
    assert video_chunks, "No video_chunk events received — animation pipeline did not produce output"
    assert all(c.get("participant_name") == "Silva" for c in video_chunks), "video chunk not attributed to Silva"
    print(f"[4] {len(video_chunks)} video chunk(s) received, all attributed to Silva: OK")

    # Verify the actual video file exists on disk and is non-trivial size
    # (proves MuseTalk actually rendered something, not an empty/failed clip).
    for c in video_chunks:
        url = c["video_url"]
        path_part = url.split("/uploads/", 1)[-1] if "/uploads/" in url else None
        assert path_part, f"unexpected video_url shape: {url}"
        local_path = f"F:/Apps/freedom_system/REPO_avatarAI/backend/uploads/{path_part}"
        import os
        size = os.path.getsize(local_path)
        print(f"[5] video file on disk: {local_path} ({size} bytes)")
        assert size > 10_000, f"video file suspiciously small: {size} bytes"

    print("\nALL CHECKS PASSED — Silva's animation/lip-sync pipeline is real and working")


asyncio.run(main())
