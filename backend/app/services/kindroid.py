"""
Kindroid API client.

Kindroid's conversation memory is server-side, keyed to `ai_id` — unlike
Anthropic/OpenAI-compatible APIs, the caller cannot send message history and
there's no endpoint to list a key's kins (each kin is configured manually via
KINDROID_<NAME>_API_KEY / KINDROID_<NAME>_AI_ID, see app/services/participants.py).

`turn()` synthesizes a short recap of the round-so-far into the message text
so a kin has some awareness of a multi-party conversation despite the API
only accepting a single string per call.
"""

import json
import logging
from typing import Optional

import httpx

from app.services.llm import LLMAuthError, LLMError, LLMRateLimited, LLMUnavailable

logger = logging.getLogger(__name__)

KINDROID_BASE_URL = "https://api.kindroid.ai/v1"


class KindroidClient:
    def __init__(self, api_key: str, ai_id: str):
        self.api_key = api_key
        self.ai_id = ai_id

    async def respond(self, message: str) -> str:
        """POST /send-message and return the kin's reply as plain text."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(
                    f"{KINDROID_BASE_URL}/send-message",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"ai_id": self.ai_id, "message": message},
                )
            except httpx.TimeoutException as e:
                raise LLMUnavailable(f"Kindroid request timed out: {e}") from e
            except httpx.RequestError as e:
                raise LLMUnavailable(f"Kindroid request failed: {e}") from e

        if resp.status_code == 200:
            raw = resp.text
            # The API sometimes wraps the reply as a JSON-encoded string
            # (observed: literal quote characters in the response body) —
            # decode if so, otherwise treat the body as already-plain text.
            try:
                decoded = json.loads(raw)
                if isinstance(decoded, str):
                    return decoded.strip()
            except (json.JSONDecodeError, TypeError):
                pass
            return raw.strip()
        if resp.status_code in (401, 403):
            raise LLMAuthError(f"Kindroid auth failed ({resp.status_code}): {resp.text[:200]}")
        if resp.status_code == 429:
            raise LLMRateLimited(f"Kindroid rate limited: {resp.text[:200]}")
        if resp.status_code >= 500:
            raise LLMUnavailable(f"Kindroid server error ({resp.status_code}): {resp.text[:200]}")
        raise LLMError(f"Kindroid request failed ({resp.status_code}): {resp.text[:200]}")

    async def turn(self, round_transcript: list[dict], system_prompt: Optional[str] = None) -> str:
        """
        `round_transcript` is a chronological list of {"speaker", "content"}
        dicts covering the human's message plus any earlier participants'
        replies so far this round. Kindroid can't ingest that as structured
        history, so we fold it into one recap string.
        """
        if not round_transcript:
            return await self.respond("")

        if len(round_transcript) == 1:
            # Just the human's message — no recap framing needed.
            return await self.respond(round_transcript[0]["content"])

        lines = [f'{turn["speaker"]} said: "{turn["content"]}"' for turn in round_transcript]
        recap = (
            "[Group conversation — multiple participants are talking.] "
            + " ".join(lines)
            + " Your turn to respond."
        )
        return await self.respond(recap)
