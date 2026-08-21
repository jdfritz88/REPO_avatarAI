"""
User-managed AI-provider credentials — the self-service replacement for
the old "hand-edit a .env file and restart the server" flow (see
app/services/participants.py, which still exists for backward
compatibility but has no working default right now — ANTHROPIC_API_KEY
was never actually configured, confirmed live via a real failed chat
message: "Could not resolve authentication method").

Default API base URLs and REAL verification calls for the well-known
providers — each one confirmed against live docs, not guessed:
  - Anthropic: GET /v1/models with x-api-key + anthropic-version headers.
    Free — a metadata lookup, no completion tokens spent.
  - Mistral / OpenAI (and any OpenAI-wire-compatible "custom" provider):
    GET {base_url}/models with Authorization: Bearer. Same no-token-cost
    shape.
  - Kindroid: no lightweight "verify key" endpoint exists at all — its own
    API surface is just POST /send-message (see app/services/kindroid.py's
    docstring: "no endpoint to list a key's kins"). A real Connect check
    for Kindroid is therefore an actual minimal message send; there is no
    way to check the key without one real round-trip against the kin.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URLS = {
    "anthropic": "https://api.anthropic.com",
    "mistral": "https://api.mistral.ai/v1",
    "openai": "https://api.openai.com/v1",
    "kindroid": "https://api.kindroid.ai/v1",
}

ANTHROPIC_VERSION = "2023-06-01"


class ConnectResult:
    def __init__(self, ok: bool, message: str):
        self.ok = ok
        self.message = message


async def resolve_base_url(provider: str, api_base_url: Optional[str]) -> str:
    if api_base_url:
        return api_base_url.rstrip("/")
    default = DEFAULT_BASE_URLS.get(provider)
    if not default:
        raise ValueError(f"No default API URL known for provider {provider!r} — api_base_url is required")
    return default


async def test_connection(
    provider: str, api_key: str, api_base_url: Optional[str], kindroid_ai_id: Optional[str] = None,
) -> ConnectResult:
    """
    Make ONE real, minimal authenticated call against the provider to
    verify the key actually works. Never raises — always returns a
    ConnectResult so the API layer can persist the outcome regardless of
    what happened.
    """
    try:
        base_url = await resolve_base_url(provider, api_base_url)
    except ValueError as e:
        return ConnectResult(False, str(e))

    try:
        if provider == "anthropic":
            return await _test_anthropic(base_url, api_key)
        if provider == "kindroid":
            if not kindroid_ai_id:
                return ConnectResult(False, "Kindroid requires a kin AI ID as well as an API key")
            return await _test_kindroid(api_key, kindroid_ai_id)
        # mistral / openai / custom — all OpenAI-wire-compatible
        return await _test_openai_compatible(base_url, api_key)
    except httpx.TimeoutException:
        return ConnectResult(False, "Connection timed out")
    except httpx.RequestError as e:
        return ConnectResult(False, f"Network error: {e}")
    except Exception as e:  # noqa: BLE001 — this result surfaces directly to the user, never guess-swallow silently
        logger.error(f"Unexpected error testing {provider} connection: {e}")
        return ConnectResult(False, f"Unexpected error: {e}")


async def _test_anthropic(base_url: str, api_key: str) -> ConnectResult:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/v1/models",
            headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
        )
    if resp.status_code == 200:
        count = len(resp.json().get("data", []))
        return ConnectResult(True, f"Connected — {count} model(s) available")
    if resp.status_code in (401, 403):
        return ConnectResult(False, "Authentication failed — check the API key")
    return ConnectResult(False, f"Unexpected response ({resp.status_code}): {resp.text[:200]}")


async def _test_openai_compatible(base_url: str, api_key: str) -> ConnectResult:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
    if resp.status_code == 200:
        data = resp.json().get("data", [])
        return ConnectResult(True, f"Connected — {len(data)} model(s) available")
    if resp.status_code == 401:
        return ConnectResult(False, "Authentication failed — check the API key")
    return ConnectResult(False, f"Unexpected response ({resp.status_code}): {resp.text[:200]}")


async def _test_kindroid(api_key: str, ai_id: str) -> ConnectResult:
    from app.services.kindroid import KindroidClient
    from app.services.llm import LLMAuthError, LLMError

    client = KindroidClient(api_key=api_key, ai_id=ai_id)
    try:
        # No lightweight verification endpoint exists (see module
        # docstring) — this is a REAL minimal message, the only way to
        # actually confirm the key+kin pair works against Kindroid's API.
        await client.respond("(connection check — no reply needed)")
        return ConnectResult(True, "Connected — kin responded successfully")
    except LLMAuthError as e:
        return ConnectResult(False, f"Authentication failed: {e}")
    except LLMError as e:
        return ConnectResult(False, f"Kindroid error: {e}")


def mask_key(api_key: str) -> str:
    if len(api_key) <= 4:
        return "*" * len(api_key)
    return "*" * (len(api_key) - 4) + api_key[-4:]
