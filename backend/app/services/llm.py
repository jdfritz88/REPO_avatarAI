"""
LLM facade for the chat pipeline.

Provider-agnostic interface that currently supports Anthropic (default) and
OpenAI. The Anthropic path takes advantage of:

  * **Prompt caching** — the system prompt is wrapped in a content block
    with `cache_control={"type": "ephemeral"}`. Cached reads cost ~10% of
    fresh input, which dominates per-token cost for chatty avatars that
    share a system prompt across many turns. Workspace-isolated as of
    Anthropic's Feb 2026 change.
  * **Extended thinking (opt-in)** — when callers pass `thinking=True`
    we set `thinking={"type": "enabled", "budget_tokens": ...}` so the
    model reasons internally before answering. Reserved for hard turns;
    using it on every turn would multiply token cost.

Exceptions are re-raised as `LLMError` subclasses so the WebSocket pipeline
can distinguish rate-limit / auth / network failures and surface
appropriate user-facing messages.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, AsyncGenerator, Dict, List, Optional

import anthropic
import openai

from app.config import settings

if TYPE_CHECKING:
    from app.services.participants import ParticipantConfig

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant in a real-time avatar conversation system. "
    "Keep replies concise and conversational so they can be spoken aloud."
)

# Extended-thinking budget. Claude 4.x Opus supports up to 128k thinking
# tokens; for an interactive avatar we want responses fast, so we cap the
# budget low. Increase for research/agentic use cases.
_DEFAULT_THINKING_BUDGET = 4096


class LLMError(Exception):
    """Base class — chat pipeline catches this and emits a typed WS error."""


class LLMRateLimited(LLMError):
    """Provider returned 429."""


class LLMAuthError(LLMError):
    """Provider returned 401/403 — usually a misconfigured API key."""


class LLMUnavailable(LLMError):
    """Network failure, timeout, or 5xx from the provider."""


def _cacheable_system(system_prompt: Optional[str]) -> list[dict]:
    """
    Build a system block list with prompt-cache marking applied to the
    (long-lived) system prompt. The SDK accepts either a plain string OR
    a list of blocks; blocks are needed to attach `cache_control` per-block.
    """
    text = system_prompt or DEFAULT_SYSTEM_PROMPT
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _map_anthropic_exception(exc: Exception) -> LLMError:
    """Translate Anthropic SDK exceptions into the typed LLMError hierarchy."""
    if isinstance(exc, anthropic.RateLimitError):
        return LLMRateLimited(str(exc))
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return LLMAuthError(str(exc))
    if isinstance(
        exc,
        (anthropic.APITimeoutError, anthropic.APIConnectionError, anthropic.InternalServerError),
    ):
        return LLMUnavailable(str(exc))
    if isinstance(exc, anthropic.BadRequestError):
        # 400 from Anthropic is usually our bug, not theirs — surface verbatim.
        return LLMError(f"Invalid request to Anthropic: {exc}")
    return LLMError(str(exc))


def _map_openai_exception(exc: Exception) -> LLMError:
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimited(str(exc))
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return LLMAuthError(str(exc))
    if isinstance(
        exc, (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)
    ):
        return LLMUnavailable(str(exc))
    return LLMError(str(exc))


class LLMService:
    """
    LLM Service for AI responses.

    With no arguments, reads the single global provider/model from `settings`
    (the original behavior — used for the app's default non-multi-agent
    chat). Pass `provider`/`model`/`api_key`/`base_url` explicitly to build a
    one-off client for a specific multi-agent participant instead — see
    `build_llm_client()` below, which is the normal way callers outside this
    module should construct participant-specific instances.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.model = model or settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.max_tokens = settings.LLM_MAX_TOKENS
        # Opt-in reasoning cap for OpenAI-compatible "thinking" models (e.g.
        # Ollama cloud glm-*:cloud / kimi-*:cloud). "none" disables reasoning
        # so they return normal `content`. Empty → not sent (default behavior).
        self.reasoning_effort = settings.LLM_REASONING_EFFORT or None

        if self.provider == "anthropic":
            self.client = anthropic.AsyncAnthropic(api_key=api_key or settings.ANTHROPIC_API_KEY)
        elif self.provider == "ollama":
            # Ollama (and vLLM / LM Studio / OpenRouter) speak the OpenAI
            # wire protocol — reuse the OpenAI client against their base URL.
            # Fully local and free; no API key required (the client insists
            # on a non-empty string, so we pass a placeholder).
            resolved_base_url = base_url or settings.OPENAI_BASE_URL or "http://localhost:11434/v1"
            self.client = openai.AsyncOpenAI(
                api_key=api_key or settings.OPENAI_API_KEY or "ollama",
                base_url=resolved_base_url,
            )
            self.provider = "openai"  # downstream code paths are identical
            logger.info(f"LLM provider 'ollama' → OpenAI-compatible client at {resolved_base_url}")
        elif self.provider == "openai":
            self.client = openai.AsyncOpenAI(
                api_key=api_key or settings.OPENAI_API_KEY,
                base_url=base_url or settings.OPENAI_BASE_URL,  # None → api.openai.com
            )

    # ── non-streaming ────────────────────────────────────────────────────────

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        thinking: bool = False,
    ) -> str:
        if self.provider == "anthropic":
            return await self._generate_anthropic(messages, system_prompt, thinking)
        if self.provider == "openai":
            return await self._generate_openai(messages, system_prompt)
        raise LLMError(f"Unsupported LLM provider: {self.provider}")

    async def turn(
        self,
        round_transcript: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Multi-agent entry point — shared shape with `KindroidClient.turn()`
        so the round-robin orchestrator (app/websocket.py `_run_round`) can
        call either without caring which provider is behind it.

        `round_transcript` is a chronological list of {"speaker", "content"}
        covering the human's message plus any earlier participants' replies
        so far this round. Unlike Kindroid, this client is stateless and
        genuinely uses the full transcript — each turn becomes a `messages`
        entry, prefixed with "[Speaker]: " so the model understands it's in
        a group conversation rather than a private 1:1 chat.

        Every turn maps to role="user", including OTHER AI participants'
        turns — not role="assistant". This isn't optional stylistic choice:
        OpenAI-compatible APIs require the message list to end on a
        user/tool turn before they'll generate the next assistant turn
        (confirmed against the real Mistral API — a trailing role="assistant"
        message is rejected with `invalid_request_message_order`). Since
        this specific call is always this participant's OWN next turn, it
        has no legitimate prior "assistant" turns of its own in a single
        round's transcript — everyone else, human or AI, is external input
        from its perspective.
        """
        messages = [
            {
                "role": "user",
                "content": f'[{turn["speaker"]}]: {turn["content"]}',
            }
            for turn in round_transcript
        ]
        # Models tend to mimic the "[Speaker]: " label pattern they see in
        # the transcript and prefix their own reply with it too (observed
        # against the real Mistral API) — tell them not to.
        group_chat_note = (
            "You are in a live multi-participant group conversation; each "
            'prior line is labeled "[Speaker]: " so you know who said what. '
            "Reply with only your own words — do not prefix your reply with "
            "your own name or brackets."
        )
        combined_prompt = f"{system_prompt}\n\n{group_chat_note}" if system_prompt else group_chat_note
        return await self.generate_response(messages, combined_prompt)

    async def _generate_anthropic(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        thinking: bool,
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": _cacheable_system(system_prompt),
            "messages": messages,
        }
        # Extended thinking — model "thinks" privately before answering. The
        # thinking tokens still count against output budget so we widen
        # max_tokens to cover both. Per Anthropic docs, temperature must be 1
        # when extended thinking is enabled.
        if thinking:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": _DEFAULT_THINKING_BUDGET}
            kwargs["max_tokens"] = max(self.max_tokens + _DEFAULT_THINKING_BUDGET, self.max_tokens)
            kwargs["temperature"] = 1.0

        try:
            response = await self.client.messages.create(**kwargs)
        except Exception as e:
            mapped = _map_anthropic_exception(e)
            logger.error(
                "anthropic_call_failed",
                extra={"error_type": type(e).__name__, "mapped": type(mapped).__name__},
            )
            raise mapped from e

        # Find the first text block (skip thinking blocks if any).
        for block in response.content or []:
            if getattr(block, "type", None) == "text" and hasattr(block, "text"):
                self._log_usage(response.usage, thinking)
                return block.text

        raise LLMError("Anthropic response contained no text block")

    def _openai_extra(self) -> dict:
        """Extra request params for the OpenAI-compatible path.

        `reasoning_effort` is forwarded via `extra_body` (not a named kwarg) so
        it lands in the raw request body regardless of SDK version and is a
        no-op for servers that ignore it. Needed for Ollama cloud thinking
        models — see LLM_REASONING_EFFORT.
        """
        if self.reasoning_effort:
            return {"extra_body": {"reasoning_effort": self.reasoning_effort}}
        return {}

    async def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self._openai_extra(),
            )
        except Exception as e:
            mapped = _map_openai_exception(e)
            logger.error("openai_call_failed", extra={"error_type": type(e).__name__})
            raise mapped from e

        return response.choices[0].message.content or ""

    # ── streaming ────────────────────────────────────────────────────────────

    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        if self.provider == "anthropic":
            async for chunk in self._stream_anthropic(messages, system_prompt):
                yield chunk
        elif self.provider == "openai":
            async for chunk in self._stream_openai(messages, system_prompt):
                yield chunk
        else:
            raise LLMError(f"Unsupported LLM provider: {self.provider}")

    async def _stream_anthropic(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=_cacheable_system(system_prompt),
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            mapped = _map_anthropic_exception(e)
            logger.error("anthropic_stream_failed", extra={"error_type": type(e).__name__})
            raise mapped from e

    async def _stream_openai(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **self._openai_extra(),
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            mapped = _map_openai_exception(e)
            logger.error("openai_stream_failed", extra={"error_type": type(e).__name__})
            raise mapped from e

    # ── helpers ──────────────────────────────────────────────────────────────

    def _log_usage(self, usage, thinking: bool) -> None:
        if usage is None:
            return
        try:
            logger.info(
                "llm_usage",
                extra={
                    "in_tokens": getattr(usage, "input_tokens", 0),
                    "out_tokens": getattr(usage, "output_tokens", 0),
                    "cache_create_tokens": getattr(usage, "cache_creation_input_tokens", 0),
                    "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0),
                    "thinking": thinking,
                },
            )
        except Exception:
            pass


def build_llm_client(participant: "ParticipantConfig"):
    """
    Build a per-participant client from a `ParticipantConfig`
    (app.services.participants) — the standard way multi-agent code
    (app/websocket.py `_run_round`) constructs a provider-specific client
    for one selected participant. Returns either an `LLMService`
    (anthropic/openai_compat) or a `KindroidClient`; both expose the shared
    `.turn(round_transcript, system_prompt)` interface the orchestrator
    relies on, so callers don't need to know which one they got.
    """
    if participant.type == "kindroid":
        from app.services.kindroid import KindroidClient

        return KindroidClient(api_key=participant.api_key, ai_id=participant.ai_id)

    provider = "anthropic" if participant.type == "anthropic" else "openai"
    return LLMService(
        provider=provider,
        model=participant.model,
        api_key=participant.api_key,
        base_url=participant.base_url,
    )


# Global instance
llm_service = LLMService()
