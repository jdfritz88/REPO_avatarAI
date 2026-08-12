"""
Participant registry for multi-agent conversations.

Builds the list of AI participants selectable into a session from environment
configuration: Mistral (direct API + OpenRouter proxy, both OpenAI-wire-
compatible), the existing Anthropic config, and Kindroid kins — auto-
discovered from any KINDROID_<NAME>_API_KEY / KINDROID_<NAME>_AI_ID pair, so
adding a kin is two new .env lines with no code change.
"""

import os
import re
from dataclasses import dataclass
from typing import Literal, Optional

from dotenv import dotenv_values

from app.config import _ENV_FILE, settings

ParticipantType = Literal["anthropic", "openai_compat", "kindroid"]


@dataclass(frozen=True)
class ParticipantConfig:
    id: str
    type: ParticipantType
    name: str
    # anthropic / openai_compat:
    api_key: str = ""
    base_url: Optional[str] = None
    model: str = ""
    # kindroid:
    ai_id: str = ""
    # Optional — this participant's own avatar (Avatar.id), used so its
    # responses animate its own face instead of the session's shared one.
    # Falls back to the session avatar when unset (e.g. text-only Mistral).
    avatar_id: str = ""


_KINDROID_KEY_RE = re.compile(r"^KINDROID_([A-Z0-9_]+)_API_KEY$")


def _discover_kindroid_kins() -> list[ParticipantConfig]:
    # KINDROID_<NAME>_* pairs are dynamically named, so they can't be static
    # Settings fields (see config.py's `extra: "ignore"`) — pydantic-settings
    # loads .env into its own object without exporting to the real process
    # os.environ, so we read the .env file directly here (merged under real
    # os.environ, which wins if the same var is actually exported).
    env = {**dotenv_values(_ENV_FILE), **os.environ}
    kins: list[ParticipantConfig] = []
    for env_name, value in env.items():
        m = _KINDROID_KEY_RE.match(env_name)
        if not m or not value:
            continue
        slug = m.group(1)  # e.g. "SILVA"
        ai_id = env.get(f"KINDROID_{slug}_AI_ID", "")
        if not ai_id:
            continue
        kins.append(
            ParticipantConfig(
                id=f"kindroid-{slug.lower()}",
                type="kindroid",
                name=slug.replace("_", " ").title(),
                api_key=value,
                ai_id=ai_id,
                avatar_id=env.get(f"KINDROID_{slug}_AVATAR_ID", ""),
            )
        )
    return kins


def get_available_participants() -> list[ParticipantConfig]:
    """Build the full list of selectable participants from current config."""
    participants: list[ParticipantConfig] = []

    if settings.MISTRAL_API_KEY:
        participants.append(
            ParticipantConfig(
                id="mistral-direct",
                type="openai_compat",
                name="Mistral",
                api_key=settings.MISTRAL_API_KEY,
                base_url="https://api.mistral.ai/v1",
                model="mistral-large-latest",
            )
        )

    if settings.OPENROUTER_API_KEY:
        participants.append(
            ParticipantConfig(
                id="mistral-openrouter",
                type="openai_compat",
                name="Mistral (OpenRouter)",
                api_key=settings.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                model="mistralai/mistral-large-2512",
            )
        )

    if settings.ANTHROPIC_API_KEY:
        participants.append(
            ParticipantConfig(
                id="anthropic-default",
                type="anthropic",
                name="Claude",
                api_key=settings.ANTHROPIC_API_KEY,
                model=settings.LLM_MODEL,
            )
        )

    participants.extend(_discover_kindroid_kins())
    return participants


def get_participant(participant_id: str) -> Optional[ParticipantConfig]:
    """Look up a single participant by id. Re-scans env each call (cheap —
    at most a handful of participants) so a newly-added kin is picked up
    without restarting the backend."""
    for p in get_available_participants():
        if p.id == participant_id:
            return p
    return None
