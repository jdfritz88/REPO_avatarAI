from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field


# User Schemas
class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = None


class UserResponse(UserBase):
    id: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# Avatar Schemas
class AvatarBase(BaseModel):
    name: str


class AvatarResponse(AvatarBase):
    id: str
    user_id: str
    image_url: str
    thumbnail_url: Optional[str] = None
    idle_video_url: Optional[str] = None
    idle_playlist_urls: Optional[List[str]] = None
    expression_photos: Optional[List[Dict[str, Any]]] = None
    status: str
    voice_id: Optional[str] = None
    avatar_metadata: Optional[Dict[str, Any]] = Field(None, alias="avatar_metadata")
    created_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


# Session Schemas
class SessionCreate(BaseModel):
    avatar_id: str
    settings: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    id: str
    user_id: str
    avatar_id: str
    status: str
    settings: Optional[Dict[str, Any]] = None
    started_at: datetime
    ended_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# Message Schemas
class MessageBase(BaseModel):
    content: str
    content_type: str = "text"


class MessageCreate(MessageBase):
    session_id: str


class MessageResponse(MessageBase):
    id: str
    session_id: str
    role: str
    audio_url: Optional[str] = None
    video_url: Optional[str] = None
    message_metadata: Optional[Dict[str, Any]] = Field(None, alias="message_metadata")
    created_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


# Conversation Schemas
class ConversationResponse(BaseModel):
    id: str
    session_id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    message_count: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class AvatarRename(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class SessionSettingsUpdate(BaseModel):
    """Allowed editable fields for a session's multi-agent settings (JSON
    column `Session.settings`, previously written-only). Restrict to a known
    allowlist for the same reason as `AvatarMetadataUpdate` below — users
    shouldn't be able to stuff arbitrary keys into the JSON column.
    """

    participant_ids: Optional[List[str]] = Field(default=None, max_length=16)
    turn_mode: Optional[Literal["round_robin", "human_directed", "free_form"]] = None
    addressing_enabled: Optional[bool] = None

    model_config = {"extra": "forbid"}


class ExpressionParams(BaseModel):
    """
    Raw LivePortrait ExpressionEditor node inputs — ranges transcribed
    directly from ComfyUI-AdvancedLivePortrait/nodes.py's INPUT_TYPES (see
    model_experiments/comfyui/expression_editor_client.py's module
    docstring), not guessed. All optional: any field left unset falls back
    to the chosen preset's value, then to the node's own defaults (0 for
    everything except src_ratio=1, crop_factor=1.7).
    """

    rotate_pitch: Optional[float] = Field(default=None, ge=-20, le=20)
    rotate_yaw: Optional[float] = Field(default=None, ge=-20, le=20)
    rotate_roll: Optional[float] = Field(default=None, ge=-20, le=20)
    blink: Optional[float] = Field(default=None, ge=-20, le=5)
    eyebrow: Optional[float] = Field(default=None, ge=-10, le=15)
    wink: Optional[float] = Field(default=None, ge=0, le=25)
    pupil_x: Optional[float] = Field(default=None, ge=-15, le=15)
    pupil_y: Optional[float] = Field(default=None, ge=-15, le=15)
    aaa: Optional[float] = Field(default=None, ge=-30, le=120)
    eee: Optional[float] = Field(default=None, ge=-20, le=15)
    woo: Optional[float] = Field(default=None, ge=-20, le=15)
    smile: Optional[float] = Field(default=None, ge=-0.3, le=1.3)
    src_ratio: Optional[float] = Field(default=None, ge=0, le=1)
    crop_factor: Optional[float] = Field(default=None, ge=1.5, le=2.5)

    model_config = {"extra": "forbid"}


class ExpressionGenerateRequest(BaseModel):
    """
    Generate a LivePortrait expression still. Either `preset` (one of the
    named presets from GET /expression-presets) or `params` (raw slider
    values from the full editor) or both — when both are given, `params`
    values override the preset's for just those fields, everything else
    still comes from the preset. `label` names the library entry; defaults
    to `preset` if omitted, or "custom" for a pure-params request.
    """

    preset: Optional[str] = Field(default=None, min_length=1, max_length=64)
    params: Optional[ExpressionParams] = None
    label: Optional[str] = Field(default=None, min_length=1, max_length=64)

    model_config = {"extra": "forbid"}


class IdleSegmentFromExpressionRequest(BaseModel):
    """Render one Hallo2 idle segment from a previously-generated expression photo."""

    photo_url: str
    slot_index: Optional[int] = Field(default=None, ge=0, lt=6)


class AvatarMetadataUpdate(BaseModel):
    """Allowed editable metadata fields for an avatar.

    Restrict to a known allowlist so users cannot stuff arbitrary keys into
    the JSON column (which would otherwise let them shadow internal flags or
    bloat the row).
    """

    system_prompt: Optional[str] = Field(default=None, max_length=8000)
    personality: Optional[str] = Field(default=None, max_length=2000)
    background_color: Optional[str] = Field(default=None, max_length=32)
    animation_style: Optional[str] = Field(default=None, max_length=32)

    model_config = {"extra": "forbid"}


# Token Schema
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[str] = None
