import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.users import get_current_user
from app.config import settings
from app.models import User
from app.services.participants import get_available_participants

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/participants")
async def list_participants(current_user: Optional[User] = Depends(get_current_user)):
    """
    List selectable multi-agent participants (Mistral variants, Kindroid
    kins, etc.) currently configured via environment — the frontend's
    participant picker calls this to populate its options. Never returns
    API keys, only ids/types/names.
    """
    return [
        {"id": p.id, "type": p.type, "name": p.name, "model": p.model or None}
        for p in get_available_participants()
    ]


@router.get("/mistral-models")
async def list_mistral_models(current_user: Optional[User] = Depends(get_current_user)):
    """
    Live model list from Mistral's own API (GET /v1/models), so the
    frontend's model picker shows what's actually available on this key
    rather than a hardcoded string. Requires MISTRAL_API_KEY to be set.
    """
    if not settings.MISTRAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MISTRAL_API_KEY is not configured",
        )
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.mistral.ai/v1/models",
                headers={"Authorization": f"Bearer {settings.MISTRAL_API_KEY}"},
            )
    except httpx.RequestError as e:
        logger.error(f"Mistral models request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Could not reach Mistral API"
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Mistral API returned {resp.status_code}",
        )

    data = resp.json().get("data", [])
    return [{"id": m["id"]} for m in data]
