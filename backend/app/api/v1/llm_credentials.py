"""
Self-service AI-provider credential management — Settings > API in the
frontend. See app/services/llm_credentials.py for the real per-provider
verification logic and app/models.py's LlmCredential for why this exists
(replaces the old .env-file-and-restart flow).
"""
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.users import get_current_user
from app.database import get_db
from app.models import LlmCredential, User
from app.schemas import (
    LlmCredentialCreate,
    LlmCredentialReorder,
    LlmCredentialResponse,
    LlmCredentialUpdate,
)
from app.services.llm_credentials import DEFAULT_BASE_URLS, mask_key, test_connection

logger = logging.getLogger(__name__)
router = APIRouter()


def _user_id(current_user: Optional[User]) -> str:
    return current_user.id if current_user else "demo-user"


def _to_response(cred: LlmCredential) -> LlmCredentialResponse:
    return LlmCredentialResponse(
        id=cred.id,
        provider=cred.provider,
        label=cred.label,
        api_key_masked=mask_key(cred.api_key),
        api_base_url=cred.api_base_url,
        kindroid_ai_id=cred.kindroid_ai_id,
        is_favorite=cred.is_favorite,
        sort_order=cred.sort_order,
        last_verified_at=cred.last_verified_at,
        last_verify_ok=cred.last_verify_ok,
        last_verify_message=cred.last_verify_message,
        created_at=cred.created_at,
    )


@router.get("/", response_model=List[LlmCredentialResponse])
async def list_credentials(
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    result = await db.execute(
        select(LlmCredential)
        .where(LlmCredential.user_id == _user_id(current_user))
        .order_by(LlmCredential.sort_order, LlmCredential.created_at)
    )
    return [_to_response(c) for c in result.scalars().all()]


@router.get("/default-base-urls")
async def get_default_base_urls():
    """Known-provider default API URLs, so the frontend can show a placeholder/hint for the 4 built-ins vs require one for "custom"."""
    return DEFAULT_BASE_URLS


@router.post("/", response_model=LlmCredentialResponse)
async def create_credential(
    body: LlmCredentialCreate,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    provider = body.provider.strip().lower()
    if provider not in DEFAULT_BASE_URLS and not body.api_base_url:
        raise HTTPException(
            status_code=400,
            detail=f"api_base_url is required for provider {provider!r} (not one of the built-in {sorted(DEFAULT_BASE_URLS)})",
        )
    if provider == "kindroid" and not body.kindroid_ai_id:
        raise HTTPException(status_code=400, detail="kindroid_ai_id is required for provider 'kindroid'")

    result = await db.execute(
        select(LlmCredential.sort_order)
        .where(LlmCredential.user_id == _user_id(current_user))
        .order_by(LlmCredential.sort_order.desc())
        .limit(1)
    )
    max_sort = result.scalar_one_or_none()
    next_sort = (max_sort + 1) if max_sort is not None else 0

    cred = LlmCredential(
        user_id=_user_id(current_user),
        provider=provider,
        label=body.label.strip(),
        api_key=body.api_key.strip(),
        api_base_url=body.api_base_url.strip() if body.api_base_url else None,
        kindroid_ai_id=body.kindroid_ai_id.strip() if body.kindroid_ai_id else None,
        sort_order=next_sort,
    )
    db.add(cred)
    await db.commit()
    await db.refresh(cred)

    # Verify immediately on creation — a saved credential the user hasn't
    # yet seen fail is worse than one that's honestly marked untested.
    outcome = await test_connection(cred.provider, cred.api_key, cred.api_base_url, cred.kindroid_ai_id)
    cred.last_verified_at = datetime.now(timezone.utc)
    cred.last_verify_ok = outcome.ok
    cred.last_verify_message = outcome.message
    await db.commit()
    await db.refresh(cred)

    logger.info(f"Created LLM credential {cred.id} ({provider}/{cred.label}) for user {_user_id(current_user)}: verify_ok={outcome.ok}")
    return _to_response(cred)


async def _get_owned(db: AsyncSession, credential_id: str, current_user: Optional[User]) -> LlmCredential:
    result = await db.execute(select(LlmCredential).where(LlmCredential.id == credential_id))
    cred = result.scalar_one_or_none()
    if not cred:
        raise HTTPException(status_code=404, detail="Credential not found")
    if cred.user_id != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Not authorised to modify this credential")
    return cred


@router.patch("/{credential_id}", response_model=LlmCredentialResponse)
async def update_credential(
    credential_id: str,
    body: LlmCredentialUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    cred = await _get_owned(db, credential_id, current_user)
    if body.label is not None:
        cred.label = body.label.strip()
    if body.api_key is not None:
        cred.api_key = body.api_key.strip()
    if body.api_base_url is not None:
        cred.api_base_url = body.api_base_url.strip() or None
    if body.kindroid_ai_id is not None:
        cred.kindroid_ai_id = body.kindroid_ai_id.strip() or None
    if body.is_favorite is not None:
        cred.is_favorite = body.is_favorite
    await db.commit()
    await db.refresh(cred)
    return _to_response(cred)


@router.delete("/{credential_id}")
async def delete_credential(
    credential_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    cred = await _get_owned(db, credential_id, current_user)
    await db.delete(cred)
    await db.commit()
    return {"deleted": True}


@router.post("/{credential_id}/connect", response_model=LlmCredentialResponse)
async def connect_credential(
    credential_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Re-run the real provider verification call on demand (the "Connect" button)."""
    cred = await _get_owned(db, credential_id, current_user)
    outcome = await test_connection(cred.provider, cred.api_key, cred.api_base_url, cred.kindroid_ai_id)
    cred.last_verified_at = datetime.now(timezone.utc)
    cred.last_verify_ok = outcome.ok
    cred.last_verify_message = outcome.message
    await db.commit()
    await db.refresh(cred)
    return _to_response(cred)


@router.put("/reorder", response_model=List[LlmCredentialResponse])
async def reorder_credentials(
    body: LlmCredentialReorder,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Persist a new drag-and-drop order — body is the full list of this user's credential ids in their new order."""
    uid = _user_id(current_user)
    result = await db.execute(select(LlmCredential).where(LlmCredential.user_id == uid))
    owned = {c.id: c for c in result.scalars().all()}

    missing = set(body.ordered_ids) - set(owned)
    if missing:
        raise HTTPException(status_code=400, detail=f"Unknown or not-owned credential id(s): {sorted(missing)}")

    for idx, cid in enumerate(body.ordered_ids):
        owned[cid].sort_order = idx
    await db.commit()

    result = await db.execute(
        select(LlmCredential).where(LlmCredential.user_id == uid).order_by(LlmCredential.sort_order)
    )
    return [_to_response(c) for c in result.scalars().all()]
