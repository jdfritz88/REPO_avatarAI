import asyncio
import logging
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.users import get_current_user
from app.database import get_db
from app.models import Avatar, User
from app.schemas import (
    AvatarMetadataUpdate,
    AvatarRename,
    AvatarResponse,
    ExpressionGenerateRequest,
    IdleSegmentFromExpressionRequest,
)
from app.services import blink
from app.services.animator import avatar_animator
from app.services.expression_editor import ExpressionEditorServiceError, expression_editor_service
from app.services.hallo2_animator import MAX_PLAYLIST_SEGMENTS, hallo2_animator
from app.services.idle_playlist import ensure_idle_playlist, is_generation_in_progress
from app.services.avatar_processor import avatar_processor
from app.services.storage import resolve_local_image, storage_service

logger = logging.getLogger(__name__)
router = APIRouter()
TMPDIR = Path(tempfile.gettempdir())


def _user_id(current_user: Optional[User]) -> str:
    return current_user.id if current_user else "demo-user"


def _validate_uuid(avatar_id: str) -> None:
    """
    Reject anything that isn't a UUID to keep S3 keys + filesystem paths safe.

    Returns 404 (not 400) for malformed IDs: a non-UUID can never name an
    existing avatar, so "not found" is the correct REST semantics and it
    avoids leaking the fact that IDs are UUIDs to a probing client.
    """
    try:
        uuid.UUID(avatar_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=404, detail="Avatar not found")


@router.post("/upload", response_model=AvatarResponse, status_code=status.HTTP_201_CREATED)
async def upload_avatar(
    name: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Upload and process an avatar image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPG, PNG, WEBP)")

    file_data: bytes = await file.read()  # type: ignore[assignment]
    if len(file_data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File must be under 10 MB")

    avatar_id = str(uuid.uuid4())
    suffix = Path(file.filename or "avatar.jpg").suffix or ".jpg"
    temp_orig = TMPDIR / f"{avatar_id}_original{suffix}"
    temp_processed = TMPDIR / f"{avatar_id}_processed.jpg"
    metadata: dict = {}

    try:
        temp_orig.write_bytes(file_data)

        _, metadata = await avatar_processor.process_image(str(temp_orig), str(temp_processed))

        image_key = f"avatars/{avatar_id}/image.jpg"
        image_url = await storage_service.upload_file(
            temp_processed.read_bytes(), image_key, content_type="image/jpeg"
        )

        # Resolve the thumbnail path defensively: an empty/missing value would
        # make Path("") == Path(".") (the cwd), so guard against it explicitly
        # and fall back to the processed image when there's no real thumbnail.
        thumb_value = metadata.get("thumbnail_path") or ""
        thumb_file = Path(thumb_value) if thumb_value else None
        thumb_key = f"avatars/{avatar_id}/thumbnail.jpg"
        thumb_bytes = (
            thumb_file.read_bytes()
            if thumb_file and thumb_file.is_file()
            else temp_processed.read_bytes()
        )
        thumbnail_url = await storage_service.upload_file(
            thumb_bytes, thumb_key, content_type="image/jpeg"
        )

    except HTTPException:
        raise
    except Exception as e:
        from PIL import UnidentifiedImageError

        if isinstance(e, UnidentifiedImageError):
            # Client sent something that isn't a decodable image — their
            # fault, not ours.
            raise HTTPException(
                status_code=400, detail="File is not a valid image (JPG, PNG, WEBP)"
            )
        logger.error(f"Avatar processing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process avatar")
    finally:
        temp_orig.unlink(missing_ok=True)
        temp_processed.unlink(missing_ok=True)
        # is_file() guards against the Path("") == "." footgun — never unlink
        # a directory.
        thumb_value = metadata.get("thumbnail_path") or ""
        if thumb_value:
            thumb_file = Path(thumb_value)
            if thumb_file.is_file():
                thumb_file.unlink(missing_ok=True)

    avatar = Avatar(
        id=avatar_id,
        user_id=_user_id(current_user),
        name=name,
        image_url=image_url,
        thumbnail_url=thumbnail_url,
        s3_key=image_key,
        status="ready",
        avatar_metadata=metadata,
    )
    db.add(avatar)
    await db.commit()
    await db.refresh(avatar)

    logger.info(f"Avatar created: {avatar_id} for user {_user_id(current_user)}")
    return avatar


@router.get("/", response_model=List[AvatarResponse])
async def list_avatars(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """List avatars belonging to the current user."""
    uid = _user_id(current_user)
    result = await db.execute(
        select(Avatar)
        .where(Avatar.user_id == uid)
        .offset(skip)
        .limit(limit)
        .order_by(Avatar.created_at.desc())
    )
    return result.scalars().all()


@router.get("/expression-presets")
async def list_expression_presets():
    """
    Named LivePortrait expression presets available via POST
    /{avatar_id}/expressions — see app/services/expression_editor.py and
    model_experiments/comfyui/expression_editor_client.py for what each one
    actually does to the ExpressionEditor node's parameters.

    Registered before /{avatar_id} below — FastAPI matches routes in
    declaration order, and a GET here would otherwise be swallowed by that
    path-param route with avatar_id="expression-presets".
    """
    return {"presets": sorted(expression_editor_service.presets.keys())}


@router.get("/{avatar_id}", response_model=AvatarResponse)
async def get_avatar(
    avatar_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    _validate_uuid(avatar_id)
    result = await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    avatar = result.scalar_one_or_none()
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    if avatar.user_id != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Not authorised to access this avatar")
    return avatar


@router.put("/{avatar_id}/photo", response_model=AvatarResponse)
async def replace_avatar_photo(
    avatar_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Replace an existing avatar's photo in place (same avatar_id, so its
    sessions/history stay attached). Any previously-rendered idle loop no
    longer matches the new photo, so it's cleared here — re-render it via
    POST /{avatar_id}/idle-video when ready.
    """
    _validate_uuid(avatar_id)
    result = await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    avatar = result.scalar_one_or_none()
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    if avatar.user_id != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Not authorised to modify this avatar")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPG, PNG, WEBP)")

    file_data: bytes = await file.read()  # type: ignore[assignment]
    if len(file_data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File must be under 10 MB")

    suffix = Path(file.filename or "avatar.jpg").suffix or ".jpg"
    temp_orig = TMPDIR / f"{avatar_id}_replace_original{suffix}"
    temp_processed = TMPDIR / f"{avatar_id}_replace_processed.jpg"
    metadata: dict = {}

    try:
        temp_orig.write_bytes(file_data)
        _, metadata = await avatar_processor.process_image(str(temp_orig), str(temp_processed))

        image_key = f"avatars/{avatar_id}/image.jpg"
        image_url = await storage_service.upload_file(
            temp_processed.read_bytes(), image_key, content_type="image/jpeg"
        )

        thumb_value = metadata.get("thumbnail_path") or ""
        thumb_file = Path(thumb_value) if thumb_value else None
        thumb_key = f"avatars/{avatar_id}/thumbnail.jpg"
        thumb_bytes = (
            thumb_file.read_bytes()
            if thumb_file and thumb_file.is_file()
            else temp_processed.read_bytes()
        )
        thumbnail_url = await storage_service.upload_file(
            thumb_bytes, thumb_key, content_type="image/jpeg"
        )

    except HTTPException:
        raise
    except Exception as e:
        from PIL import UnidentifiedImageError

        if isinstance(e, UnidentifiedImageError):
            raise HTTPException(
                status_code=400, detail="File is not a valid image (JPG, PNG, WEBP)"
            )
        logger.error(f"Avatar photo replace error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process avatar photo")
    finally:
        temp_orig.unlink(missing_ok=True)
        temp_processed.unlink(missing_ok=True)
        thumb_value = metadata.get("thumbnail_path") or ""
        if thumb_value:
            thumb_file = Path(thumb_value)
            if thumb_file.is_file():
                thumb_file.unlink(missing_ok=True)

    try:
        avatar.image_url = image_url
        avatar.thumbnail_url = thumbnail_url
        avatar.avatar_metadata = metadata
        avatar.idle_video_url = None  # stale — no longer matches the new photo
        await db.commit()
        await db.refresh(avatar)
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to save replaced photo for avatar {avatar_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update avatar photo")

    # Drop the animator's cached local copy of the OLD image (see
    # resolve_local_image) — otherwise chat and idle-video renders would
    # keep animating the photo we just replaced.
    (TMPDIR / "avatars" / f"{avatar_id}.jpg").unlink(missing_ok=True)

    logger.info(f"Avatar {avatar_id} photo replaced")
    return avatar


@router.post("/{avatar_id}/idle-video", response_model=AvatarResponse)
async def generate_idle_video(
    avatar_id: str,
    duration_s: int = Query(60, ge=1, le=120, description="Idle loop length in seconds"),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Render (or re-roll) this avatar's idle loop — played on <video loop>
    client-side whenever nothing is being said, the same way Kindroid
    pre-renders one idle clip per avatar instead of animating idle live.

    Not time-critical, so this can take several minutes on a contended GPU;
    the request simply awaits it (this codebase has no background job queue
    — every other render works the same way, awaited inline).
    """
    _validate_uuid(avatar_id)
    result = await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    avatar = result.scalar_one_or_none()
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    if avatar.user_id != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Not authorised to modify this avatar")

    local_image = await resolve_local_image(avatar.id, avatar.s3_key)
    local_video = TMPDIR / f"{avatar_id}_idle.mp4"
    local_blinked = TMPDIR / f"{avatar_id}_idle_blinked.mp4"

    try:
        await avatar_animator.generate_idle_loop(local_image, str(local_video), duration_s=duration_s)

        # Blink compositing runs on the CPU (OpenCV, no GPU contention) and
        # is engine-agnostic — see blink.py for why neither MuseTalk nor the
        # simple fallback can ever produce blinking on their own.
        final_video = await asyncio.to_thread(
            blink.apply_blinks, str(local_video), local_image, str(local_blinked)
        )

        video_key = f"avatars/{avatar_id}/idle.mp4"
        await storage_service.upload_file(
            Path(final_video).read_bytes(), video_key, content_type="video/mp4"
        )
        idle_video_url = await storage_service.serving_url(video_key)
    except Exception as e:
        logger.error(f"Idle video generation failed for avatar {avatar_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate idle video")
    finally:
        local_video.unlink(missing_ok=True)
        local_blinked.unlink(missing_ok=True)

    try:
        avatar.idle_video_url = idle_video_url
        await db.commit()
        await db.refresh(avatar)
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to save idle video for avatar {avatar_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save idle video")

    logger.info(f"Idle video generated for avatar {avatar_id}")
    return avatar


@router.post("/{avatar_id}/idle-playlist", response_model=AvatarResponse)
async def generate_idle_playlist(
    avatar_id: str,
    num_segments: int = Query(
        MAX_PLAYLIST_SEGMENTS, ge=1, le=MAX_PLAYLIST_SEGMENTS,
        description="How many independent idle segments to render",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Render this avatar's Hallo2 idle playlist — up to MAX_PLAYLIST_SEGMENTS
    independently-generated, real-blinking clips (unlike the MuseTalk-based
    /idle-video above, which can never blink — see blink.py and
    hallo2_animator.py for why). The frontend plays these back in random
    order, looping until a chat turn interrupts it; each segment is
    individually trimmed to loop back near the avatar's own resting pose,
    so any segment can follow any other without a visible cut.

    Each segment measured at 270-330s in this session's own testing, so a
    full 6-segment playlist can take 25-35 minutes — this endpoint awaits
    the whole batch (same "no job queue, just await it" pattern as every
    other render in this app) and returns whatever succeeded even if a
    later segment fails partway through.
    """
    _validate_uuid(avatar_id)
    result = await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    avatar = result.scalar_one_or_none()
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    if avatar.user_id != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Not authorised to modify this avatar")

    # Shared with the auto-trigger on chat-session open (app/websocket.py)
    # so both paths generate/cache/reuse the same way — see
    # app/services/idle_playlist.py.
    if is_generation_in_progress(avatar_id):
        # Someone else (an auto-trigger, or another tab) already has this
        # in flight — not a failure, just return current state rather than
        # kicking off a second redundant Hallo2 batch for the same avatar.
        return avatar

    playlist_urls = await ensure_idle_playlist(avatar_id, num_segments=num_segments)
    if not playlist_urls:
        raise HTTPException(status_code=500, detail="Hallo2 failed to generate any idle segments")

    await db.refresh(avatar)
    logger.info(f"Idle playlist ready for avatar {avatar_id}: {len(playlist_urls)} segments")
    return avatar


@router.post("/{avatar_id}/expressions", response_model=AvatarResponse)
async def generate_expression_photo(
    avatar_id: str,
    body: ExpressionGenerateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Generate one identity-preserving expression-edited still of this avatar
    via LivePortrait and add it to the avatar's expression-photo library
    (avatar.expression_photos). Hallo2 itself has no expression input — see
    hallo2_animator.py's docstring — so this is the only way to give the
    idle-segment rotation more than one resting expression to work from.
    Library entries can later be rendered into a new idle segment via
    POST /{avatar_id}/idle-playlist/from-expression.
    """
    _validate_uuid(avatar_id)
    if body.preset is None and body.params is None:
        raise HTTPException(status_code=400, detail="Provide at least one of preset or params")

    result = await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    avatar = result.scalar_one_or_none()
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    if avatar.user_id != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Not authorised to modify this avatar")

    local_image = await resolve_local_image(avatar.id, avatar.s3_key)
    photo_id = str(uuid.uuid4())
    work_path = TMPDIR / f"{avatar_id}_expr_{photo_id}.png"

    raw_params = body.params.model_dump(exclude_none=True) if body.params else None
    label = body.label or body.preset or "custom"

    try:
        await expression_editor_service.generate(
            local_image, str(work_path), preset=body.preset, params=raw_params
        )
    except ExpressionEditorServiceError as e:
        logger.error(f"Expression generation failed for avatar {avatar_id} ({label}): {e}")
        raise HTTPException(status_code=500, detail=f"Expression generation failed: {e}")

    photo_key = f"avatars/{avatar_id}/expressions/{photo_id}.png"
    await storage_service.upload_file(work_path.read_bytes(), photo_key, content_type="image/png")
    photo_url = await storage_service.serving_url(photo_key)
    work_path.unlink(missing_ok=True)

    entry = {
        "id": photo_id,
        "label": label,
        "key": photo_key,
        "url": photo_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    avatar.expression_photos = list(avatar.expression_photos or []) + [entry]
    await db.commit()
    await db.refresh(avatar)

    logger.info(f"Generated expression photo '{label}' for avatar {avatar_id}: {photo_key}")
    return avatar


@router.delete("/{avatar_id}/expressions/{photo_id}", response_model=AvatarResponse)
async def delete_expression_photo(
    avatar_id: str,
    photo_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Remove one photo from the expression-photo library. Doesn't touch any idle segment already rendered from it."""
    _validate_uuid(avatar_id)
    result = await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    avatar = result.scalar_one_or_none()
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    if avatar.user_id != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Not authorised to modify this avatar")

    photos = avatar.expression_photos or []
    matching = next((p for p in photos if p.get("id") == photo_id), None)
    if not matching:
        raise HTTPException(status_code=404, detail="Expression photo not found on this avatar")

    await storage_service.delete_file(matching["key"])
    avatar.expression_photos = [p for p in photos if p.get("id") != photo_id]
    await db.commit()
    await db.refresh(avatar)
    return avatar


@router.post("/{avatar_id}/idle-playlist/from-expression", response_model=AvatarResponse)
async def render_idle_segment_from_expression(
    avatar_id: str,
    body: IdleSegmentFromExpressionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """
    Render one Hallo2 idle segment using a previously-generated expression
    photo (see POST /{avatar_id}/expressions) as the source image instead
    of the avatar's default photo. `slot_index` replaces that position in
    idle_playlist_urls; omitted, it appends (capped at MAX_PLAYLIST_SEGMENTS).
    """
    _validate_uuid(avatar_id)
    result = await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    avatar = result.scalar_one_or_none()
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    if avatar.user_id != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Not authorised to modify this avatar")

    photos = avatar.expression_photos or []
    matching = next((p for p in photos if p.get("url") == body.photo_url), None)
    if not matching:
        raise HTTPException(status_code=404, detail="Expression photo not found on this avatar")

    existing = list(avatar.idle_playlist_urls or [])
    if body.slot_index is None and len(existing) >= MAX_PLAYLIST_SEGMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Idle playlist already has {MAX_PLAYLIST_SEGMENTS} segments — specify slot_index to replace one",
        )

    cache_path = TMPDIR / f"expr_source_{matching['id']}.png"
    if not cache_path.exists():
        cache_path.write_bytes(await storage_service.download_file(matching["key"]))

    idx = body.slot_index if body.slot_index is not None else len(existing)
    output_path = TMPDIR / f"{avatar_id}_idle_from_expr_{uuid.uuid4().hex[:8]}.mp4"

    try:
        local_result = await hallo2_animator.generate_one_idle_segment(
            str(cache_path), str(output_path), seed=1042 + idx
        )
    except Exception as e:
        logger.error(f"Idle-from-expression render failed for avatar {avatar_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Render failed: {e}")

    video_key = f"avatars/{avatar_id}/idle_playlist/segment_{idx}.mp4"
    await storage_service.upload_file(Path(local_result).read_bytes(), video_key, content_type="video/mp4")
    video_url = await storage_service.serving_url(video_key)
    Path(local_result).unlink(missing_ok=True)

    if body.slot_index is not None and body.slot_index < len(existing):
        existing[body.slot_index] = video_url
    else:
        existing.append(video_url)
    avatar.idle_playlist_urls = existing
    await db.commit()
    await db.refresh(avatar)

    logger.info(f"Rendered idle segment from expression '{matching['label']}' for avatar {avatar_id} at slot {idx}")
    return avatar


@router.put("/{avatar_id}/voice", response_model=AvatarResponse)
async def set_avatar_voice(
    avatar_id: str,
    voice_id: Optional[str] = Query(
        default=None,
        description="Voice profile ID to assign. Omit or pass an empty string to unassign.",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Assign (or clear) a voice profile on an avatar."""
    _validate_uuid(avatar_id)
    result = await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    avatar = result.scalar_one_or_none()
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    if avatar.user_id != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Not authorised to modify this avatar")

    try:
        normalized = (voice_id or "").strip() or None
        avatar.voice_id = normalized
        await db.commit()
        await db.refresh(avatar)
        logger.info(f"Avatar {avatar_id} voice set to: {normalized!r}")
        return avatar
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to set voice for avatar {avatar_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update avatar voice")


@router.patch("/{avatar_id}/metadata", response_model=AvatarResponse)
async def update_avatar_metadata(
    avatar_id: str,
    payload: AvatarMetadataUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Merge an allowlist of metadata fields into avatar_metadata."""
    _validate_uuid(avatar_id)
    result = await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    avatar = result.scalar_one_or_none()
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    if avatar.user_id != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Not authorised to modify this avatar")

    existing: dict = avatar.avatar_metadata or {}
    if isinstance(existing, str):
        import json as _json

        try:
            existing = _json.loads(existing)
        except Exception:
            existing = {}

    # Only merge fields the caller actually set (exclude_unset=True keeps the
    # PATCH semantics — omitted fields are left untouched, not nulled out).
    update_data = payload.model_dump(exclude_unset=True)
    try:
        # Assign a NEW dict — mutating the ORM-held dict in place and
        # re-assigning the same object defeats SQLAlchemy's change detection
        # (old is new → no UPDATE emitted), so the metadata silently never
        # persisted even though the endpoint returned 200.
        avatar.avatar_metadata = {**existing, **update_data}
        await db.commit()
        await db.refresh(avatar)
        logger.info(f"Avatar {avatar_id} metadata updated: {list(update_data.keys())}")
        return avatar
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to update metadata for avatar {avatar_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update avatar metadata")


@router.patch("/{avatar_id}/name", response_model=AvatarResponse)
async def rename_avatar(
    avatar_id: str,
    payload: AvatarRename,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Rename an avatar."""
    _validate_uuid(avatar_id)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    result = await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    avatar = result.scalar_one_or_none()
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    if avatar.user_id != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Not authorised to modify this avatar")
    try:
        avatar.name = name
        await db.commit()
        await db.refresh(avatar)
        logger.info(f"Avatar {avatar_id} renamed to: {name!r}")
        return avatar
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to rename avatar {avatar_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to rename avatar")


@router.delete("/{avatar_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_avatar(
    avatar_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    _validate_uuid(avatar_id)
    result = await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    avatar = result.scalar_one_or_none()
    if not avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")
    if avatar.user_id != _user_id(current_user):
        raise HTTPException(status_code=403, detail="Not authorised to delete this avatar")

    # Delete the DB row first (sessions/messages/conversations cascade), THEN
    # the stored files — if the DB delete fails we haven't orphaned the row by
    # removing its image out from under it.
    try:
        await db.delete(avatar)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to delete avatar {avatar_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete avatar")

    try:
        await storage_service.delete_file(avatar.s3_key)
        await storage_service.delete_file(avatar.s3_key.replace("image.jpg", "thumbnail.jpg"))
    except Exception as e:
        # Row is gone; leftover files are harmless and reaped by the cleanup task.
        logger.warning(f"Could not delete stored files for avatar {avatar_id}: {e}")

    logger.info(f"Avatar deleted: {avatar_id}")
