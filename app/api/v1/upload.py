"""
Upload Endpoint — Step 1 of the streaming analysis flow.

POST /api/v1/upload
  Accepts audio file, validates it, saves to temp storage.
  Returns audit_id immediately — no processing yet.
  Client then opens EventSource to GET /api/v1/stream/{audit_id}

This decouples file upload from analysis so the frontend can
open the SSE stream and start receiving events without waiting.
"""

import os
import time
import hashlib
import tempfile
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.utils.audio import get_audio_info
from app.utils.audit_id import generate_audit_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["streaming"])

# In-memory store: audit_id → pending analysis metadata
# In production this would be Redis; for now a simple dict suffices.
_pending: dict = {}


def get_pending(audit_id: str) -> dict | None:
    """Return pending analysis info by audit_id."""
    return _pending.get(audit_id)


def clear_pending(audit_id: str):
    """Remove pending entry after streaming completes."""
    _pending.pop(audit_id, None)


@router.post("/upload")
async def upload_audio(
    audio_file: UploadFile = File(...),
    call_id: str = Form(default="DEMO-CALL"),
    customer_id: str = Form(default=""),
    branch_code: str = Form(default=""),
):
    """
    Step 1: Receive audio file and return audit_id immediately.

    The client then opens an SSE stream to /api/v1/stream/{audit_id}
    to receive results progressively as the analysis runs.
    """
    upload_start = time.time()
    audit_id = generate_audit_id(call_id)

    logger.info(f"[{audit_id}] File upload received: {audio_file.filename}")

    # ── Validate + save file ──────────────────────────────────────────
    try:
        suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            dir=str(settings.AUDIO_TEMP_DIR),
        ) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name

        file_size_mb = len(content) / (1024 * 1024)

        if file_size_mb > settings.MAX_AUDIO_SIZE_MB:
            os.unlink(tmp_path)
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size_mb:.1f}MB (max {settings.MAX_AUDIO_SIZE_MB}MB)"
            )

        audio_hash = hashlib.sha256(content).hexdigest()
        audio_info = get_audio_info(tmp_path)

        if audio_info["duration_s"] > settings.MAX_AUDIO_DURATION_S:
            os.unlink(tmp_path)
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long: {audio_info['duration_s']:.0f}s (max {settings.MAX_AUDIO_DURATION_S}s)"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process upload: {str(e)}")

    # ── Store metadata for streaming endpoint ─────────────────────────
    _pending[audit_id] = {
        "audit_id": audit_id,
        "tmp_path": tmp_path,
        "call_id": call_id,
        "customer_id": customer_id,
        "branch_code": branch_code,
        "filename": audio_file.filename or "audio",
        "file_size_mb": round(file_size_mb, 2),
        "audio_hash": audio_hash,
        "audio_info": audio_info,
        "upload_time": time.time(),
    }

    upload_ms = int((time.time() - upload_start) * 1000)
    logger.info(
        f"[{audit_id}] Upload ready: {audio_info['duration_s']:.1f}s, "
        f"{audio_info['sample_rate']}Hz, {file_size_mb:.1f}MB in {upload_ms}ms"
    )

    return JSONResponse(content={
        "audit_id": audit_id,
        "status": "queued",
        "filename": audio_file.filename,
        "duration_s": audio_info["duration_s"],
        "file_size_mb": round(file_size_mb, 2),
        "stream_url": f"/api/v1/stream/{audit_id}",
        "upload_ms": upload_ms,
    })
