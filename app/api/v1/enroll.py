"""
Speaker Enrollment Endpoints — Register and manage reference voices.
"""

import os
from pathlib import Path
import tempfile
import logging
from fastapi import APIRouter, File, UploadFile, Form, HTTPException

from app.models.registry import registry
from app.utils.audio import load_audio
from app.xai.enrollment import enroll_speaker, list_enrolled_speakers, delete_enrolled_speaker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Voice Enrollment"])


@router.post("/enroll")
async def enroll_new_speaker(
    audio_file: UploadFile = File(...),
    speaker_name: str = Form(...)
):
    """
    Enroll a reference voice print.
    Extracts a 256-dim embedding using the active LCNN model and saves it.
    """
    if not speaker_name.strip():
        raise HTTPException(status_code=400, detail="Speaker name cannot be empty")
        
    try:
        suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name
            
        # Load and preprocess audio (exactly 2s / 32000 samples)
        audio_np = load_audio(tmp_path)
        
        # Perform enrollment
        result = enroll_speaker(speaker_name, audio_np, registry)
        
        # Clean up
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
            
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
            
        return result
    except Exception as e:
        logger.error(f"Failed to enroll speaker: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enrolled")
async def get_enrolled():
    """List all currently enrolled speakers."""
    return list_enrolled_speakers()


@router.delete("/enrolled/{speaker_id}")
async def delete_speaker(speaker_id: str):
    """Delete an enrolled speaker's voiceprint."""
    success = delete_enrolled_speaker(speaker_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Speaker not found: {speaker_id}")
    return {"status": "success", "message": f"Deleted speaker {speaker_id}"}
