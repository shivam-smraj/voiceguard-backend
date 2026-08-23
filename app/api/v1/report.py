"""
Report Endpoint — Serve forensic PDF reports.
GET /api/v1/report/{audit_id}/pdf — Download PDF
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["reports"])


@router.get("/report/{audit_id}/pdf")
async def download_pdf(audit_id: str):
    """Download the forensic PDF report for a given audit ID."""
    pdf_path = settings.PDF_OUTPUT_DIR / f"{audit_id}.pdf"

    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF report not found for audit ID: {audit_id}"
        )

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"VoiceGuard_Forensic_{audit_id}.pdf",
    )
