"""
VoiceGuard AI — FastAPI Main Application

Loads models at startup, registers API routes, serves the forensic analysis API.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.registry import registry
from app.api.v1.health import router as health_router
from app.api.v1.analyze import router as analyze_router
from app.api.v1.report import router as report_router
from app.api.v1.enroll import router as enroll_router
from app.api.v1.upload import router as upload_router
from app.api.v1.stream import router as stream_router
from app.api.ws.live import router as live_router
from app.api.v1.model_test import router as model_test_router

# ── Logging Setup ─────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("voiceguard")


# ── Application Lifespan ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Load all detection models into memory.
    Shutdown: Clean up resources.
    """
    logger.info("=" * 60)
    logger.info(f"  {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"  Environment: {settings.ENVIRONMENT}")
    logger.info("=" * 60)

    # Load models
    logger.info("Loading detection models...")
    registry.load_all()

    active = registry.get_active_models()
    logger.info(f"Models loaded: {len(active)} active")
    for m in active:
        logger.info(f"  - {m.name} | EER={m.eer}%% | weight={m.weight:.3f}")

    logger.info(f"Audio config: SR={settings.SR}, chunk={settings.CHUNK_SEC}s, max={settings.MAX_AUDIO_DURATION_S}s")
    logger.info(f"Sliding window: {settings.CHUNK_SEC}s chunks, {settings.CHUNK_OVERLAP_SEC}s overlap")
    logger.info(f"WebSocket live analysis: ws://{settings.HOST}:{settings.PORT}/ws/live")
    logger.info(f"Serving on {settings.HOST}:{settings.PORT}")
    logger.info("=" * 60)

    yield  # App runs here

    # Shutdown
    logger.info("Shutting down VoiceGuard AI...")


# ── FastAPI App ───────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=(
        "Audio Forensics API for deepfake voice detection. "
        "Provides ensemble detection with LCNN-MFCC + LCNN-LFCC models, "
        "Grad-CAM visual explanations, biometric analysis, and forensic PDF reports."
    ),
    lifespan=lifespan,
)

# CORS — read allowed origins from env var so production can lock to Vercel domain
# Default: allow all (fine for dev). In production, set:
#   CORS_ORIGINS=https://voiceguardfrontend.vercel.app
_cors_raw = os.environ.get("CORS_ORIGINS", "*")
_origins: list[str] = (
    [o.strip() for o in _cors_raw.split(",") if o.strip()]
    if _cors_raw != "*" else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────

app.include_router(health_router)
app.include_router(analyze_router)
app.include_router(report_router)
app.include_router(enroll_router)
app.include_router(upload_router)
app.include_router(stream_router)
app.include_router(live_router)
app.include_router(model_test_router)  # Hidden: /api/v1/test/model/{1,2,3,4}


# ── Root ──────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "analyze": "/api/v1/analyze (POST — full-length sliding window)",
        "live_stream": "ws://host:8000/ws/live (WebSocket — real-time audio)",
    }
