"""
Health Check Endpoint — Reports system status and model availability.
"""

from fastapi import APIRouter
from app.config import settings
from app.models.registry import registry

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health_check():
    """System health check — returns model load status."""
    active_models = registry.get_active_models()
    model_status = {}

    for key, entry in registry.entries.items():
        model_status[key] = {
            "name": entry.name,
            "active": entry.active,
            "loaded": entry.loaded,
            "eer": entry.eer,
            "weight": round(entry.weight, 4),
            "feature_mode": entry.feature_mode,
        }

    return {
        "status": "healthy" if active_models else "degraded",
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "models_active": len(active_models),
        "models_total": len(registry.entries),
        "models": model_status,
        "audio_config": {
            "sample_rate": settings.SR,
            "clip_seconds": settings.CLIP_SEC,
            "clip_samples": settings.CLIP_LEN,
            "max_frames": settings.MAX_FRAMES,
        },
    }
