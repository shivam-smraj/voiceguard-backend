"""
Analyze Endpoint — Core audio forensics API.
POST /api/v1/analyze — Upload audio, get verdict + XAI evidence.

Now supports FULL-LENGTH audio via sliding window analysis:
  - Audio is split into 2-second overlapping chunks
  - Each chunk is scored independently by the ensemble
  - Per-chunk scores create a temporal heatmap
  - Final verdict = aggregation of all chunk scores
  - XAI runs on the most suspicious chunk
"""

import os
import time
import hashlib
import tempfile
import logging
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.registry import registry
from app.features.pipeline import extract_all_features
from app.utils.audio import load_audio, load_audio_full, load_audio_chunks, get_audio_info
from app.utils.audit_id import generate_audit_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/analyze")
async def analyze_audio(
    audio_file: UploadFile = File(...),
    call_id: str = Form(default="DEMO-CALL"),
    customer_id: str = Form(default=""),
    branch_code: str = Form(default=""),
    run_xai: bool = Form(default=True),
):
    """
    Full forensic analysis pipeline with sliding window support.

    Pipeline:
      1. Validate + decode audio
      2. Split into 2-second chunks (overlapping)
      3. Run ensemble detection on EVERY chunk
      4. Aggregate: overall verdict + temporal heatmap
      5. Run XAI engine on the most suspicious chunk
      6. Generate multi-page forensic PDF

    Returns JSON with:
      - verdict, ensemble_score (overall)
      - temporal_scores: per-chunk scores (temporal heatmap)
      - xai: full XAI evidence from the worst chunk
      - pdf_url: forensic report
    """
    start_time = time.time()
    audit_id = generate_audit_id(call_id)

    logger.info(f"[{audit_id}] Starting full-length analysis for call_id={call_id}")

    # ── Step 1: Save and validate audio ──────────────────────────────
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

        logger.info(
            f"[{audit_id}] Audio loaded: {audio_info['duration_s']:.1f}s, "
            f"{audio_info['sample_rate']}Hz, {file_size_mb:.1f}MB"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {str(e)}")

    # ── Step 2: Split into chunks ────────────────────────────────────
    try:
        detection_start = time.time()

        chunks = load_audio_chunks(tmp_path)
        n_chunks = len(chunks)

        # Also load full audio for XAI biometrics
        audio_full = load_audio_full(tmp_path)

        logger.info(
            f"[{audit_id}] Audio split into {n_chunks} chunks "
            f"({settings.CHUNK_SEC}s each, {settings.CHUNK_OVERLAP_SEC}s overlap)"
        )

    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Audio chunking failed: {str(e)}")

    # ── Step 3: Run detection on EVERY chunk ─────────────────────────
    try:
        chunk_results = []
        all_spoof_scores = []

        for chunk in chunks:
            # Extract features for this chunk
            features = extract_all_features(chunk["audio_np"])

            # Run ensemble
            result = registry.ensemble_predict(features)

            chunk_results.append({
                "chunk_idx": chunk["chunk_idx"],
                "start_s": chunk["start_s"],
                "end_s": chunk["end_s"],
                "ensemble_score": result["ensemble_score"],
                "verdict": result["verdict"],
                "per_model": result["per_model"],
            })
            all_spoof_scores.append(result["ensemble_score"])

        detection_ms = int((time.time() - detection_start) * 1000)

        # ── Aggregate scores ─────────────────────────────────────────
        scores_arr = np.array(all_spoof_scores)

        # Overall score = weighted mean favoring high-scoring chunks
        # This ensures a single 95% fake chunk in 50 real chunks still triggers
        overall_score = float(_aggregate_scores(scores_arr))
        overall_verdict = "FAKE" if overall_score >= 50.0 else "REAL"

        # Find the worst (most suspicious) chunk
        worst_idx = int(np.argmax(scores_arr))
        best_idx = int(np.argmin(scores_arr))

        # Confidence label
        confidence_label = _get_confidence_label(overall_score)

        # Temporal statistics
        n_fake_chunks = int(np.sum(scores_arr >= 50.0))
        n_real_chunks = n_chunks - n_fake_chunks
        fake_pct = round(n_fake_chunks / max(1, n_chunks) * 100, 1)

        logger.info(
            f"[{audit_id}] Detection complete: {overall_verdict} ({overall_score:.1f}%) "
            f"[{confidence_label}] | {n_fake_chunks}/{n_chunks} chunks FAKE "
            f"| worst={scores_arr[worst_idx]:.1f}% at chunk {worst_idx} "
            f"| {detection_ms}ms"
        )

        # Build ensemble_result for backward compat
        ensemble_result = {
            "ensemble_score": round(overall_score, 2),
            "verdict": overall_verdict,
            "confidence_label": confidence_label,
            "per_model": chunk_results[worst_idx]["per_model"],
        }

    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    # ── Step 4: XAI Engine on the worst chunk ────────────────────────
    xai_result = None
    xai_ms = 0

    if run_xai:
        xai_start = time.time()
        try:
            # Extract features from the worst chunk for XAI
            worst_chunk = chunks[worst_idx]
            worst_features = extract_all_features(worst_chunk["audio_np"])

            from app.xai.engine import run_xai_engine
            xai_result = await run_xai_engine(
                audio_np=worst_chunk["audio_np"],
                audio_full=audio_full,
                features=worst_features,
                registry=registry,
                audit_id=audit_id,
                verdict=overall_verdict,
                ensemble_score=overall_score,
                confidence_label=confidence_label,
            )

            # Inject temporal context into XAI results
            if xai_result:
                xai_result["worst_chunk_idx"] = worst_idx
                xai_result["worst_chunk_time"] = f"{worst_chunk['start_s']:.1f}-{worst_chunk['end_s']:.1f}s"

            xai_ms = int((time.time() - xai_start) * 1000)
            logger.info(f"[{audit_id}] XAI engine completed in {xai_ms}ms")
        except Exception as e:
            logger.error(f"[{audit_id}] XAI engine failed: {e}", exc_info=True)
            xai_ms = int((time.time() - xai_start) * 1000)

    # ── Step 5: Build response ───────────────────────────────────────
    total_ms = int((time.time() - start_time) * 1000)

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    response = {
        "audit_id": audit_id,
        "call_id": call_id,
        "verdict": overall_verdict,
        "ensemble_score": round(overall_score, 2),
        "confidence_label": confidence_label,
        "per_model": ensemble_result["per_model"],

        # ── Full-length temporal analysis (NEW) ──────────────────────
        "temporal_analysis": {
            "n_chunks": n_chunks,
            "chunk_duration_s": settings.CHUNK_SEC,
            "overlap_s": settings.CHUNK_OVERLAP_SEC,
            "total_duration_s": round(audio_info["duration_s"], 1),
            "processing_sr": settings.SR,
            "original_sr": audio_info["sample_rate"],

            # Per-chunk scores (the temporal heatmap)
            "chunk_scores": [
                {
                    "idx": cr["chunk_idx"],
                    "start_s": cr["start_s"],
                    "end_s": cr["end_s"],
                    "score": cr["ensemble_score"],
                    "verdict": cr["verdict"],
                }
                for cr in chunk_results
            ],

            # Summary stats
            "n_fake_chunks": n_fake_chunks,
            "n_real_chunks": n_real_chunks,
            "fake_chunk_pct": fake_pct,
            "max_score": round(float(scores_arr.max()), 2),
            "min_score": round(float(scores_arr.min()), 2),
            "mean_score": round(float(scores_arr.mean()), 2),
            "std_score": round(float(scores_arr.std()), 2),
            "worst_chunk_idx": worst_idx,
            "worst_chunk_time": f"{chunks[worst_idx]['start_s']:.1f}-{chunks[worst_idx]['end_s']:.1f}s",
        },

        "audio_info": {
            "duration_s": audio_info["duration_s"],
            "sample_rate": audio_info["sample_rate"],
            "processing_sr": settings.SR,
            "audio_hash": audio_hash,
            "file_size_mb": round(file_size_mb, 2),
        },
        "xai": xai_result,
        "performance": {
            "total_ms": total_ms,
            "detection_ms": detection_ms,
            "xai_ms": xai_ms,
            "chunks_processed": n_chunks,
            "ms_per_chunk": round(detection_ms / max(1, n_chunks)),
        },
    }

    # ── Step 6: Generate PDF Report ──────────────────────────────────
    try:
        from app.report.pdf_generator import generate_forensic_pdf
        pdf_path = str(settings.PDF_OUTPUT_DIR / f"{audit_id}.pdf")
        generate_forensic_pdf(response, output_path=pdf_path)
        response["pdf_url"] = f"/api/v1/report/{audit_id}/pdf"
        logger.info(f"[{audit_id}] PDF generated: {pdf_path}")
    except Exception as e:
        logger.error(f"[{audit_id}] PDF generation failed: {e}", exc_info=True)
        response["pdf_url"] = None

    logger.info(
        f"[{audit_id}] Analysis complete: {overall_verdict} "
        f"({overall_score:.1f}%) | {n_chunks} chunks | {total_ms}ms total"
    )

    return JSONResponse(content=json.loads(json.dumps(response, default=_numpy_serializer)))


# ── Score Aggregation ─────────────────────────────────────────────────

def _aggregate_scores(scores: np.ndarray) -> float:
    """
    Aggregate per-chunk scores into a single overall score.

    Strategy: Use a mix of max and mean that favors detection.
    If ANY chunk is strongly fake, the overall score should be high.

    Formula: 0.6 * max + 0.3 * top_10_pct_mean + 0.1 * overall_mean
    This ensures a single highly suspicious chunk dominates.
    """
    if len(scores) == 0:
        return 0.0

    if len(scores) == 1:
        return float(scores[0])

    max_score = float(np.max(scores))
    mean_score = float(np.mean(scores))

    # Top 10% mean (or at least top 1)
    k = max(1, len(scores) // 10)
    top_k = np.sort(scores)[-k:]
    top_mean = float(np.mean(top_k))

    # Weighted aggregation: heavy on max, moderate on top%, light on mean
    overall = 0.6 * max_score + 0.3 * top_mean + 0.1 * mean_score
    return round(overall, 2)


def _get_confidence_label(score: float) -> str:
    """Map ensemble score to confidence label."""
    if score >= 85 or score <= 15:
        return "CRITICAL"
    elif score >= 70 or score <= 30:
        return "HIGH"
    elif score >= 55 or score <= 45:
        return "MEDIUM"
    else:
        return "LOW"


def _numpy_serializer(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
