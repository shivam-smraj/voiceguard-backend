"""
WebSocket Live Audio Stream — Real-time deepfake detection.

Protocol:
  1. Client connects to ws://host:8000/ws/live
  2. Client sends audio chunks as binary frames (2 seconds of PCM float32 @ 16kHz)
  3. Server processes each chunk and sends back JSON verdict immediately
  4. Server maintains a rolling window of scores for trend analysis
  5. Client can also send JSON control messages:
     - {"action": "start", "call_id": "...", "sr": 16000}
     - {"action": "stop"}
     - {"action": "status"}

Response format (sent after each chunk):
  {
    "type": "verdict",
    "chunk_idx": 0,
    "score": 87.3,
    "verdict": "FAKE",
    "confidence": "CRITICAL",
    "rolling_avg": 82.1,
    "trend": "rising",
    "alert": true,
    "elapsed_ms": 245,
    "per_model": {...}
  }
"""

import json
import time
import logging
import asyncio
from typing import Optional
from collections import deque

import numpy as np
import torch

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.config import settings
from app.models.registry import registry
from app.features.pipeline import extract_all_features
from app.utils.audit_id import generate_audit_id

logger = logging.getLogger(__name__)

router = APIRouter(tags=["live"])


class LiveSession:
    """Manages state for a single live audio analysis session."""

    def __init__(self, call_id: str = "LIVE"):
        self.call_id = call_id
        self.audit_id = generate_audit_id(call_id)
        self.chunk_idx = 0
        self.scores = deque(maxlen=100)   # Rolling window of last 100 scores
        self.all_scores = []               # All scores ever
        self.start_time = time.time()
        self.alert_sent = False
        self.sr = settings.SR
        self.chunk_samples = settings.CHUNK_SAMPLES

    @property
    def rolling_avg(self) -> float:
        if not self.scores:
            return 0.0
        return float(np.mean(list(self.scores)))

    @property
    def trend(self) -> str:
        """Compute trend from last 5 scores."""
        if len(self.scores) < 3:
            return "stable"
        recent = list(self.scores)[-5:]
        if len(recent) < 2:
            return "stable"
        slope = recent[-1] - recent[0]
        if slope > 5:
            return "rising"
        elif slope < -5:
            return "falling"
        return "stable"

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.start_time

    @property
    def n_fake_chunks(self) -> int:
        return sum(1 for s in self.all_scores if s >= 50.0)


@router.websocket("/ws/live")
async def live_audio_stream(ws: WebSocket):
    """
    WebSocket endpoint for real-time audio analysis.

    Accepts binary audio frames (PCM float32 @ 16kHz, 2-second chunks)
    and returns JSON verdicts after each chunk.
    """
    await ws.accept()
    session: Optional[LiveSession] = None

    logger.info("WebSocket client connected for live analysis")

    try:
        # Send welcome message
        await ws.send_json({
            "type": "connected",
            "message": "VoiceGuard AI Live Analysis ready",
            "expected_format": "PCM float32, 16kHz, 2-second chunks (32000 samples x 4 bytes = 128000 bytes)",
            "models_active": len(registry.get_active_models()),
        })

        while True:
            # Receive message (binary = audio, text = control)
            message = await ws.receive()

            if "bytes" in message and message["bytes"]:
                # ── Binary audio chunk ────────────────────────────────
                raw_bytes = message["bytes"]

                if session is None:
                    session = LiveSession()
                    logger.info(f"[{session.audit_id}] Auto-started live session")

                # Convert bytes to float32 numpy array
                try:
                    audio_chunk = np.frombuffer(raw_bytes, dtype=np.float32)
                except Exception:
                    # Try int16 format
                    try:
                        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                        audio_chunk = audio_int16.astype(np.float32) / 32768.0
                    except Exception as e:
                        await ws.send_json({
                            "type": "error",
                            "message": f"Invalid audio format: {e}",
                        })
                        continue

                # Pad or trim to exact chunk size
                if len(audio_chunk) != session.chunk_samples:
                    if len(audio_chunk) < session.chunk_samples:
                        audio_chunk = np.pad(audio_chunk, (0, session.chunk_samples - len(audio_chunk)))
                    else:
                        audio_chunk = audio_chunk[:session.chunk_samples]

                # Normalize
                peak = np.abs(audio_chunk).max()
                if peak > 1e-6:
                    audio_chunk = audio_chunk / peak

                # ── Process the chunk ─────────────────────────────────
                chunk_start = time.time()
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, _process_chunk, audio_chunk, session.chunk_idx,
                    )
                    chunk_ms = int((time.time() - chunk_start) * 1000)

                    score = result["ensemble_score"]
                    verdict = result["verdict"]
                    confidence = result["confidence_label"]

                    session.scores.append(score)
                    session.all_scores.append(score)

                    # Alert logic
                    should_alert = score >= 70.0 or session.rolling_avg >= 60.0
                    is_first_alert = should_alert and not session.alert_sent
                    if is_first_alert:
                        session.alert_sent = True

                    response = {
                        "type": "verdict",
                        "chunk_idx": session.chunk_idx,
                        "time_s": round(session.chunk_idx * settings.CHUNK_SEC, 1),
                        "score": round(score, 2),
                        "verdict": verdict,
                        "confidence": confidence,
                        "rolling_avg": round(session.rolling_avg, 2),
                        "trend": session.trend,
                        "alert": should_alert,
                        "first_alert": is_first_alert,
                        "elapsed_ms": chunk_ms,
                        "per_model": result.get("per_model", {}),
                        "session": {
                            "audit_id": session.audit_id,
                            "total_chunks": session.chunk_idx + 1,
                            "n_fake": session.n_fake_chunks,
                            "session_duration_s": round(session.elapsed_s, 1),
                        },
                    }

                    await ws.send_json(response)

                    log_level = logging.WARNING if should_alert else logging.INFO
                    logger.log(
                        log_level,
                        f"[{session.audit_id}] Chunk {session.chunk_idx}: "
                        f"{verdict} ({score:.1f}%) | rolling={session.rolling_avg:.1f}% "
                        f"| trend={session.trend} | {chunk_ms}ms"
                    )

                    session.chunk_idx += 1

                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}", exc_info=True)
                    await ws.send_json({
                        "type": "error",
                        "message": f"Processing failed: {str(e)}",
                        "chunk_idx": session.chunk_idx if session else 0,
                    })

            elif "text" in message and message["text"]:
                # ── Control message ───────────────────────────────────
                try:
                    ctrl = json.loads(message["text"])
                    action = ctrl.get("action", "")

                    if action == "start":
                        call_id = ctrl.get("call_id", "LIVE")
                        session = LiveSession(call_id=call_id)
                        if ctrl.get("sr"):
                            session.sr = ctrl["sr"]
                        logger.info(f"[{session.audit_id}] Live session started")
                        await ws.send_json({
                            "type": "session_started",
                            "audit_id": session.audit_id,
                            "call_id": call_id,
                            "sr": session.sr,
                            "chunk_size_bytes": session.chunk_samples * 4,
                        })

                    elif action == "stop":
                        if session:
                            summary = _build_session_summary(session)
                            await ws.send_json(summary)
                            logger.info(
                                f"[{session.audit_id}] Session stopped: "
                                f"{session.chunk_idx} chunks"
                            )
                            session = None
                        else:
                            await ws.send_json({
                                "type": "error", "message": "No active session",
                            })

                    elif action == "status":
                        if session:
                            await ws.send_json({
                                "type": "status",
                                "audit_id": session.audit_id,
                                "chunks_processed": session.chunk_idx,
                                "rolling_avg": round(session.rolling_avg, 2),
                                "trend": session.trend,
                                "n_fake": session.n_fake_chunks,
                                "session_duration_s": round(session.elapsed_s, 1),
                            })
                        else:
                            await ws.send_json({
                                "type": "status", "message": "No active session",
                            })

                    else:
                        await ws.send_json({
                            "type": "error",
                            "message": f"Unknown action: {action}",
                        })

                except json.JSONDecodeError:
                    await ws.send_json({
                        "type": "error", "message": "Invalid JSON",
                    })

    except WebSocketDisconnect:
        if session:
            logger.info(
                f"[{session.audit_id}] WebSocket disconnected after "
                f"{session.chunk_idx} chunks ({session.elapsed_s:.1f}s)"
            )
        else:
            logger.info("WebSocket client disconnected (no session)")

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


def _process_chunk(audio_np: np.ndarray, chunk_idx: int) -> dict:
    """Process a single audio chunk (sync, for executor)."""
    features = extract_all_features(audio_np)
    result = registry.ensemble_predict(features)
    return result


def _build_session_summary(session: LiveSession) -> dict:
    """Build end-of-session summary."""
    scores = np.array(session.all_scores) if session.all_scores else np.array([0.0])

    if len(scores) > 0:
        max_score = float(np.max(scores))
        mean_score = float(np.mean(scores))
        k = max(1, len(scores) // 10)
        top_k = np.sort(scores)[-k:]
        top_mean = float(np.mean(top_k))
        overall = 0.6 * max_score + 0.3 * top_mean + 0.1 * mean_score
    else:
        overall = 0.0

    return {
        "type": "session_summary",
        "audit_id": session.audit_id,
        "call_id": session.call_id,
        "total_chunks": session.chunk_idx,
        "total_duration_s": round(session.chunk_idx * settings.CHUNK_SEC, 1),
        "session_elapsed_s": round(session.elapsed_s, 1),
        "overall_score": round(overall, 2),
        "overall_verdict": "FAKE" if overall >= 50.0 else "REAL",
        "n_fake_chunks": session.n_fake_chunks,
        "n_real_chunks": session.chunk_idx - session.n_fake_chunks,
        "fake_pct": round(session.n_fake_chunks / max(1, session.chunk_idx) * 100, 1),
        "max_score": round(float(scores.max()), 2),
        "min_score": round(float(scores.min()), 2),
        "mean_score": round(float(scores.mean()), 2),
        "all_scores": [round(float(s), 2) for s in session.all_scores],
    }
