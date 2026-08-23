"""
Hidden Model Testing Routes — SSE Streaming (low-latency, parallel)
=====================================================================

Architecture:
  • Each route returns a StreamingResponse (text/event-stream / SSE)
  • ALL chunks for a model are submitted to a shared ThreadPoolExecutor simultaneously
  • Each chunk result is pushed to the browser the INSTANT it finishes
  • First result appears in ~0.1-0.3s; all chunks done in ~1-4s
  • XAI (Grad-CAM / CQT heatmap) is sent at the end in a `done` event

SSE Event types:
  header  → { type, total_chunks, model_key, window_sec }
  chunk   → { type, chunk_id, start_s, end_s, spoof_prob, verdict }
  xai     → { type, gradcam_b64 / cqt_heatmap_b64, ... }  (streamed separately)
  done    → { type, avg_score, max_score, verdict, confidence, elapsed_ms, ... }
  error   → { type, message }

Route mapping:
  POST /api/v1/test/model/1  → LCNN-MFCC   (2s chunks, MFCC Grad-CAM)
  POST /api/v1/test/model/2  → LCNN-LFCC   (2s chunks, LFCC Grad-CAM)
  POST /api/v1/test/model/3  → AASIST-L    (2s chunks, no XAI)
  POST /api/v1/test/model/4  → DualResNet  (4s windows, CQT heatmap)
"""

import asyncio
import functools
import json
import logging
import threading
import time
import numpy as np
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from app.models.registry import registry
from app.config import settings

# ── Import DualResNet feature module at TOP LEVEL ──────────────────────────────
# CRITICAL: importing lazily (inside function bodies) caused Python’s module-import
# lock to be held by the warmup background thread while real HTTP requests waited
# up to 20s. Importing here at module load time eliminates that contention.
from app.features.dual_resnet_features import (
    compute_full_audio_features,
    infer_window_fast,
    chunk_audio_4s,
    DR_SR,
    _CQT_FREQS,
)
from app.utils.augmentor import run_inference_augmentation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/test", tags=["model-test"])

# ── Shared thread pool ───────────────────────────────────────────────────────
# 8 workers: 4 models × 2 concurrent chunks each = good CPU saturation
_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="model_test")


# ── Speed caps (keep interactive) ───────────────────────────────────────────
MAX_TEST_CHUNKS_2S  = 30   # 2s models: covers first ~45s of audio
MAX_TEST_WINDOWS_4S = 6    # DualResNet: 6 windows → first result in ~1-2s

# ── DualResNet ONNX session cache ────────────────────────────────────────────
# Loaded ONCE at first request, reused for all subsequent requests.
# This eliminates the 1-3s cold-start on every upload.
_DR_ONNX_SESSION: object = None
_DR_ONNX_LOCK = threading.Lock()


def _get_dr_session():
    """Return (or lazily load) the cached ONNX inference session."""
    global _DR_ONNX_SESSION
    if _DR_ONNX_SESSION is not None:
        return _DR_ONNX_SESSION
    with _DR_ONNX_LOCK:
        if _DR_ONNX_SESSION is not None:   # double-checked locking
            return _DR_ONNX_SESSION
        import onnxruntime as ort
        from pathlib import Path
        onnx_path = (
            Path(settings.MODELS_DIR)
            / "DUAL_BRANCH_RESNET"
            / "dual_branch_resnet_simplified.onnx"
        )
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads        = 4   # threads inside each op
        opts.inter_op_num_threads        = 2   # parallel ops
        opts.graph_optimization_level    = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern          = True
        opts.enable_cpu_mem_arena        = True
        _DR_ONNX_SESSION = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        logger.info("[DualResNet] ONNX session loaded and cached ✓")
    return _DR_ONNX_SESSION


def _warmup_dr_session():
    """
    Pre-warm the ONNX session AND feature extraction pipeline at server startup.

    Without this warm-up, the FIRST real request pays:
      - scipy STFT cold-start:        ~5-10s (first scipy import & JIT)
      - torchaudio LFCC cold-start:   ~1-3s  (first kernel compile)
    After this warm-up, all requests run in < 200ms for feature extraction.
    """
    try:
        # 1. Load + dummy-run ONNX session
        session = _get_dr_session()
        dummy_lfcc = np.zeros((1, 3, 20, 400), dtype=np.float32)
        dummy_cqt  = np.zeros((1, 1, 84, 125), dtype=np.float32)
        session.run(None, {"lfcc": dummy_lfcc, "cqt": dummy_cqt})
        logger.info("[DualResNet] ONNX warm-up complete ✓")

        # 2. Warm up CQT (scipy STFT + filterbank) and LFCC (torchaudio)
        #    Use 4s dummy audio so both branches run through their full code path.
        dummy_audio = np.zeros(64000, dtype=np.float32)   # 4s silence
        compute_full_audio_features(dummy_audio, DR_SR)
        logger.info("[DualResNet] CQT+LFCC feature warm-up complete ✓")
    except Exception as e:
        logger.warning(f"[DualResNet] warm-up failed (non-fatal): {e}")



# ── Helpers ───────────────────────────────────────────────────────────────────

async def _read_audio(file: UploadFile) -> np.ndarray:
    """Read uploaded file → float32 numpy array at 16 kHz."""
    suffix = os.path.splitext(file.filename or ".wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        from app.utils.audio import load_audio_full
        audio_np = load_audio_full(tmp_path, sr=settings.SR)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return audio_np


def _confidence_label(p: float) -> str:
    if p >= 85 or p <= 15: return "CRITICAL"
    if p >= 70 or p <= 30: return "HIGH"
    if p >= 55 or p <= 45: return "MEDIUM"
    return "LOW"


def _sse(payload: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


def _make_2s_chunks(audio_np: np.ndarray) -> list:
    sr           = settings.SR
    chunk_samp   = int(settings.CHUNK_SEC * sr)
    hop_samp     = int((settings.CHUNK_SEC - settings.CHUNK_OVERLAP_SEC) * sr)
    if hop_samp <= 0:
        hop_samp = chunk_samp

    chunks = []
    for i, start in enumerate(range(0, len(audio_np), hop_samp)):
        if i >= MAX_TEST_CHUNKS_2S:
            break
        chunk = audio_np[start: start + chunk_samp]
        if len(chunk) < chunk_samp:
            chunk = np.pad(chunk, (0, chunk_samp - len(chunk)))
        chunks.append({
            "audio":   chunk.astype(np.float32),
            "start_s": round(start / sr, 2),
            "end_s":   round(min(start + chunk_samp, len(audio_np)) / sr, 2),
        })
    return chunks


# ── Synchronous chunk processors (run in thread pool) ────────────────────────

def _infer_2s_chunk(chunk: dict, idx: int, model_key: str) -> dict:
    """Process one 2-second chunk. Runs in a thread."""
    from app.features.pipeline import extract_all_features

    entry = registry.entries.get(model_key)
    if not entry or not entry.loaded:
        raise RuntimeError(f"Model {model_key} not loaded")

    feats = extract_all_features(chunk["audio"])

    if entry.feature_mode == "mfcc":
        tensor_in = feats["mfcc"]
    elif entry.feature_mode == "lfcc":
        tensor_in = feats["lfcc"]
    else:
        tensor_in = feats["raw_waveform"]

    pred = registry.predict_single(entry, tensor_in)
    prob = pred["spoof_probability"]

    result = {
        "type":       "chunk",
        "chunk_id":   idx,
        "start_s":    chunk["start_s"],
        "end_s":      chunk["end_s"],
        "spoof_prob": round(prob * 100, 2),
        "verdict":    "FAKE" if prob >= 0.5 else "REAL",
    }

    # Attach raw features for XAI (not serialised — used in memory only)
    result["_feats"] = feats
    return result



def _compute_gradcam(model_key: str, feats: dict, chunk_id: int) -> str:
    """Compute Grad-CAM and render heatmap. Returns base64 PNG or ''."""
    try:
        from app.xai.gradcam import generate_gradcam_for_model
        from app.xai.heatmap_renderer import render_heatmap_trio

        entry = registry.entries[model_key]
        if model_key == "lcnn_mfcc":
            heatmap = generate_gradcam_for_model(entry.model, feats["mfcc"])
            return render_heatmap_trio(
                feats["mfcc_raw"].numpy(), heatmap,
                title=f"MFCC Grad-CAM — chunk {chunk_id} (worst)",
                evidence_type="gradcam_mfcc",
            )
        else:
            heatmap = generate_gradcam_for_model(entry.model, feats["lfcc"])
            return render_heatmap_trio(
                feats["lfcc_raw"].numpy(), heatmap,
                title=f"LFCC Grad-CAM — chunk {chunk_id} (worst)",
                evidence_type="gradcam_lfcc",
            )
    except Exception as e:
        logger.warning(f"Grad-CAM failed ({model_key}): {e}")
        return ""


def _compute_cqt_xai(feats: dict, score_pct: float) -> tuple:
    """Compute CQT heatmap + harmonic metrics. Returns (b64_png, metrics_dict)."""
    try:
        from app.xai.cqt_viz import render_cqt_heatmap, compute_harmonic_metrics
        cqt_raw   = feats["cqt_raw"]
        cqt_freqs = feats["cqt_freqs"]
        heatmap   = render_cqt_heatmap(
            cqt_raw, cqt_freqs,
            title=f"CQT Evidence Map — {score_pct:.1f}% FAKE",
        )
        metrics = compute_harmonic_metrics(cqt_raw, cqt_freqs)
        return heatmap, metrics
    except Exception as e:
        logger.warning(f"CQT XAI failed: {e}")
        return "", {}


# ── SSE generator for 2-second models (M1/M2/M3) ────────────────────────────

async def _stream_2s(
    audio_np:  np.ndarray,
    model_key: str,
    do_gradcam: bool = False,
) -> AsyncGenerator[str, None]:
    """Stream per-chunk results immediately as they complete."""

    chunks = _make_2s_chunks(audio_np)
    total  = len(chunks)
    loop   = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    # Send header so frontend knows total chunk count immediately
    yield _sse({"type": "header", "total_chunks": total,
                "model_key": model_key, "window_sec": settings.CHUNK_SEC})

    t0 = time.time()
    results: dict[int, dict] = {}

    # ── Launch ALL chunks simultaneously into thread pool ──────────────
    async def _run_one(chunk: dict, idx: int):
        try:
            res = await loop.run_in_executor(
                _EXECUTOR, _infer_2s_chunk, chunk, idx, model_key
            )
        except Exception as exc:
            res = {
                "type": "chunk", "chunk_id": idx,
                "start_s": chunk["start_s"], "end_s": chunk["end_s"],
                "spoof_prob": 0.0, "verdict": "ERROR",
                "_feats": None,
            }
            logger.warning(f"[{model_key}] chunk {idx} error: {exc}")
        await queue.put(res)

    tasks = [asyncio.create_task(_run_one(c, i)) for i, c in enumerate(chunks)]

    # ── Stream results as they arrive ─────────────────────────────────
    worst_score = -1.0
    worst_feats = None
    worst_idx   = 0

    for _ in range(total):
        res = await queue.get()
        cid = res["chunk_id"]
        results[cid] = res

        if res.get("_feats") and res["spoof_prob"] > worst_score:
            worst_score = res["spoof_prob"]
            worst_feats = res["_feats"]
            worst_idx   = cid

        # Send chunk without internal _feats key
        wire = {k: v for k, v in res.items() if not k.startswith("_")}
        yield _sse(wire)

    await asyncio.gather(*tasks, return_exceptions=True)
    elapsed_ms = int((time.time() - t0) * 1000)

    scores    = [r["spoof_prob"] for r in results.values()]
    avg_score = float(np.mean(scores)) if scores else 0.0
    max_score = float(np.max(scores))  if scores else 0.0

    # ── Compute and stream XAI (Grad-CAM) ─────────────────────────────
    if do_gradcam and worst_feats is not None:
        gradcam_b64 = await loop.run_in_executor(
            _EXECUTOR,
            functools.partial(_compute_gradcam, model_key, worst_feats, worst_idx)
        )
        yield _sse({
            "type":        "xai",
            "xai_type":    "gradcam",
            "gradcam_b64": gradcam_b64,
            "worst_chunk": worst_idx,
            "description": (
                "Grad-CAM on final ResNet block — shows which feature coefficients "
                "× time frames drove the FAKE verdict in the worst-scoring chunk."
            ),
        })

    # ── Send summary done event ────────────────────────────────────────
    yield _sse({
        "type":        "done",
        "avg_score":   round(avg_score, 2),
        "max_score":   round(max_score, 2),
        "verdict":     "FAKE" if avg_score >= 50 else "REAL",
        "confidence":  _confidence_label(avg_score),
        "chunk_count": total,
        "elapsed_ms":  elapsed_ms,
    })


# ── SSE generator for DualResNet (4-second windows) ──────────────────────────

async def _stream_4s(audio_np: np.ndarray) -> AsyncGenerator[str, None]:
    """
    FAST PATH: Stream per-window results for DualBranchResNet18.

    Architecture:
      1. Compute CQT + LFCC ONCE on the full audio (parallel threads, ~1-2s)
      2. Slice each window from pre-computed features (microseconds per window)
      3. Run ONNX inference per window in parallel (~70ms each)
      4. Stream results as they finish

    For 10s audio (4 windows):
      Before: 4 × ~10s = 40s
      After:  ~1.5s CQT + ~0.3s LFCC + 4 × 0.07s ONNX ≈ 2s  ✓
    """
    from pathlib import Path

    # Validate ONNX exists
    onnx_path = (
        Path(settings.MODELS_DIR)
        / "DUAL_BRANCH_RESNET"
        / "dual_branch_resnet_simplified.onnx"
    )
    if not onnx_path.exists():
        yield _sse({"type": "error", "message": f"ONNX not found: {onnx_path}"})
        return

    # Ensure ONNX session is cached (instant if already loaded)
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(_EXECUTOR, _get_dr_session)
    except Exception as exc:
        yield _sse({"type": "error", "message": f"ONNX session failed: {exc}"})
        return

    # ── Build window list ──────────────────────────────────────────────
    all_windows = chunk_audio_4s(audio_np, sr=DR_SR, stride_s=2.0)
    if len(all_windows) > MAX_TEST_WINDOWS_4S:
        step    = len(all_windows) / MAX_TEST_WINDOWS_4S
        indices = sorted(set(int(i * step) for i in range(MAX_TEST_WINDOWS_4S)))
        windows = [all_windows[i] for i in indices]
        sampled = True
    else:
        windows = all_windows
        sampled = False

    total = len(windows)
    yield _sse({
        "type":          "header",
        "total_chunks":  total,
        "total_windows": len(all_windows),
        "sampled":       sampled,
        "model_key":     "dual_resnet",
        "window_sec":    4,
    })

    t0 = time.time()

    # ── STEP 1: Compute full-audio CQT + LFCC ONCE (the expensive step) ──
    # This replaces N × per-window CQT calls with a single call.
    try:
        full_feats = await loop.run_in_executor(
            _EXECUTOR,
            functools.partial(compute_full_audio_features, audio_np, DR_SR),
        )
    except Exception as exc:
        yield _sse({"type": "error", "message": f"Feature extraction failed: {exc}"})
        return

    cqt_full  = full_feats["cqt_full"]   # (84, T_full)
    lfcc_full = full_feats["lfcc_full"]  # (3, 20, T_full)
    t_feat = time.time() - t0
    logger.info(f"[DualResNet] full-audio features in {t_feat*1000:.0f}ms "
                f"| {total} windows to infer")

    # ── STEP 2 + 3: Slice + ONNX inference per window (all parallel) ──
    queue: asyncio.Queue = asyncio.Queue()
    results: dict[int, dict] = {}
    session = _get_dr_session()

    async def _run_one(win: dict):
        try:
            res = await loop.run_in_executor(
                _EXECUTOR,
                functools.partial(
                    infer_window_fast, win, cqt_full, lfcc_full, session
                ),
            )
        except Exception as exc:
            res = {
                "type": "chunk", "chunk_id": win["chunk_id"],
                "start_s": win["start_s"], "end_s": win["end_s"],
                "spoof_prob": 0.0, "verdict": "ERROR",
                "_cqt_raw": None, "_lfcc_raw": None,
            }
            logger.warning(f"[model/4] window {win['chunk_id']} error: {exc}")
        await queue.put(res)

    tasks = [asyncio.create_task(_run_one(w)) for w in windows]

    worst_score  = -1.0
    worst_cqt_raw = None

    for _ in range(total):
        res = await queue.get()
        cid = res["chunk_id"]
        results[cid] = res

        if res.get("_cqt_raw") is not None and res["spoof_prob"] > worst_score:
            worst_score   = res["spoof_prob"]
            worst_cqt_raw = res["_cqt_raw"]  # (84, 125)

        wire = {k: v for k, v in res.items() if not k.startswith("_")}
        yield _sse(wire)

    await asyncio.gather(*tasks, return_exceptions=True)
    elapsed_ms = int((time.time() - t0) * 1000)

    valid     = [r["spoof_prob"] for r in results.values() if r["verdict"] != "ERROR"]
    avg_score = float(np.mean(valid)) if valid else 0.0
    max_score = float(np.max(valid))  if valid else 0.0

    # ── CQT XAI on worst window ────────────────────────────────────────
    if worst_cqt_raw is not None:
        # _CQT_FREQS is imported at module top level
        xai_feats = {"cqt_raw": worst_cqt_raw, "cqt_freqs": _CQT_FREQS}
        cqt_b64, metrics = await loop.run_in_executor(
            _EXECUTOR,
            functools.partial(_compute_cqt_xai, xai_feats, worst_score)
        )
        yield _sse({
            "type":             "xai",
            "xai_type":         "cqt_heatmap",
            "cqt_heatmap_b64":  cqt_b64,
            "harmonic_metrics": metrics,
            "description": (
                "CQT heatmap shows frequency-time artifact evidence from the most "
                "suspicious 4s window. Y-axis = actual Hz (log scale). "
                "Harmonic metrics quantify synthesis regularity anomalies."
            ),
        })

    yield _sse({
        "type":          "done",
        "avg_score":     round(avg_score, 2),
        "max_score":     round(max_score, 2),
        "verdict":       "FAKE" if avg_score >= 50 else "REAL",
        "confidence":    _confidence_label(avg_score),
        "chunk_count":   total,
        "total_windows": len(all_windows),
        "sampled":       sampled,
        "elapsed_ms":    elapsed_ms,
    })


# ── SSE Response factory ──────────────────────────────────────────────────────

def _sse_response(generator: AsyncGenerator) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/model/1")
async def test_lcnn_mfcc(
    file: UploadFile = File(...),
    telephony: bool = False,
    telephony_mode: str = "random",
    noise: bool = False,
    reverb: bool = False,
):
    """LCNN-MFCC — 2s chunks, MFCC Grad-CAM. SSE stream."""
    audio_np = await _read_audio(file)
    if telephony or noise or reverb:
        audio_np, logs = run_inference_augmentation(
            audio_np, sr=settings.SR,
            telephony=telephony, telephony_mode=telephony_mode,
            noise=noise, reverb=reverb
        )
        logger.info(f"[Model Test 1] Applied augmentations: {logs}")
    return _sse_response(_stream_2s(audio_np, "lcnn_mfcc", do_gradcam=True))


@router.post("/model/2")
async def test_lcnn_lfcc(
    file: UploadFile = File(...),
    telephony: bool = False,
    telephony_mode: str = "random",
    noise: bool = False,
    reverb: bool = False,
):
    """LCNN-LFCC — 2s chunks, LFCC Grad-CAM. SSE stream."""
    audio_np = await _read_audio(file)
    if telephony or noise or reverb:
        audio_np, logs = run_inference_augmentation(
            audio_np, sr=settings.SR,
            telephony=telephony, telephony_mode=telephony_mode,
            noise=noise, reverb=reverb
        )
        logger.info(f"[Model Test 2] Applied augmentations: {logs}")
    return _sse_response(_stream_2s(audio_np, "lcnn_lfcc", do_gradcam=True))


@router.post("/model/3")
async def test_aasist(
    file: UploadFile = File(...),
    telephony: bool = False,
    telephony_mode: str = "random",
    noise: bool = False,
    reverb: bool = False,
):
    """AASIST-L — 2s chunks. SSE stream."""
    audio_np = await _read_audio(file)
    if telephony or noise or reverb:
        audio_np, logs = run_inference_augmentation(
            audio_np, sr=settings.SR,
            telephony=telephony, telephony_mode=telephony_mode,
            noise=noise, reverb=reverb
        )
        logger.info(f"[Model Test 3] Applied augmentations: {logs}")
    return _sse_response(_stream_2s(audio_np, "aasist", do_gradcam=False))


@router.post("/model/4")
async def test_dual_resnet(
    file: UploadFile = File(...),
    telephony: bool = False,
    telephony_mode: str = "random",
    noise: bool = False,
    reverb: bool = False,
):
    """DualResNet-LFCC+CQT — 4s windows, CQT heatmap. SSE stream."""
    audio_np = await _read_audio(file)
    if telephony or noise or reverb:
        audio_np, logs = run_inference_augmentation(
            audio_np, sr=settings.SR,
            telephony=telephony, telephony_mode=telephony_mode,
            noise=noise, reverb=reverb
        )
        logger.info(f"[Model Test 4] Applied augmentations: {logs}")
    return _sse_response(_stream_4s(audio_np))


# ── Startup pre-warm ──────────────────────────────────────────────────────────
# Load + JIT-warm the ONNX session in a background thread when the router is
# first imported, so the first real request doesn't pay the cold-start cost.

def _background_warmup():
    try:
        _warmup_dr_session()
    except Exception as e:
        logger.warning(f"[DualResNet] background warm-up failed (non-fatal): {e}")

_warmup_thread = threading.Thread(target=_background_warmup, daemon=True)
_warmup_thread.start()
