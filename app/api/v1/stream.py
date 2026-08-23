"""
Stream Endpoint — Step 2 of the streaming analysis flow.

GET /api/v1/stream/{audit_id}
  Server-Sent Events (SSE) stream.
  Runs the full analysis pipeline and emits events as each
  component finishes — client sees results progressively.

Event types emitted (in order):
  started            → analysis begins, n_chunks known
  chunk_score        → one per chunk as detection runs
  preliminary_verdict → after first 3 chunks
  detection_complete → all chunks done, final verdict
  xai_started        → XAI pipeline begins
  gradcam_ready      → Grad-CAM heatmaps done
  biometrics_ready   → 6 biometric tests done
  shap_ready         → GradientSHAP map done
  visuals_ready      → spectrogram/waveform/F0 done
  phonemes_ready     → Whisper alignment done
  embedding_ready    → PCA projection done
  speaker_ready      → voice enrollment match done
  findings_ready     → artifact regions ranked
  attribution_ready  → TTS tool fingerprint done
  summary_ready      → forensic text done
  pdf_generating     → PDF render starting
  pdf_ready          → PDF complete, download URL live
  module_error       → one module failed (stream continues)
  complete           → all done
"""

import os
import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

import numpy as np
import torch

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models.registry import registry
from app.features.pipeline import extract_all_features, extract_features
from app.features.parallel import parallel_extract_all
from app.models.batch_infer import batch_ensemble_predict
from app.utils.audio import load_audio_full, load_audio_chunks
from app.api.v1.upload import get_pending, clear_pending

logger = logging.getLogger(__name__)

# ── CPU optimisation: use all available cores for PyTorch ops ─────────
_N_CORES = os.cpu_count() or 4
torch.set_num_threads(_N_CORES)
torch.set_num_interop_threads(max(1, _N_CORES // 2))
logger.info(f"PyTorch threads: intra={_N_CORES}, inter={max(1, _N_CORES // 2)}")

router = APIRouter(prefix="/api/v1", tags=["streaming"])

# More workers than before — feature extraction + XAI run concurrently
_executor = ThreadPoolExecutor(max_workers=max(8, _N_CORES * 2))

# Batch size for inference — tune this for your CPU
# 8 is a good default; reduce to 4 if you run out of RAM
_BATCH_SIZE = 8


# ── SSE helper ────────────────────────────────────────────────────────

def _sse(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event message."""
    payload = json.dumps(data, default=_numpy_serializer)
    return f"event: {event_type}\ndata: {payload}\n\n"


def _numpy_serializer(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


# ── Score helpers (mirrors analyze.py) ───────────────────────────────

def _aggregate_scores(scores: np.ndarray) -> float:
    if len(scores) == 0:
        return 0.0
    if len(scores) == 1:
        return float(scores[0])
    max_score = float(np.max(scores))
    mean_score = float(np.mean(scores))
    k = max(1, len(scores) // 10)
    top_k = np.sort(scores)[-k:]
    top_mean = float(np.mean(top_k))
    return round(0.6 * max_score + 0.3 * top_mean + 0.1 * mean_score, 2)


def _confidence_label(score: float) -> str:
    if score >= 85 or score <= 15:
        return "CRITICAL"
    elif score >= 70 or score <= 30:
        return "HIGH"
    elif score >= 55 or score <= 45:
        return "MEDIUM"
    return "LOW"


# ── Main streaming generator ──────────────────────────────────────────

async def _run_streaming_analysis(pending: dict) -> AsyncGenerator[str, None]:
    """
    Full analysis pipeline as an async generator.
    Yields SSE strings as each stage completes.
    """
    audit_id   = pending["audit_id"]
    tmp_path   = pending["tmp_path"]
    call_id    = pending["call_id"]
    audio_info = pending["audio_info"]
    loop       = asyncio.get_event_loop()
    start_time = time.time()

    # ── Load audio ────────────────────────────────────────────────────
    try:
        chunks    = load_audio_chunks(tmp_path)
        audio_full = load_audio_full(tmp_path)
        n_chunks  = len(chunks)
    except Exception as e:
        yield _sse("error", {"module": "audio_load", "error": str(e)})
        yield _sse("complete", {"error": True})
        return

    yield _sse("started", {
        "audit_id":   audit_id,
        "call_id":    call_id,
        "n_chunks":   n_chunks,
        "duration_s": audio_info["duration_s"],
        "filename":   pending["filename"],
    })

    # ── Detection: PARALLEL extract + BATCH inference ─────────────────
    #
    # Strategy (6-10x faster than sequential):
    #   Phase A — Parallel feature extraction
    #             All CPU cores extract MFCC/LFCC simultaneously
    #   Phase B — Batched inference
    #             Stack _BATCH_SIZE chunks → one forward pass per model
    #             Stream chunk_score events as each batch finishes
    #
    chunk_results    = []
    all_scores       = []
    preliminary_sent = False
    PRELIM_AFTER     = 3
    detection_start  = time.time()

    # ── Phase A: Extract ALL features in parallel ──────────────────────
    logger.info(f"[{audit_id}] Extracting features for {n_chunks} chunks in parallel...")
    t_extract_start = time.time()

    indexed_features = await loop.run_in_executor(
        _executor,
        parallel_extract_all,
        chunks,
    )
    # indexed_features: [(chunk_idx, features_dict), ...] in order

    extract_ms = int((time.time() - t_extract_start) * 1000)
    logger.info(
        f"[{audit_id}] Parallel extraction: {len(indexed_features)}/{n_chunks} chunks "
        f"in {extract_ms}ms ({extract_ms/max(1,n_chunks):.0f}ms/chunk)"
    )

    # Build a lookup: chunk_idx → chunk metadata
    chunk_meta = {c["chunk_idx"]: c for c in chunks}

    # ── Phase B: Batch inference — process _BATCH_SIZE chunks at a time ─
    logger.info(f"[{audit_id}] Batch inference (batch={_BATCH_SIZE})...")

    for batch_start in range(0, len(indexed_features), _BATCH_SIZE):
        batch = indexed_features[batch_start : batch_start + _BATCH_SIZE]
        batch_indices   = [b[0] for b in batch]
        batch_features  = [b[1] for b in batch]

        try:
            # ONE forward pass per model for the entire batch
            batch_results = await loop.run_in_executor(
                _executor,
                batch_ensemble_predict,
                registry,
                batch_features,
            )
        except Exception as e:
            logger.error(f"[{audit_id}] Batch inference failed: {e}")
            for idx in batch_indices:
                yield _sse("module_error", {
                    "module": f"chunk_{idx}",
                    "error": str(e),
                })
            continue

        # Stream chunk_score for each item in the batch
        for result, chunk_idx in zip(batch_results, batch_indices):
            meta  = chunk_meta[chunk_idx]
            score = result["ensemble_score"]
            all_scores.append(score)
            chunk_results.append({
                "chunk_idx":      chunk_idx,
                "start_s":        meta["start_s"],
                "end_s":          meta["end_s"],
                "ensemble_score": score,
                "verdict":        result["verdict"],
                "per_model":      result["per_model"],
            })

        # Compute running aggregate after batch
        running_score   = float(_aggregate_scores(np.array(all_scores)))
        running_verdict = "FAKE" if running_score >= 50.0 else "REAL"

        # Emit one chunk_score event per chunk in the batch
        for result, chunk_idx in zip(batch_results, batch_indices):
            meta  = chunk_meta[chunk_idx]
            score = result["ensemble_score"]
            yield _sse("chunk_score", {
                "chunk_index":     chunk_idx,
                "score":           round(score, 2),
                "verdict":         result["verdict"],
                "running_score":   round(running_score, 2),
                "running_verdict": running_verdict,
                "n_chunks_done":   len(all_scores),
                "n_chunks_total":  n_chunks,
                "start_s":         meta["start_s"],
                "end_s":           meta["end_s"],
                "per_model":       result["per_model"],
            })

        # Preliminary verdict after first N chunks processed
        if len(all_scores) >= PRELIM_AFTER and not preliminary_sent:
            preliminary_sent = True
            yield _sse("preliminary_verdict", {
                "verdict":         running_verdict,
                "score":           round(running_score, 2),
                "confidence":      _confidence_label(running_score),
                "based_on_chunks": len(all_scores),
                "total_chunks":    n_chunks,
            })

        # Yield to the event loop between batches so SSE flushes immediately
        await asyncio.sleep(0)

    # ── Detection complete ────────────────────────────────────────────
    if not all_scores:
        yield _sse("error", {"module": "detection", "error": "No chunks processed"})
        yield _sse("complete", {"error": True})
        return

    scores_arr    = np.array(all_scores)
    overall_score = _aggregate_scores(scores_arr)
    overall_verdict = "FAKE" if overall_score >= 50.0 else "REAL"
    confidence_label = _confidence_label(overall_score)

    worst_idx = int(np.argmax(scores_arr))
    n_fake    = int(np.sum(scores_arr >= 50.0))
    detection_ms = int((time.time() - detection_start) * 1000)
    batch_ms  = detection_ms - extract_ms  # inference-only time

    logger.info(
        f"[{audit_id}] Detection: {n_chunks} chunks | "
        f"extract={extract_ms}ms | batch_infer={batch_ms}ms | "
        f"total={detection_ms}ms | {detection_ms/max(1,n_chunks):.0f}ms/chunk"
    )

    temporal_analysis = {
        "n_chunks":         n_chunks,
        "chunk_duration_s": settings.CHUNK_SEC,
        "overlap_s":        settings.CHUNK_OVERLAP_SEC,
        "total_duration_s": round(audio_info["duration_s"], 1),
        "processing_sr":    settings.SR,
        "original_sr":      audio_info["sample_rate"],
        "chunk_scores": [
            {
                "idx":     cr["chunk_idx"],
                "start_s": cr["start_s"],
                "end_s":   cr["end_s"],
                "score":   cr["ensemble_score"],
                "verdict": cr["verdict"],
            }
            for cr in chunk_results
        ],
        "n_fake_chunks":  n_fake,
        "n_real_chunks":  n_chunks - n_fake,
        "fake_chunk_pct": round(n_fake / max(1, n_chunks) * 100, 1),
        "max_score":      round(float(scores_arr.max()), 2),
        "min_score":      round(float(scores_arr.min()), 2),
        "mean_score":     round(float(scores_arr.mean()), 2),
        "std_score":      round(float(scores_arr.std()), 2),
        "worst_chunk_idx":  worst_idx,
        "worst_chunk_time": f"{chunks[worst_idx]['start_s']:.1f}-{chunks[worst_idx]['end_s']:.1f}s",
    }

    yield _sse("detection_complete", {
        "audit_id":          audit_id,
        "verdict":           overall_verdict,
        "score":             overall_score,
        "confidence_label":  confidence_label,
        "detection_ms":      detection_ms,
        "temporal_analysis": temporal_analysis,
        "per_model":         chunk_results[worst_idx]["per_model"],
    })

    # ── XAI Phase — extract worst chunk features ─────────────────────
    worst_chunk = chunks[worst_idx]
    try:
        worst_features = await loop.run_in_executor(
            _executor, extract_all_features, worst_chunk["audio_np"]
        )
    except Exception as e:
        yield _sse("module_error", {"module": "feature_extraction", "error": str(e)})
        yield _sse("complete", {"error": True})
        return

    yield _sse("xai_started", {
        "modules": [
            "gradcam", "biometrics", "shap", "visuals",
            "phonemes", "embedding", "speaker", "attribution", "summary"
        ],
        "worst_chunk_idx":  worst_idx,
        "worst_chunk_time": temporal_analysis["worst_chunk_time"],
    })

    # ── Import XAI workers lazily ─────────────────────────────────────
    from app.xai.gradcam       import generate_gradcam_for_model
    from app.xai.regions       import find_artifact_regions
    from app.xai.heatmap_renderer import render_heatmap_trio
    from app.xai.statistics    import AudioStatisticsAnalyzer
    from app.xai.spectrogram   import render_spectrogram, render_waveform, render_f0_contour
    from app.xai.shap_explainer import simulate_codec_degradation, explain_features_with_shap
    from app.xai.embedding     import extract_embedding, project_embedding, render_embedding_plot
    from app.xai.phoneme       import align_findings_to_phonemes
    from app.xai.enrollment    import compare_speaker
    from app.xai.attribution   import attribute_tts_source
    from app.xai.summary       import generate_summary as _gen_summary
    from app.xai.engine        import _compute_channel_matrix, _biometric_findings, _run_attribution

    # ── Launch Phase 1 parallel tasks (closure vars for heatmaps) ─────
    _heatmap_mfcc_b64 = ""
    _heatmap_lfcc_b64 = ""
    _findings_list    = []
    mfcc_entry = registry.entries.get("lcnn_mfcc")
    lfcc_entry = registry.entries.get("lcnn_lfcc")

    # Grad-CAM tasks
    def _gradcam_mfcc():
        if not mfcc_entry or not mfcc_entry.loaded:
            return None
        return generate_gradcam_for_model(mfcc_entry.model, worst_features.get("mfcc"))

    def _gradcam_lfcc():
        if not lfcc_entry or not lfcc_entry.loaded:
            return None
        return generate_gradcam_for_model(lfcc_entry.model, worst_features.get("lfcc"))

    def _biometrics():
        analyzer = AudioStatisticsAnalyzer()
        return analyzer.analyze(worst_chunk["audio_np"])

    def _visuals():
        result = {}
        result["spectrogram"]  = render_spectrogram(audio_full, settings.SR)
        result["waveform"]     = render_waveform(audio_full, settings.SR)
        result["f0_contour"]   = render_f0_contour(audio_full, settings.SR)
        return result

    def _shap():
        if not mfcc_entry or not mfcc_entry.loaded or worst_features.get("mfcc") is None:
            return ""
        degraded = simulate_codec_degradation(worst_chunk["audio_np"], sr=settings.SR)
        degraded_feat = extract_features(degraded, mode="mfcc")
        baseline = degraded_feat.unsqueeze(0).unsqueeze(0).float()
        return explain_features_with_shap(
            mfcc_entry.model,
            worst_features["mfcc"].float(),
            baseline,
        )

    # Schedule all phase-1 tasks
    t_mfcc     = loop.run_in_executor(_executor, _gradcam_mfcc)
    t_lfcc     = loop.run_in_executor(_executor, _gradcam_lfcc)
    t_bio      = loop.run_in_executor(_executor, _biometrics)
    t_visuals  = loop.run_in_executor(_executor, _visuals)
    t_shap     = loop.run_in_executor(_executor, _shap)

    # ── Collect phase-1 results as they complete ──────────────────────
    phase1_futures = {
        "gradcam_mfcc": t_mfcc,
        "gradcam_lfcc": t_lfcc,
        "biometrics":   t_bio,
        "visuals":      t_visuals,
        "shap":         t_shap,
    }

    mfcc_gradcam = lfcc_gradcam = biometrics = visuals = shap_b64 = None

    # We use asyncio.as_completed via a helper that tracks which key finished
    pending_tasks = {name: fut for name, fut in phase1_futures.items()}
    done_results  = {}

    # Poll until all complete, yielding events as each finishes
    while pending_tasks:
        await asyncio.sleep(0.05)  # small yield to event loop
        newly_done = []
        for name, fut in list(pending_tasks.items()):
            if fut.done():
                newly_done.append(name)
        for name in newly_done:
            fut = pending_tasks.pop(name)
            try:
                result = fut.result()
                done_results[name] = result
            except Exception as e:
                logger.error(f"[{audit_id}] {name} failed: {e}")
                yield _sse("module_error", {"module": name, "error": str(e)})
                done_results[name] = None
                continue

            # Emit events as each module finishes
            if name == "biometrics" and result is not None:
                biometrics = result
                yield _sse("biometrics_ready", {
                    "biometric_drift": result,
                })

            elif name == "shap" and result is not None:
                shap_b64 = result
                yield _sse("shap_ready", {"shap_b64": result})

            elif name == "visuals" and result is not None:
                visuals = result
                yield _sse("visuals_ready", {
                    "spectrogram_b64": result.get("spectrogram", ""),
                    "waveform_b64":    result.get("waveform", ""),
                    "f0_contour_b64":  result.get("f0_contour", ""),
                })

            elif name in ("gradcam_mfcc", "gradcam_lfcc"):
                if name == "gradcam_mfcc":
                    mfcc_gradcam = result
                else:
                    lfcc_gradcam = result

                # Emit Grad-CAM once both are done
                if mfcc_gradcam is not None or lfcc_gradcam is not None:
                    if "gradcam_mfcc" not in pending_tasks and "gradcam_lfcc" not in pending_tasks:
                        # Both finished — build findings and heatmaps
                        findings = []
                        if mfcc_gradcam is not None:
                            findings.extend(find_artifact_regions(mfcc_gradcam, feature_mode="mfcc"))
                        if lfcc_gradcam is not None:
                            findings.extend(find_artifact_regions(lfcc_gradcam, feature_mode="lfcc"))
                        findings.sort(key=lambda f: f["confidence"], reverse=True)
                        for i, f in enumerate(findings):
                            f["rank"] = i + 1
                            f["finding_id"] = f"{f['evidence_type'].upper().split('_')[0]}-{i+1}"

                        # Render heatmaps
                        heatmap_mfcc_b64 = ""
                        heatmap_lfcc_b64 = ""
                        try:
                            if mfcc_gradcam is not None:
                                heatmap_mfcc_b64 = render_heatmap_trio(
                                    worst_features["mfcc_raw"].numpy(), mfcc_gradcam,
                                    title="MFCC Grad-CAM Analysis",
                                    findings=findings, evidence_type="gradcam_mfcc",
                                )
                        except Exception as e:
                            logger.error(f"[{audit_id}] MFCC heatmap render failed: {e}")
                        try:
                            if lfcc_gradcam is not None:
                                heatmap_lfcc_b64 = render_heatmap_trio(
                                    worst_features["lfcc_raw"].numpy(), lfcc_gradcam,
                                    title="LFCC Grad-CAM Analysis",
                                    findings=findings, evidence_type="gradcam_lfcc",
                                )
                        except Exception as e:
                            logger.error(f"[{audit_id}] LFCC heatmap render failed: {e}")

                        _heatmap_mfcc_b64 = heatmap_mfcc_b64
                        _heatmap_lfcc_b64 = heatmap_lfcc_b64
                        _findings_list    = findings
                        yield _sse("gradcam_ready", {
                            "heatmap_mfcc_b64": heatmap_mfcc_b64,
                            "heatmap_lfcc_b64": heatmap_lfcc_b64,
                            "findings":         findings,
                            "n_findings":       len(findings),
                        })

    # After phase 1: add biometric findings if needed
    # Use the findings built during gradcam phase (or empty if gradcam failed)
    findings = _findings_list

    if biometrics and biometrics.get("n_suspicious", 0) > 0:
        bio_findings = _biometric_findings(biometrics)
        findings.extend(bio_findings)
        findings.sort(key=lambda f: f["confidence"], reverse=True)
        for i, f in enumerate(findings):
            f["rank"] = i + 1
            f["finding_id"] = f"{f['evidence_type'].upper().split('_')[0]}-{i+1}"

    # ── Phase 2: Embedding, Phoneme, Speaker (parallel) ──────────────
    def _embedding():
        entry = lfcc_entry or mfcc_entry
        feat_key = "lfcc" if lfcc_entry else "mfcc"
        if not entry or not entry.loaded:
            return None
        feat = worst_features.get(feat_key)
        if feat is None:
            return None
        emb = extract_embedding(entry.model, feat)
        proj = project_embedding(emb)
        try:
            proj["plot_b64"] = render_embedding_plot(proj)
        except Exception:
            proj["plot_b64"] = ""
        return proj

    def _phonemes():
        return align_findings_to_phonemes(worst_chunk["audio_np"], findings)

    def _speaker():
        return compare_speaker(worst_chunk["audio_np"], registry)

    t_emb     = loop.run_in_executor(_executor, _embedding)
    t_phoneme = loop.run_in_executor(_executor, _phonemes)
    t_speaker = loop.run_in_executor(_executor, _speaker)

    phase2_futures = {
        "embedding": t_emb,
        "phonemes":  t_phoneme,
        "speaker":   t_speaker,
    }

    embedding_proj = None
    phoneme_data   = {"transcript": "", "words": []}
    speaker_match  = {"match_found": False, "speaker_name": "No match", "similarity": 0.0, "all_comparisons": []}

    while phase2_futures:
        await asyncio.sleep(0.05)
        newly_done = [name for name, fut in phase2_futures.items() if fut.done()]
        for name in newly_done:
            fut = phase2_futures.pop(name)
            try:
                result = fut.result()
            except Exception as e:
                logger.error(f"[{audit_id}] {name} failed: {e}")
                yield _sse("module_error", {"module": name, "error": str(e)})
                continue

            if name == "embedding" and result is not None:
                embedding_proj = result
                yield _sse("embedding_ready", {
                    "embedding_projection": result,
                    "embedding_plot_b64":   result.get("plot_b64", ""),
                })

            elif name == "phonemes" and result is not None:
                phoneme_data = result
                yield _sse("phonemes_ready", {
                    "transcript":      result.get("transcript", ""),
                    "phoneme_timeline": result.get("words", []),
                })

            elif name == "speaker" and result is not None:
                speaker_match = result
                yield _sse("speaker_ready", {"speaker_match": result})

    # ── Phase 3: Attribution + Summary + Channel ──────────────────────
    def _attribution():
        return _run_attribution(biometrics, worst_chunk["audio_np"], audit_id, overall_verdict, embedding_proj)

    attribution = await loop.run_in_executor(_executor, _attribution)
    if attribution:
        yield _sse("attribution_ready", {"threat_attribution": attribution})

    channel = _compute_channel_matrix(worst_chunk["audio_np"])

    full_duration  = len(audio_full) / settings.SR
    chunk_duration = len(worst_chunk["audio_np"]) / settings.SR
    if full_duration > 3.0:
        analysis_mode    = "POST_CALL_FORENSIC"
        processing_note  = (
            f"Analysis on {full_duration:.1f}s file. "
            f"XAI ran on the most suspicious {chunk_duration:.1f}s window."
        )
    else:
        analysis_mode   = "REAL_TIME"
        processing_note = "Analysis performed in real-time mode on 2-second chunk."

    try:
        summary_data = _gen_summary(
            overall_verdict, overall_score, confidence_label,
            findings, biometrics, attribution, channel, embedding_proj,
        )
    except Exception as e:
        logger.error(f"[{audit_id}] Summary failed: {e}")
        summary_data = {}

    yield _sse("summary_ready", {
        "summary_text":       (summary_data or {}).get("summary_text", ""),
        "forensic_conclusion": (summary_data or {}).get("forensic_conclusion", ""),
        "detailed_analysis":  (summary_data or {}).get("detailed_analysis", ""),
        "channel_matrix":     channel,
        "analysis_mode":      analysis_mode,
        "processing_note":    processing_note,
    })

    # ── Build full response dict for PDF ─────────────────────────────
    xai_result = {
        "heatmap_mfcc_b64":    _heatmap_mfcc_b64,
        "heatmap_lfcc_b64":    _heatmap_lfcc_b64,
        "shap_b64":            shap_b64 or "",
        "spectrogram_b64":     (visuals or {}).get("spectrogram", ""),
        "waveform_b64":        (visuals or {}).get("waveform", ""),
        "f0_contour_b64":      (visuals or {}).get("f0_contour", ""),
        "biometric_radar_b64": (visuals or {}).get("biometric_radar", ""),
        "embedding_projection": embedding_proj,
        "embedding_plot_b64":  (embedding_proj or {}).get("plot_b64", ""),
        "findings":            findings,
        "n_findings":          len(findings),
        "biometric_drift":     biometrics,
        "threat_attribution":  attribution,
        "channel_matrix":      channel,
        "summary_text":        (summary_data or {}).get("summary_text", ""),
        "forensic_conclusion": (summary_data or {}).get("forensic_conclusion", ""),
        "detailed_analysis":   (summary_data or {}).get("detailed_analysis", ""),
        "transcript":          phoneme_data.get("transcript", ""),
        "phoneme_timeline":    phoneme_data.get("words", []),
        "speaker_match":       speaker_match,
        "analysis_mode":       analysis_mode,
        "processing_note":     processing_note,
        "analysis_window_s":   round(chunk_duration, 1),
        "full_duration_s":     round(full_duration, 1),
        "worst_chunk_idx":     worst_idx,
        "worst_chunk_time":    temporal_analysis["worst_chunk_time"],
    }

    # Re-fetch heatmap b64 from done_results if available
    # (they were computed inside the gradcam_ready branch above)
    # We store them in a closure variable
    full_response = {
        "audit_id":          audit_id,
        "call_id":           call_id,
        "verdict":           overall_verdict,
        "ensemble_score":    overall_score,
        "confidence_label":  confidence_label,
        "per_model":         chunk_results[worst_idx]["per_model"],
        "temporal_analysis": temporal_analysis,
        "audio_info": {
            "duration_s":    audio_info["duration_s"],
            "sample_rate":   audio_info["sample_rate"],
            "processing_sr": settings.SR,
            "audio_hash":    pending["audio_hash"],
            "file_size_mb":  pending["file_size_mb"],
        },
        "xai": xai_result,
        "performance": {
            "total_ms":         int((time.time() - start_time) * 1000),
            "detection_ms":     detection_ms,
            "xai_ms":           0,
            "chunks_processed": n_chunks,
            "ms_per_chunk":     round(detection_ms / max(1, n_chunks)),
        },
    }

    # ── PDF Generation ────────────────────────────────────────────────
    yield _sse("pdf_generating", {"audit_id": audit_id})

    try:
        from app.report.pdf_generator import generate_forensic_pdf
        pdf_path = str(settings.PDF_OUTPUT_DIR / f"{audit_id}.pdf")

        def _make_pdf():
            generate_forensic_pdf(full_response, output_path=pdf_path)

        await loop.run_in_executor(_executor, _make_pdf)
        pdf_url = f"/api/v1/report/{audit_id}/pdf"
        logger.info(f"[{audit_id}] PDF generated: {pdf_path}")
        yield _sse("pdf_ready", {"pdf_url": pdf_url, "audit_id": audit_id})
    except Exception as e:
        logger.error(f"[{audit_id}] PDF generation failed: {e}", exc_info=True)
        yield _sse("module_error", {"module": "pdf", "error": str(e)})

    # ── Cleanup ───────────────────────────────────────────────────────
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
    clear_pending(audit_id)

    total_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"[{audit_id}] Streaming analysis complete: {overall_verdict} "
        f"({overall_score:.1f}%) | {n_chunks} chunks | {total_ms}ms"
    )

    yield _sse("complete", {
        "audit_id":    audit_id,
        "verdict":     overall_verdict,
        "score":       overall_score,
        "total_ms":    total_ms,
    })


# ── FastAPI Route ─────────────────────────────────────────────────────

@router.get("/stream/{audit_id}")
async def stream_analysis(audit_id: str):
    """
    Server-Sent Events stream for progressive analysis results.

    Open with:
      const es = new EventSource('/api/v1/stream/' + auditId);
      es.addEventListener('chunk_score', e => console.log(JSON.parse(e.data)));
    """
    pending = get_pending(audit_id)
    if not pending:
        raise HTTPException(
            status_code=404,
            detail=f"No pending analysis found for audit_id={audit_id}. "
                   f"Upload audio first via POST /api/v1/upload"
        )

    async def event_stream():
        try:
            async for event in _run_streaming_analysis(pending):
                yield event
        except asyncio.CancelledError:
            logger.info(f"[{audit_id}] Client disconnected")
            raise
        except Exception as e:
            logger.error(f"[{audit_id}] Stream error: {e}", exc_info=True)
            yield _sse("error", {"module": "stream", "error": str(e)})
            yield _sse("complete", {"error": True})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering": "no",      # Disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )
