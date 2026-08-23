"""
XAI Engine — Orchestrates all explainability modules.

Runs in parallel with graceful degradation:
  1. Grad-CAM (MFCC + LFCC) → heatmaps + artifact regions
  2. Biometric statistics → jitter, shimmer, HNR, F0, pause energy, formant velocity
  3. Spectrogram + waveform + F0 contour → visual evidence images
  4. Embedding projection → PCA 2D scatter plot
  5. TTS attribution → heuristic fingerprinting
  6. Forensic summary → officer text + legal conclusion
"""

import asyncio
import logging
import time
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.config import settings
from app.xai.gradcam import generate_gradcam_for_model
from app.xai.regions import find_artifact_regions
from app.xai.statistics import AudioStatisticsAnalyzer

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound XAI tasks
_executor = ThreadPoolExecutor(max_workers=4)


async def run_xai_engine(
    audio_np: np.ndarray,
    audio_full: np.ndarray,
    features: Dict,
    registry,
    audit_id: str,
    verdict: str = "UNKNOWN",
    ensemble_score: float = 0.0,
    confidence_label: str = "LOW",
) -> Dict:
    """
    Run ALL XAI analysis modules.

    Executes in parallel with graceful degradation.
    """
    loop = asyncio.get_event_loop()
    start = time.time()

    logger.info(f"[{audit_id}] Starting XAI engine (full)")

    # ── Phase 1: Parallel core analyses ───────────────────────────────

    # Grad-CAM for MFCC
    mfcc_gradcam_task = loop.run_in_executor(
        _executor, _run_gradcam, registry, "lcnn_mfcc", features.get("mfcc"), audit_id,
    )

    # Grad-CAM for LFCC
    lfcc_gradcam_task = loop.run_in_executor(
        _executor, _run_gradcam, registry, "lcnn_lfcc", features.get("lfcc"), audit_id,
    )

    # Biometric statistics — run on the 2s CHUNK (audio_np), NOT full audio
    # This is critical for speed (2s vs 74s) and for correct biometric values.
    # TTS biometrics are only meaningful on short, consistent speech windows.
    stats_task = loop.run_in_executor(
        _executor, _run_statistics, audio_np, audit_id,
    )

    # Spectrogram + waveform + F0 contour (visual evidence)
    visuals_task = loop.run_in_executor(
        _executor, _run_visual_evidence, audio_full, audit_id,
    )

    # GradientSHAP baseline simulation & task launch
    shap_task = None
    mfcc_entry = registry.entries.get("lcnn_mfcc")
    if mfcc_entry and mfcc_entry.active and mfcc_entry.model and features.get("mfcc") is not None:
        try:
            from app.xai.shap_explainer import simulate_codec_degradation
            from app.features.pipeline import extract_features
            degraded_audio_np = simulate_codec_degradation(audio_np, sr=settings.SR)
            degraded_mfcc = extract_features(degraded_audio_np, mode='mfcc')
            baseline_mfcc_tensor = degraded_mfcc.unsqueeze(0).unsqueeze(0).float()
            
            shap_task = loop.run_in_executor(
                _executor, _run_shap_explanation,
                mfcc_entry.model, features.get("mfcc").float(), baseline_mfcc_tensor, audit_id
            )
        except Exception as e:
            logger.error(f"[{audit_id}] SHAP task prep failed: {e}")

    # Gather Phase 1
    tasks = [mfcc_gradcam_task, lfcc_gradcam_task, stats_task, visuals_task]
    if shap_task is not None:
        tasks.append(shap_task)

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )

    mfcc_gradcam = results[0] if not isinstance(results[0], Exception) else None
    lfcc_gradcam = results[1] if not isinstance(results[1], Exception) else None
    biometrics = results[2] if not isinstance(results[2], Exception) else None
    visuals = results[3] if not isinstance(results[3], Exception) else None
    shap_b64 = results[4] if len(results) > 4 and not isinstance(results[4], Exception) else ""

    for i, label in enumerate(["MFCC Grad-CAM", "LFCC Grad-CAM", "Statistics", "Visuals"]):
        if isinstance(results[i], Exception):
            logger.error(f"[{audit_id}] {label} failed: {results[i]}")
            
    if len(results) > 4 and isinstance(results[4], Exception):
        logger.error(f"[{audit_id}] GradientSHAP failed: {results[4]}")

    # ── Phase 2: Region finding ───────────────────────────────────────
    findings = []

    if mfcc_gradcam is not None:
        mfcc_findings = find_artifact_regions(mfcc_gradcam, feature_mode='mfcc')
        findings.extend(mfcc_findings)

    if lfcc_gradcam is not None:
        lfcc_findings = find_artifact_regions(lfcc_gradcam, feature_mode='lfcc')
        findings.extend(lfcc_findings)

    # Add biometric-based findings
    if biometrics and biometrics.get("n_suspicious", 0) > 0:
        findings.extend(_biometric_findings(biometrics))

    # Sort and re-rank
    findings.sort(key=lambda f: f["confidence"], reverse=True)
    for i, f in enumerate(findings):
        f["rank"] = i + 1
        f["finding_id"] = f"{f['evidence_type'].upper().split('_')[0]}-{i+1}"

    # ── Phase 3a: Heatmaps + Embedding + Phoneme + Speaker (parallel) ──

    # Heatmap rendering
    heatmaps_task = loop.run_in_executor(
        _executor, _run_heatmap_rendering,
        features, mfcc_gradcam, lfcc_gradcam, audit_id, findings,
    )

    # Embedding projection
    embedding_task = loop.run_in_executor(
        _executor, _run_embedding_projection,
        registry, features, audit_id,
    )

    # Whisper Phoneme alignment
    phoneme_task = loop.run_in_executor(
        _executor, _run_phoneme_alignment,
        audio_np, findings, audit_id,
    )

    # Voice enrollment matching
    speaker_task = loop.run_in_executor(
        _executor, _run_speaker_comparison,
        audio_np, registry, audit_id,
    )

    # Gather Phase 3a
    results_3a = await asyncio.gather(
        heatmaps_task, embedding_task, phoneme_task, speaker_task,
        return_exceptions=True,
    )

    heatmaps = results_3a[0] if not isinstance(results_3a[0], Exception) else {}
    embedding_proj = results_3a[1] if not isinstance(results_3a[1], Exception) else None
    phoneme_data = results_3a[2] if not isinstance(results_3a[2], Exception) else {"transcript": "", "words": []}
    speaker_match = results_3a[3] if not isinstance(results_3a[3], Exception) else {
        "match_found": False, "speaker_name": "No match (error)",
        "similarity": 0.0, "all_comparisons": []
    }

    for i, label in enumerate(["Heatmaps", "Embedding", "Phoneme Alignment", "Speaker Verification"]):
        if isinstance(results_3a[i], Exception):
            logger.error(f"[{audit_id}] {label} failed: {results_3a[i]}")

    # ── Phase 3b: Attribution (needs embedding_proj from 3a) ──────────
    attribution_task = loop.run_in_executor(
        _executor, _run_attribution,
        biometrics, audio_np, audit_id, verdict, embedding_proj,
    )

    attribution = await attribution_task
    if isinstance(attribution, Exception):
        logger.error(f"[{audit_id}] Attribution failed: {attribution}")
        attribution = None

    # ── Phase 4: Summary text generation ──────────────────────────────
    channel = _compute_channel_matrix(audio_np)  # Use 2s chunk for channel too

    # Determine analysis mode for Issue 9
    full_duration = len(audio_full) / settings.SR
    chunk_duration = len(audio_np) / settings.SR
    if full_duration > 3.0:
        analysis_mode = "POST_CALL_FORENSIC"
        processing_note = (
            f"This analysis was performed on a {full_duration:.1f}-second file. "
            f"XAI analysis ran on the most suspicious {chunk_duration:.1f}s window. "
            f"Real-time detection operates on 2-second live chunks."
        )
    else:
        analysis_mode = "REAL_TIME"
        processing_note = "Analysis performed in real-time mode on 2-second chunk."

    summary_data = _run_summary(
        verdict, ensemble_score, confidence_label,
        findings, biometrics, attribution, channel, embedding_proj,
    )

    elapsed_ms = int((time.time() - start) * 1000)
    logger.info(
        f"[{audit_id}] XAI engine done: {len(findings)} findings, "
        f"{len(visuals or {})} visuals in {elapsed_ms}ms"
    )

    return {
        # Heatmap images
        "heatmap_mfcc_b64": heatmaps.get("mfcc"),
        "heatmap_lfcc_b64": heatmaps.get("lfcc"),
        "shap_b64": shap_b64,

        # Visual evidence images
        "spectrogram_b64": (visuals or {}).get("spectrogram"),
        "waveform_b64": (visuals or {}).get("waveform"),
        "f0_contour_b64": (visuals or {}).get("f0_contour"),
        "biometric_radar_b64": (visuals or {}).get("biometric_radar"),

        # Embedding projection
        "embedding_projection": embedding_proj,
        "embedding_plot_b64": (embedding_proj.get("plot_b64") if embedding_proj else None) or (visuals or {}).get("embedding_plot"),

        # Findings
        "findings": findings,
        "n_findings": len(findings),

        # Biometrics
        "biometric_drift": biometrics,

        # Attribution
        "threat_attribution": attribution or _default_attribution(),

        # Channel
        "channel_matrix": channel,

        # Summary text
        "summary_text": (summary_data or {}).get("summary_text", ""),
        "forensic_conclusion": (summary_data or {}).get("forensic_conclusion", ""),
        "detailed_analysis": (summary_data or {}).get("detailed_analysis", ""),
        
        # Phonetic Alignment
        "transcript": phoneme_data.get("transcript", ""),
        "phoneme_timeline": phoneme_data.get("words", []),

        # Voice Enrollment Match
        "speaker_match": speaker_match,

        # Analysis metadata (Issue 9)
        "analysis_mode": analysis_mode,
        "processing_note": processing_note,
        "analysis_window_s": round(chunk_duration, 1),
        "full_duration_s": round(full_duration, 1),
    }


# ── Worker Functions ──────────────────────────────────────────────────

def _run_gradcam(registry, model_key: str, input_tensor, audit_id: str):
    """Run Grad-CAM for a single model (thread-safe)."""
    if input_tensor is None:
        return None

    entry = registry.entries.get(model_key)
    if not entry or not entry.loaded or not entry.model:
        return None

    logger.info(f"[{audit_id}] Running Grad-CAM for {entry.name}")
    heatmap = generate_gradcam_for_model(entry.model, input_tensor)
    logger.info(f"[{audit_id}] Grad-CAM {entry.name}: shape={heatmap.shape}, max={heatmap.max():.3f}")
    return heatmap


def _run_statistics(audio_full: np.ndarray, audit_id: str):
    """Run biometric statistics (thread-safe)."""
    logger.info(f"[{audit_id}] Running biometric statistics (6 tests)")
    analyzer = AudioStatisticsAnalyzer()
    result = analyzer.analyze(audio_full)
    logger.info(
        f"[{audit_id}] Biometrics: score={result['overall_biometric_score']}, "
        f"suspicious={result['n_suspicious']}/6"
    )
    return result


def _run_visual_evidence(audio_full: np.ndarray, audit_id: str) -> Dict:
    """Generate all visual evidence images (thread-safe)."""
    visuals = {}

    try:
        from app.xai.spectrogram import render_spectrogram, render_waveform, render_f0_contour

        logger.info(f"[{audit_id}] Rendering spectrogram")
        visuals["spectrogram"] = render_spectrogram(audio_full, settings.SR)

        logger.info(f"[{audit_id}] Rendering waveform")
        visuals["waveform"] = render_waveform(audio_full, settings.SR)

        logger.info(f"[{audit_id}] Rendering F0 contour")
        visuals["f0_contour"] = render_f0_contour(audio_full, settings.SR)

    except Exception as e:
        logger.error(f"[{audit_id}] Visual rendering failed: {e}", exc_info=True)

    return visuals


def _run_heatmap_rendering(features, mfcc_cam, lfcc_cam, audit_id: str, findings: list = None) -> Dict:
    """Render Grad-CAM heatmap images (thread-safe)."""
    result = {}

    try:
        from app.xai.heatmap_renderer import render_heatmap_trio

        if mfcc_cam is not None:
            try:
                result["mfcc"] = render_heatmap_trio(
                    features["mfcc_raw"].numpy(), mfcc_cam,
                    title="MFCC Grad-CAM Analysis",
                    findings=findings,
                    evidence_type="gradcam_mfcc",
                )
            except Exception as e:
                logger.error(f"[{audit_id}] MFCC heatmap render failed: {e}")

        if lfcc_cam is not None:
            try:
                result["lfcc"] = render_heatmap_trio(
                    features["lfcc_raw"].numpy(), lfcc_cam,
                    title="LFCC Grad-CAM Analysis",
                    findings=findings,
                    evidence_type="gradcam_lfcc",
                )
            except Exception as e:
                logger.error(f"[{audit_id}] LFCC heatmap render failed: {e}")

    except ImportError as e:
        logger.error(f"[{audit_id}] Heatmap renderer unavailable: {e}")

    return result


def _run_embedding_projection(registry, features, audit_id: str) -> Optional[Dict]:
    """Extract embedding and project to 2D (thread-safe)."""
    try:
        from app.xai.embedding import extract_embedding, project_embedding, render_embedding_plot

        # Use LFCC model for embedding (larger feature space)
        entry = registry.entries.get("lcnn_lfcc")
        feat_key = "lfcc"

        if not entry or not entry.loaded or not entry.model:
            entry = registry.entries.get("lcnn_mfcc")
            feat_key = "mfcc"

        if not entry or not entry.loaded or not entry.model:
            return None

        feat_tensor = features.get(feat_key)
        if feat_tensor is None:
            return None

        logger.info(f"[{audit_id}] Extracting 256-dim embedding from {entry.name}")
        embedding = extract_embedding(entry.model, feat_tensor)

        logger.info(f"[{audit_id}] Projecting embedding to 2D")
        projection = project_embedding(embedding)
        
        try:
            projection["plot_b64"] = render_embedding_plot(projection)
        except Exception as pe:
            logger.error(f"[{audit_id}] Embedding plot rendering failed: {pe}")
            projection["plot_b64"] = ""

        return projection

    except Exception as e:
        logger.error(f"[{audit_id}] Embedding projection failed: {e}", exc_info=True)
        return None


def _run_attribution(biometrics, audio_chunk, audit_id: str, verdict: str = "UNKNOWN",
                     embedding_proj=None) -> Optional[Dict]:
    """Run TTS attribution analysis (thread-safe)."""
    try:
        from app.xai.attribution import attribute_tts_source

        logger.info(f"[{audit_id}] Running TTS attribution")
        result = attribute_tts_source(biometrics, audio_chunk, settings.SR, verdict=verdict)

        # ── Issue 2: Cross-check with embedding distance ──────────
        if embedding_proj and isinstance(embedding_proj, dict):
            dist_human = embedding_proj.get("distance_to_human", 999)
            distances = embedding_proj.get("cluster_distances", {})
            # Find nearest fake cluster distance
            fake_distances = {k: v for k, v in distances.items() if "Human" not in k}
            nearest_fake_dist = min(fake_distances.values()) if fake_distances else 999

            if dist_human < nearest_fake_dist:
                result["embedding_contradiction"] = True
                result["signature_notes"] = (
                    f"Embedding projects near human cluster (dist={dist_human:.2f}) — "
                    f"closer than nearest TTS cluster (dist={nearest_fake_dist:.2f}). "
                    f"Attribution model cannot determine generation tool with confidence. "
                    f"The deepfake verdict is based on ensemble model consensus, "
                    f"not tool identification."
                )

        # ── Issue 3: Check if attribution is statistically meaningful ──
        top_conf = result.get("confidence", 0)
        runner_conf = result.get("runner_up_conf", 0)
        if abs(top_conf - runner_conf) < 5.0:
            result["attribution_reliable"] = False
            result["suspected_tool"] = "UNABLE TO ATTRIBUTE"
            result["signature_notes"] = (
                f"No dominant TTS signature detected. "
                f"Attribution confidence too low "
                f"(top: {top_conf:.1f}%, runner-up: {runner_conf:.1f}%, "
                f"spread: {abs(top_conf - runner_conf):.1f}%). "
                f"This does not affect the deepfake verdict, which is based "
                f"on ensemble model consensus ({verdict})."
            )
        else:
            result["attribution_reliable"] = True

        logger.info(
            f"[{audit_id}] Attribution: {result.get('suspected_tool')} "
            f"({result.get('confidence', 0):.0f}%)"
        )
        return result

    except Exception as e:
        logger.error(f"[{audit_id}] Attribution failed: {e}", exc_info=True)
        return _default_attribution()


def _run_summary(verdict, score, confidence_label, findings, biometrics,
                 attribution, channel, embedding_proj) -> Optional[Dict]:
    """Generate forensic summary text."""
    try:
        from app.xai.summary import generate_summary
        return generate_summary(
            verdict, score, confidence_label,
            findings, biometrics, attribution, channel, embedding_proj,
        )
    except Exception as e:
        logger.error(f"Summary generation failed: {e}", exc_info=True)
        return None


# ── Biometric Findings ────────────────────────────────────────────────

def _biometric_findings(biometrics: Dict):
    """Convert biometric anomalies to artifact findings."""
    findings = []

    if biometrics.get("jitter_status") == "MACHINE_PROFILE":
        findings.append({
            "finding_id": "STAT-J", "rank": 0,
            "artifact_type": "jitter_deficit", "evidence_type": "statistical",
            "freq_range": [50, 300], "time_range": [0.0, 2.0],
            "coeff_range": [0, 0],
            "confidence": round(min(90, 50 + biometrics.get("jitter_deviation_pct", 0)), 1),
            "reason": (
                f"Jitter deficit: {biometrics['jitter_pct']:.3f}% "
                f"(baseline: {biometrics['jitter_baseline']}% ± 0.21%). "
                f"Pitch period perturbation is unnaturally low — machine-generated speech."
            ),
            "phoneme_match": None, "artifact_ids": [], "bbox": None,
        })

    if biometrics.get("shimmer_status") == "LINEAR_GRID":
        findings.append({
            "finding_id": "STAT-S", "rank": 0,
            "artifact_type": "shimmer_linear", "evidence_type": "statistical",
            "freq_range": [50, 500], "time_range": [0.0, 2.0],
            "coeff_range": [0, 0], "confidence": 75.0,
            "reason": (
                f"Shimmer linearity: {biometrics['shimmer_db']:.3f} dB "
                f"(baseline: {biometrics['shimmer_baseline']} dB ± 0.12 dB). "
                f"Amplitude perturbation forms linear grid — vocoder quantization."
            ),
            "phoneme_match": None, "artifact_ids": [], "bbox": None,
        })

    if biometrics.get("hnr_status") == "TOO_CLEAN":
        findings.append({
            "finding_id": "STAT-H", "rank": 0,
            "artifact_type": "hnr_too_clean", "evidence_type": "statistical",
            "freq_range": [0, 8000], "time_range": [0.0, 2.0],
            "coeff_range": [0, 0], "confidence": 70.0,
            "reason": (
                f"HNR too clean: {biometrics['hnr_db']:.1f} dB "
                f"(baseline: {biometrics['hnr_baseline']} dB ± 4.1 dB). "
                f"Signal is unnaturally harmonic — no airflow turbulence noise."
            ),
            "phoneme_match": None, "artifact_ids": [], "bbox": None,
        })

    if biometrics.get("pause_status") == "DIGITAL_SILENCE":
        pe = biometrics.get("pause_energy", {})
        findings.append({
            "finding_id": "STAT-P", "rank": 0,
            "artifact_type": "digital_silence", "evidence_type": "statistical",
            "freq_range": [0, 8000], "time_range": [0.0, 2.0],
            "coeff_range": [0, 0], "confidence": 72.0,
            "reason": (
                f"Digital silence detected: min RMS = {pe.get('min_rms', 0):.8f} "
                f"(baseline min: {pe.get('baseline_min', 0.0008)}). "
                f"{pe.get('n_silent_frames', 0)} frames with near-zero energy — "
                f"indicates digitally generated rather than recorded audio."
            ),
            "phoneme_match": None, "artifact_ids": [], "bbox": None,
        })

    if biometrics.get("formant_status") == "INHUMAN_SPEED":
        fv = biometrics.get("formant_velocity", {})
        findings.append({
            "finding_id": "STAT-F2", "rank": 0,
            "artifact_type": "formant_velocity", "evidence_type": "statistical",
            "freq_range": [800, 3500], "time_range": [0.0, 2.0],
            "coeff_range": [0, 0],
            "confidence": round(min(85, 60 + fv.get('n_violations', 0) * 5), 1),
            "reason": (
                f"Formant F2 velocity anomaly: max {fv.get('max_hz_per_frame', 0):.0f} Hz/frame "
                f"(human limit: {fv.get('limit', 50):.0f} Hz/frame). "
                f"{fv.get('n_violations', 0)} frames exceed 2x the human articulation limit. "
                f"Vocal tract cannot physically move this fast."
            ),
            "phoneme_match": None, "artifact_ids": [], "bbox": None,
        })

    return findings


# ── Channel Matrix ────────────────────────────────────────────────────

def _compute_channel_matrix(audio: np.ndarray) -> Dict:
    """Compute channel integrity metrics."""
    try:
        frame_len = int(0.025 * settings.SR)
        hop = int(0.010 * settings.SR)

        energies = []
        for i in range(0, len(audio) - frame_len, hop):
            frame = audio[i:i+frame_len]
            energies.append(np.mean(frame**2))

        energies = np.array(energies)
        energies = energies[energies > 0]

        if len(energies) < 10:
            return _empty_channel()

        sorted_e = np.sort(energies)
        noise_floor = np.mean(sorted_e[:len(sorted_e)//10])
        signal_level = np.mean(sorted_e[-len(sorted_e)//4:])

        if noise_floor > 0:
            snr_db = 10 * np.log10(signal_level / noise_floor)
        else:
            snr_db = 40.0

        # Fix M10: Cap SNR at realistic maximum (studio = ~40 dB, TTS = often >50 dB)
        # SNR above 45 dB is physically unlikely for real recordings
        snr_capped = min(snr_db, 45.0)
        snr_suspicious = snr_db > 35.0  # Flag: could indicate TTS (no real noise floor)

        noise_floor_db = 10 * np.log10(noise_floor + 1e-10)

        # Reverb estimation
        autocorr = np.correlate(audio[:settings.SR], audio[:settings.SR], mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        decay = autocorr[settings.SR//10] / (autocorr[0] + 1e-10)

        if decay > 0.3:
            reverb = "HIGH"
        elif decay > 0.1:
            reverb = "MEDIUM"
        else:
            reverb = "LOW"

        # High-frequency energy check (codec detection)
        S = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / settings.SR)
        hf_mask = freqs > 4000
        hf_ratio = float(np.sum(S[hf_mask]**2) / (np.sum(S**2) + 1e-10))

        if hf_ratio < 0.02:
            codec = "G.711 (telephony)"
        elif hf_ratio < 0.05:
            codec = "G.722 (wideband)"
        else:
            codec = "None detected (wideband)"

        integrity = min(100, max(0, snr_capped * 2))

        # Build explanation with SNR context
        snr_note = ""
        if snr_suspicious:
            snr_note = (
                f" Note: Raw SNR ({snr_db:.0f} dB) exceeds typical recording levels "
                f"(capped to {snr_capped:.0f} dB) — anomalously clean signal may "
                f"indicate synthetic generation rather than real-world recording."
            )

        return {
            "snr_estimated_db": round(float(snr_capped), 1),
            "snr_raw_db": round(float(snr_db), 1),
            "snr_suspicious": snr_suspicious,
            "codec_detected": codec,
            "artifacts_survive_codec": integrity > 40,
            "noise_floor_db": round(float(noise_floor_db), 1),
            "reverb_level": reverb,
            "hf_energy_ratio": round(hf_ratio, 4),
            "channel_integrity_score": round(float(integrity), 1),
            "matrix_explanation": (
                f"Channel SNR is {snr_capped:.0f} dB with {reverb.lower()} reverb. "
                f"Codec: {codec}. HF energy ratio: {hf_ratio:.4f}. "
                f"Forensic artifacts detected in this channel condition are "
                f"{'likely genuine findings' if integrity > 60 else 'potentially degraded by noise'}."
                f"{snr_note}"
            ),
        }
    except Exception:
        return _empty_channel()


def _empty_channel() -> Dict:
    return {
        "snr_estimated_db": 0, "codec_detected": "unknown",
        "artifacts_survive_codec": True, "noise_floor_db": -60,
        "reverb_level": "LOW", "hf_energy_ratio": 0.0,
        "channel_integrity_score": 50,
        "matrix_explanation": "Channel analysis incomplete.",
    }


def _default_attribution() -> Dict:
    return {
        "suspected_tool": "Unknown TTS", "confidence": 0,
        "runner_up": "N/A", "runner_up_conf": 0,
        "signature_notes": "Attribution analysis unavailable.",
        "cluster_label": "N/A", "source_probs": {},
    }


def _run_shap_explanation(model, input_tensor, baseline_tensor, audit_id: str) -> str:
    """Run GradientSHAP explanation (thread-safe)."""
    try:
        from app.xai.shap_explainer import explain_features_with_shap
        logger.info(f"[{audit_id}] Running GradientSHAP explanation")
        return explain_features_with_shap(model, input_tensor, baseline_tensor)
    except Exception as e:
        logger.error(f"[{audit_id}] SHAP explanation task failed: {e}")
        return ""


def _run_phoneme_alignment(audio_np, findings, audit_id: str) -> dict:
    """Run Whisper Phoneme alignment (thread-safe)."""
    try:
        from app.xai.phoneme import align_findings_to_phonemes
        logger.info(f"[{audit_id}] Running Whisper Phoneme alignment")
        return align_findings_to_phonemes(audio_np, findings)
    except Exception as e:
        logger.error(f"[{audit_id}] Phoneme alignment task failed: {e}")
        return {"transcript": "Alignment failed.", "words": []}


def _run_speaker_comparison(audio_np, registry, audit_id: str) -> dict:
    """Run voice enrollment comparison (thread-safe)."""
    try:
        from app.xai.enrollment import compare_speaker
        logger.info(f"[{audit_id}] Running Voice Enrollment comparison")
        return compare_speaker(audio_np, registry)
    except Exception as e:
        logger.error(f"[{audit_id}] Speaker comparison task failed: {e}")
        return {
            "match_found": False, "speaker_name": "Error",
            "similarity": 0.0, "all_comparisons": []
        }

