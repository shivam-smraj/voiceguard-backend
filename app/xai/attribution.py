"""
Heuristic TTS Attribution — Fingerprint which TTS system generated the audio.

Since we don't have a pre-trained Random Forest classifier, this module uses
spectral and biometric feature analysis to estimate the likely TTS source.

Attribution signals:
  - Jitter/Shimmer/HNR patterns → different TTS systems have distinct profiles
  - Spectral energy distribution → vocoder signatures
  - F0 stability → neural vs concatenative TTS
  - High-frequency energy → HiFi-GAN vs WaveGlow vs WaveRNN
"""

import logging
import numpy as np
from typing import Dict

logger = logging.getLogger(__name__)


# TTS system profiles derived from published literature + ASVspoof analysis
TTS_PROFILES = {
    "F5-TTS": {
        "jitter_range": (0.01, 0.15),    # very low jitter
        "shimmer_range": (0.02, 0.12),   # very low shimmer
        "hnr_range": (25, 40),           # very clean
        "f0_var_range": (0.1, 1.5),      # stable pitch
        "hf_energy_range": (0.01, 0.15), # moderate HF
        "description": "Zero-shot voice cloning with flow-matching. Produces highly natural-sounding "
                       "speech with minimal jitter/shimmer artifacts.",
        "vocoder": "HiFi-GAN v2",
    },
    "ChatTTS": {
        "jitter_range": (0.05, 0.25),
        "shimmer_range": (0.05, 0.20),
        "hnr_range": (20, 35),
        "f0_var_range": (0.5, 3.0),      # more variable
        "hf_energy_range": (0.02, 0.20),
        "description": "Conversational TTS with prosodic variation tokens. May show controlled "
                       "pitch variation but unnatural shimmer patterns.",
        "vocoder": "WaveGlow / Vocos",
    },
    "IndicSynth TTS": {
        "jitter_range": (0.02, 0.20),
        "shimmer_range": (0.03, 0.15),
        "hnr_range": (22, 38),
        "f0_var_range": (0.3, 2.5),
        "hf_energy_range": (0.01, 0.10), # less HF energy
        "description": "Indian-language specific TTS pipeline. Trained on Indic voice data with "
                       "language-specific phoneme models.",
        "vocoder": "WaveRNN / HiFi-GAN",
    },
    "ASVspoof LA Attack": {
        "jitter_range": (0.00, 0.10),
        "shimmer_range": (0.00, 0.08),
        "hnr_range": (28, 45),           # extremely clean
        "f0_var_range": (0.0, 0.8),      # very static
        "hf_energy_range": (0.05, 0.30), # strong HF artifacts
        "description": "Logical access attack from ASVspoof challenge. Typically very clean "
                       "with unnaturally high HNR and minimal perturbation.",
        "vocoder": "Various (challenge submissions)",
    },
    "ASVspoof DF Attack": {
        "jitter_range": (0.10, 0.40),
        "shimmer_range": (0.10, 0.30),
        "hnr_range": (15, 28),
        "f0_var_range": (1.0, 5.0),      # more variable
        "hf_energy_range": (0.10, 0.40), # significant HF
        "description": "Deepfake attack from ASVspoof challenge. May include voice conversion "
                       "artifacts with more natural-looking but still anomalous features.",
        "vocoder": "Various VC systems",
    },
}


def attribute_tts_source(
    biometrics: Dict,
    audio_np: np.ndarray,
    sr: int = 16000,
    verdict: str = "UNKNOWN",
) -> Dict:
    """
    Attribute the likely TTS system that generated the audio.

    Uses biometric measurements + spectral analysis to match against
    known TTS system profiles.

    Args:
        biometrics: Biometric measurements dict
        audio_np: Audio waveform
        sr: Sample rate
        verdict: Ensemble verdict ("FAKE" or "REAL") — used to prevent
                 contradictory Human attribution when verdict is FAKE

    Returns:
        ThreatAttribution dict
    """
    if biometrics is None:
        return _unknown_attribution()

    try:
        # Extract spectral features for attribution
        hf_energy = _compute_hf_energy(audio_np, sr)

        # Get biometric values
        jitter = biometrics.get('jitter_pct', 0.5)
        shimmer = biometrics.get('shimmer_db', 0.35)
        hnr = biometrics.get('hnr_db', 18.0)
        f0_var = biometrics.get('f0_variance_hz', 2.5)

        # Score each TTS profile
        scores = {}
        for name, profile in TTS_PROFILES.items():
            score = _score_profile(jitter, shimmer, hnr, f0_var, hf_energy, profile)
            scores[name] = score

        # Also score against "Human" profile
        human_score = _score_human(jitter, shimmer, hnr, f0_var)
        scores["Human (IndicVoices-R)"] = human_score

        # ── Fix C1: Prevent verdict/attribution contradiction ─────
        # If ensemble says FAKE, suppress the Human score so it
        # never wins the attribution ranking.
        if verdict == "FAKE":
            # Penalize human score — it shouldn't be #1 when verdict is FAKE
            scores["Human (IndicVoices-R)"] = min(
                human_score * 0.3,
                max(s for name, s in scores.items() if name != "Human (IndicVoices-R)") * 0.5
            )

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else ("N/A", 0)

        # Normalize confidence
        total = sum(s for _, s in ranked) + 1e-10
        top_conf = round((top[1] / total) * 100, 1)
        runner_conf = round((runner[1] / total) * 100, 1)

        # Generate signature notes
        if top[0] in TTS_PROFILES:
            profile = TTS_PROFILES[top[0]]
            notes = (
                f"{profile['description']} "
                f"Vocoder signature: {profile['vocoder']}. "
                f"HF energy ratio: {hf_energy:.4f}."
            )
        elif "Human" in top[0]:
            notes = (
                "Biometric markers are within expected human population ranges. "
                "No TTS vocoder signature detected in high-frequency bands."
            )
        else:
            notes = "Insufficient data for detailed attribution."

        return {
            "suspected_tool": top[0],
            "confidence": top_conf,
            "runner_up": runner[0],
            "runner_up_conf": runner_conf,
            "signature_notes": notes,
            "cluster_label": f"Cluster-{top[0][:3].upper()}",
            "source_probs": {name: round((s / total) * 100, 1) for name, s in ranked},
        }

    except Exception as e:
        logger.error(f"Attribution failed: {e}", exc_info=True)
        return _unknown_attribution()


def _score_profile(jitter, shimmer, hnr, f0_var, hf_energy, profile) -> float:
    """Score how well measurements match a TTS profile (higher = better match)."""
    score = 0.0

    # Jitter match
    j_lo, j_hi = profile["jitter_range"]
    if j_lo <= jitter <= j_hi:
        score += 25.0
    else:
        dist = min(abs(jitter - j_lo), abs(jitter - j_hi))
        score += max(0, 25.0 - dist * 50)

    # Shimmer match
    s_lo, s_hi = profile["shimmer_range"]
    if s_lo <= shimmer <= s_hi:
        score += 20.0
    else:
        dist = min(abs(shimmer - s_lo), abs(shimmer - s_hi))
        score += max(0, 20.0 - dist * 40)

    # HNR match
    h_lo, h_hi = profile["hnr_range"]
    if h_lo <= hnr <= h_hi:
        score += 25.0
    else:
        dist = min(abs(hnr - h_lo), abs(hnr - h_hi))
        score += max(0, 25.0 - dist * 3)

    # F0 variance match
    f_lo, f_hi = profile["f0_var_range"]
    if f_lo <= f0_var <= f_hi:
        score += 15.0
    else:
        dist = min(abs(f0_var - f_lo), abs(f0_var - f_hi))
        score += max(0, 15.0 - dist * 5)

    # HF energy match
    hf_lo, hf_hi = profile["hf_energy_range"]
    if hf_lo <= hf_energy <= hf_hi:
        score += 15.0
    else:
        dist = min(abs(hf_energy - hf_lo), abs(hf_energy - hf_hi))
        score += max(0, 15.0 - dist * 30)

    return max(0, score)


def _score_human(jitter, shimmer, hnr, f0_var) -> float:
    """Score how well measurements match the human baseline."""
    score = 0.0

    # Human jitter: 0.48% ± 0.21%
    if 0.20 <= jitter <= 1.0:
        score += 25.0
    elif jitter < 0.20:
        score += max(0, 25.0 - (0.20 - jitter) * 100)
    else:
        score += max(0, 25.0 - (jitter - 1.0) * 20)

    # Human shimmer: 0.35 dB ± 0.12 dB
    if 0.15 <= shimmer <= 0.8:
        score += 25.0
    elif shimmer < 0.15:
        score += max(0, 25.0 - (0.15 - shimmer) * 100)
    else:
        score += max(0, 25.0 - (shimmer - 0.8) * 30)

    # Human HNR: 18.2 ± 4.1 dB
    if 10.0 <= hnr <= 28.5:
        score += 25.0
    elif hnr > 28.5:
        score += max(0, 25.0 - (hnr - 28.5) * 5)
    else:
        score += max(0, 25.0 - (10.0 - hnr) * 5)

    # Human F0 variance: 2.4 ± 1.1 Hz
    if 1.0 <= f0_var <= 6.0:
        score += 25.0
    elif f0_var < 1.0:
        score += max(0, 25.0 - (1.0 - f0_var) * 25)
    else:
        score += max(0, 25.0 - (f0_var - 6.0) * 5)

    return max(0, score)


def _compute_hf_energy(audio: np.ndarray, sr: int) -> float:
    """Compute ratio of high-frequency energy (>4kHz) to total energy."""
    try:
        S = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
        hf_mask = freqs > 4000
        hf_energy = np.sum(S[hf_mask] ** 2)
        total_energy = np.sum(S ** 2) + 1e-10
        return float(hf_energy / total_energy)
    except Exception:
        return 0.05


def _unknown_attribution() -> Dict:
    return {
        "suspected_tool": "Unknown TTS",
        "confidence": 0,
        "runner_up": "N/A",
        "runner_up_conf": 0,
        "signature_notes": "Insufficient data for attribution analysis.",
        "cluster_label": "N/A",
        "source_probs": {},
    }
