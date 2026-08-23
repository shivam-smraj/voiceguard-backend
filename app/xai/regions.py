"""
Region Finder — Extracts artifact regions from Grad-CAM heatmaps.
Converts hot spots into structured ArtifactFinding objects with
frequency ranges, time ranges, and human-readable reason strings.
"""

import numpy as np
from scipy import ndimage
from typing import List, Dict, Tuple
import logging

from app.config import settings

logger = logging.getLogger(__name__)


# ── Frequency-to-artifact mapping ─────────────────────────────────────
# Based on the Blueprint v2.0 research findings

ARTIFACT_MAP = [
    # (freq_range_hz, artifact_type, description_template)
    ((5500, 8000), "neural_vocoder",
     "Neural vocoder harmonic grid detected at {freq_lo}-{freq_hi} Hz. "
     "Consistent with HiFi-GAN/WaveGlow synthesis artifacts."),

    ((3500, 5500), "spectral_envelope",
     "Spectral envelope anomaly at {freq_lo}-{freq_hi} Hz. "
     "Unnatural spectral smoothness typical of TTS models."),

    ((2000, 3500), "formant_f2",
     "Formant F2 trajectory anomaly at {freq_lo}-{freq_hi} Hz. "
     "Vocal tract shape inconsistent with natural speech."),

    ((800, 2000), "formant_f1",
     "Formant F1 anomaly at {freq_lo}-{freq_hi} Hz. "
     "First formant pattern not matching natural production."),

    ((50, 800), "f0_pattern",
     "Fundamental frequency (F0) pattern anomaly at {freq_lo}-{freq_hi} Hz. "
     "Pitch contour too regular for natural speech."),
]


def find_artifact_regions(
    heatmap: np.ndarray,
    feature_mode: str = 'lfcc',
    threshold: float = None,
    min_area: int = None,
    max_findings: int = 7,
) -> List[Dict]:
    """
    Find suspicious regions in a Grad-CAM heatmap.

    Args:
        heatmap: (n_coeff, max_frames) array, values in [0,1]
        feature_mode: 'lfcc' or 'mfcc' — determines coefficient mapping
        threshold: activation threshold (default from settings)
        min_area: minimum connected component area (default from settings)
        max_findings: maximum number of findings to return

    Returns:
        List of finding dicts, sorted by confidence (highest first)
    """
    threshold = threshold or settings.GRADCAM_THRESHOLD
    min_area = min_area or settings.GRADCAM_MIN_AREA

    # ── Defensive: ensure heatmap is exactly 2D (n_coeff, n_frames) ──────────
    heatmap = np.squeeze(heatmap)   # remove any size-1 dims
    if heatmap.ndim == 1:
        # Degenerate single-frame: wrap into (n_coeff, 1)
        heatmap = heatmap[:, np.newaxis]
    elif heatmap.ndim > 2:
        logger.warning(
            "find_artifact_regions: heatmap has %d dims (shape %s), "
            "collapsing extra leading dims.",
            heatmap.ndim, heatmap.shape,
        )
        while heatmap.ndim > 2:
            heatmap = heatmap[0]

    if heatmap.ndim != 2 or heatmap.size == 0:
        logger.error(
            "find_artifact_regions: cannot reduce heatmap to 2D (shape=%s) — returning no findings.",
            heatmap.shape,
        )
        return []

    n_coeff, n_frames = heatmap.shape


    # Binary mask of hot regions
    binary = (heatmap > threshold).astype(np.int32)

    # Connected component labeling
    labeled, n_components = ndimage.label(binary)

    if n_components == 0:
        return []

    findings = []

    for comp_id in range(1, n_components + 1):
        # Get component mask
        mask = (labeled == comp_id)
        area = mask.sum()

        if area < min_area:
            continue

        # Bounding box: (row_min, col_min, row_max, col_max)
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()

        # Average activation in this region
        region_activation = heatmap[mask].mean()
        confidence = min(100.0, region_activation * 100)

        # Convert coefficient indices to frequency ranges
        freq_lo, freq_hi = _coeff_to_freq(row_min, row_max, n_coeff, feature_mode)

        # Convert frame indices to time ranges
        time_lo = col_min * settings.HOP_LEN / settings.SR
        time_hi = (col_max + 1) * settings.HOP_LEN / settings.SR

        # Determine artifact type and reason
        artifact_type, reason = _classify_region(
            freq_lo, freq_hi, row_min, row_max, n_coeff, feature_mode
        )

        # Determine evidence type
        evidence_type = f"gradcam_{feature_mode}"

        findings.append({
            "finding_id": f"{feature_mode.upper()}-{len(findings)+1}",
            "rank": 0,  # will be set after sorting
            "artifact_type": artifact_type,
            "evidence_type": evidence_type,
            "freq_range": [int(freq_lo), int(freq_hi)],
            "time_range": [round(time_lo, 3), round(time_hi, 3)],
            "coeff_range": [int(row_min), int(row_max)],
            "confidence": round(confidence, 1),
            "reason": reason,
            "phoneme_match": None,   # filled by PhonemeAligner later
            "artifact_ids": [],
            "bbox": [int(row_min), int(col_min), int(row_max - row_min + 1), int(col_max - col_min + 1)],
        })

    # Sort by confidence (highest first), limit to max_findings
    findings.sort(key=lambda f: f["confidence"], reverse=True)
    findings = findings[:max_findings]

    # Assign ranks
    for i, f in enumerate(findings):
        f["rank"] = i + 1

    return findings


def _coeff_to_freq(
    row_min: int,
    row_max: int,
    n_coeff: int,
    feature_mode: str,
) -> Tuple[float, float]:
    """
    Convert coefficient indices to approximate frequency range in Hz.

    For LFCC/MFCC with deltas:
      - Rows 0..N_base-1 = base coefficients (spectral envelope)
      - Rows N_base..2*N_base-1 = delta (velocity)
      - Rows 2*N_base..3*N_base-1 = delta-delta (acceleration)

    Base coefficient mapping: linear scale from 0 to Nyquist/2.
    """
    if feature_mode == 'lfcc':
        n_base = settings.N_LFCC    # 60
    else:
        n_base = settings.N_MFCC    # 40

    nyquist = settings.SR / 2.0   # 8000 Hz

    # Map row to base coefficient index (modulo for delta/deltadelta)
    base_min = row_min % n_base
    base_max = row_max % n_base

    # Fix H7: Modulo can invert the range when crossing a boundary
    if base_min > base_max:
        base_min, base_max = base_max, base_min

    # Approximate frequency mapping
    freq_lo = (base_min / n_base) * nyquist
    freq_hi = ((base_max + 1) / n_base) * nyquist

    # Final safety: ensure lo < hi
    if freq_lo > freq_hi:
        freq_lo, freq_hi = freq_hi, freq_lo

    return freq_lo, freq_hi


def _classify_region(
    freq_lo: float,
    freq_hi: float,
    row_min: int,
    row_max: int,
    n_coeff: int,
    feature_mode: str,
) -> Tuple[str, str]:
    """
    Classify a region by its frequency range and coefficient type.

    Returns (artifact_type, reason_string)
    """
    if feature_mode == 'lfcc':
        n_base = settings.N_LFCC
    else:
        n_base = settings.N_MFCC

    # Check if this is in delta or delta-delta bands
    is_delta = row_min >= n_base and row_max < 2 * n_base
    is_delta2 = row_min >= 2 * n_base

    if is_delta:
        return "formant_transition", (
            f"Formant transition velocity anomaly detected (delta coefficients "
            f"{row_min}-{row_max}). Rate of spectral change at "
            f"{freq_lo:.0f}-{freq_hi:.0f} Hz is inconsistent with natural speech."
        )
    elif is_delta2:
        return "spectral_accel", (
            f"Spectral acceleration anomaly (delta-delta coefficients "
            f"{row_min}-{row_max}). Second-order spectral dynamics at "
            f"{freq_lo:.0f}-{freq_hi:.0f} Hz indicate machine-generated speech."
        )

    # Base coefficient — match by frequency range
    freq_mid = (freq_lo + freq_hi) / 2

    for (f_lo, f_hi), artifact_type, desc_template in ARTIFACT_MAP:
        if freq_mid >= f_lo and freq_mid <= f_hi:
            reason = desc_template.format(freq_lo=int(freq_lo), freq_hi=int(freq_hi))
            return artifact_type, reason

    # Fallback
    return "spectral_envelope", (
        f"Spectral anomaly detected at {freq_lo:.0f}-{freq_hi:.0f} Hz "
        f"(coefficients {row_min}-{row_max}). Pattern inconsistent with "
        f"natural human speech production."
    )
