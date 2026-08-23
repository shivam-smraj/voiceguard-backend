"""
Feature Extraction Pipeline — LFCC + MFCC with delta features.
Matches the training notebook extract_features() function exactly.

LFCC: 60 base + 60 delta + 60 delta-delta = 180 rows × 200 cols
MFCC: 40 base + 40 delta + 40 delta-delta = 120 rows × 200 cols
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# ── Build feature transforms (once, global) ──────────────────────────

_LFCC_TRANSFORM = torchaudio.transforms.LFCC(
    sample_rate=settings.SR,
    n_lfcc=settings.N_LFCC,
    speckwargs={
        'n_fft': settings.N_FFT,
        'hop_length': settings.HOP_LEN,
        'win_length': settings.WIN_LEN,
    }
)

_MFCC_TRANSFORM = torchaudio.transforms.MFCC(
    sample_rate=settings.SR,
    n_mfcc=settings.N_MFCC,
    melkwargs={
        'n_fft': settings.N_FFT,
        'hop_length': settings.HOP_LEN,
        'win_length': settings.WIN_LEN,
        'n_mels': settings.N_MELS,
    }
)


def extract_features(audio_np: np.ndarray, mode: str = 'lfcc') -> torch.Tensor:
    """
    Extract LFCC or MFCC features with Δ + ΔΔ from raw audio.

    This function is a direct copy of the training notebook's extract_features().

    Args:
        audio_np : float32 numpy array shape (CLIP_LEN,) = (32000,)
        mode     : 'lfcc' or 'mfcc'

    Returns:
        float32 tensor shape (n_coeffs*3, MAX_FRAMES)
        LFCC: (180, 200)  |  MFCC: (120, 200)
    """
    wf = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)   # (1, 32000)

    with torch.no_grad():
        if mode == 'lfcc':
            base = _LFCC_TRANSFORM(wf).squeeze(0)       # (60, T)
        else:
            base = _MFCC_TRANSFORM(wf).squeeze(0)       # (40, T)

        # Compute Δ and ΔΔ
        d1 = torchaudio.functional.compute_deltas(base)   # velocity
        d2 = torchaudio.functional.compute_deltas(d1)     # acceleration

        # Stack: (3×n_coeffs, T)
        feat = torch.cat([base, d1, d2], dim=0)

        # Pad or trim to exactly MAX_FRAMES
        T = feat.shape[-1]
        if T < settings.MAX_FRAMES:
            feat = F.pad(feat, (0, settings.MAX_FRAMES - T))
        else:
            feat = feat[..., :settings.MAX_FRAMES]

        # Zero-mean unit-variance normalisation per coefficient track
        mean = feat.mean(dim=-1, keepdim=True)
        std  = feat.std(dim=-1, keepdim=True) + 1e-8
        feat = (feat - mean) / std

    return feat.float()   # (n_coeffs*3, MAX_FRAMES)


def extract_all_features(audio_np: np.ndarray) -> dict:
    """
    Extract all feature types needed for the ensemble.

    Args:
        audio_np: float32 numpy array shape (CLIP_LEN,) = (32000,)

    Returns:
        dict with keys:
            "mfcc" → tensor (1, 1, 120, 200) — ready for LCNN-MFCC
            "lfcc" → tensor (1, 1, 180, 200) — ready for LCNN-LFCC
            "mfcc_raw" → tensor (120, 200) — without batch/channel dims
            "lfcc_raw" → tensor (180, 200) — without batch/channel dims
            "raw_waveform" → numpy array (32000,)
    """
    lfcc_feat = extract_features(audio_np, mode='lfcc')   # (180, 200)
    mfcc_feat = extract_features(audio_np, mode='mfcc')   # (120, 200)

    return {
        "lfcc": lfcc_feat.unsqueeze(0).unsqueeze(0),   # (1, 1, 180, 200)
        "mfcc": mfcc_feat.unsqueeze(0).unsqueeze(0),   # (1, 1, 120, 200)
        "lfcc_raw": lfcc_feat,                          # (180, 200)
        "mfcc_raw": mfcc_feat,                          # (120, 200)
        "raw_waveform": audio_np,                       # (32000,)
    }

