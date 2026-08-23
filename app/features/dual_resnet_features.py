"""
Dual-Branch ResNet Feature Extraction — FAST path.

KEY OPTIMISATION (v2):
  OLD: compute CQT per 4s window (librosa.cqt × N windows = 30–50s total)
  NEW: compute CQT ONCE on full audio with res_type='fft', slice per window.
       Plus LFCC computed once on full audio, delta/ΔΔ on full signal, then slice.

Speedup breakdown for 10s audio (4 windows):
  Before: 4 × (librosa.cqt ~10s + LFCC ~0.3s + ONNX ~0.07s) = ~41s
  After:  1×CQT ~1s  +  1×LFCC ~0.3s  +  4×ONNX ~0.07s   = ~1.6s  ✓

Feature shapes the DualBranchResNet18 ONNX model expects:
  LFCC : (1, 3, 20, 400)  — 3-channel (base + Δ + ΔΔ), 20 coefficients, 400 frames
  CQT  : (1, 1, 84, 125)  — 1-channel magnitude, 84 bins (7 octaves), 125 frames

Both targets assume 4-second audio at 16kHz (64000 samples):
  LFCC hop=160 → 64000/160=400 frames ✓
  CQT  hop=512 → 64000/512=125 frames ✓
"""

import logging
import numpy as np
import torch
import torchaudio
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import stft as scipy_stft   # top-level — avoid per-request cold-start

import librosa   # top-level — avoid per-call import overhead

logger = logging.getLogger(__name__)

# ── DualResNet constants (must match training notebook) ──────────────────────
DR_SR         = 16000
DR_CLIP_SEC   = 4.0
DR_CLIP_LEN   = 64000    # 4s × 16kHz

DR_N_LFCC     = 20       # base LFCC coefficients
DR_LFCC_T     = 400      # time frames for LFCC branch  (64000/160 = 400)
DR_HOP_LFCC   = 160      # 10ms hop

DR_N_BINS     = 84       # CQT bins = 7 octaves × 12 bins/octave
DR_CQT_T      = 125      # time frames for CQT branch   (64000/512 = 125)
DR_HOP_CQT    = 512      # ~32ms hop
DR_FMIN       = 32.7     # C1 = lowest CQT frequency

# ── Pre-computed CQT filterbank (built once at import, ~0.5s) ────────────────
# librosa.cqt() takes 20-30s per call because it builds the filter bank each time.
# We pre-compute the filterbank matrix once and keep it in memory.
# CQT is then just: STFT → |filterbank @ STFT_matrix|  (matrix multiply, ~0.1s)
#
# librosa.filters.constant_q returns a complex filterbank (n_bins, n_fft//2+1)
# Multiplying by STFT magnitude gives the constant-Q transform.

_CQT_N_FFT = 2048   # FFT size for fast STFT-based CQT approximation

# Pre-computed CQT frequency labels for XAI visualisation (Hz per bin)
_CQT_FREQS = librosa.cqt_frequencies(n_bins=DR_N_BINS, fmin=DR_FMIN, bins_per_octave=12)

# Build a Mel-like frequency mapping from FFT bins → CQT bins.
# Each CQT bin centre frequency (log-spaced): _CQT_FREQS[b]
# We map STFT frequencies to the nearest CQT bin using a triangular filterbank.
# This is an approximate CQT computed in ~0.05s instead of ~30s.

def _build_stft_to_cqt_matrix(n_fft: int, sr: int) -> np.ndarray:
    """
    Build (84, n_fft//2+1) triangular filterbank mapping STFT bins -> CQT bins.
    Fully vectorised - no Python loops.
    """
    n_stft  = n_fft // 2 + 1
    freqs   = np.linspace(0, sr / 2.0, n_stft, dtype=np.float64)  # (n_stft,)
    f_c     = _CQT_FREQS.astype(np.float64)                        # (84,)

    # Geometric lower/upper edges of each bin
    f_lo = np.empty(DR_N_BINS, dtype=np.float64)
    f_hi = np.empty(DR_N_BINS, dtype=np.float64)
    f_lo[0]    = f_c[0] / 2.0
    f_lo[1:]   = np.sqrt(f_c[:-1] * f_c[1:])
    f_hi[:-1]  = np.sqrt(f_c[:-1] * f_c[1:])
    f_hi[-1]   = f_c[-1] * np.sqrt(2.0)

    # Broadcast: freqs (1, n_stft), f_c/lo/hi (84, 1)
    f  = freqs[np.newaxis, :]   # (1,    n_stft)
    fc = f_c[:, np.newaxis]     # (84,   1)
    fl = f_lo[:, np.newaxis]    # (84,   1)
    fh = f_hi[:, np.newaxis]    # (84,   1)

    # Triangular weights: rising slope left of fc, falling slope right
    rise = (f - fl) / np.maximum(fc - fl, 1e-10)
    fall = (fh - f) / np.maximum(fh - fc, 1e-10)
    fb   = np.where(f < fc, rise, fall)
    fb   = np.clip(fb, 0.0, None).astype(np.float32)

    # L1 normalise each row
    row_sums = fb.sum(axis=1, keepdims=True)
    fb = np.divide(fb, row_sums, out=np.zeros_like(fb), where=row_sums > 0)
    return fb   # (84, n_stft)


try:
    _CQT_FILTERBANK = _build_stft_to_cqt_matrix(_CQT_N_FFT, DR_SR)  # (84, n_fft//2+1)
    _CQT_FB_OK = True
    logger.info(f"CQT triangular filterbank ready: {_CQT_FILTERBANK.shape}")
except Exception as _e:
    logger.warning(f"CQT filterbank build failed: {_e}")
    _CQT_FILTERBANK = None
    _CQT_FB_OK = False


# ── Shared LFCC transform (built once, reused) ───────────────────────────────
_DR_LFCC_TRANSFORM = torchaudio.transforms.LFCC(
    sample_rate=DR_SR,
    n_lfcc=DR_N_LFCC,
    speckwargs={
        "n_fft":       512,
        "hop_length":  DR_HOP_LFCC,
        "win_length":  400,
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# FAST PATH  (used by streaming inference)
# Compute CQT and LFCC ONCE on the full audio, then slice per window.
# ─────────────────────────────────────────────────────────────────────────────

def compute_cqt_full(audio_np: np.ndarray) -> np.ndarray:
    """
    Compute CQT on the ENTIRE audio signal — FAST filterbank approach.

    Uses pre-computed filterbank + scipy STFT:
      Old librosa.cqt: ~20-30s for 10s audio
      New filterbank  : ~0.05-0.2s for 10s audio  (100-600x faster)

    Returns:
        cqt_db : (84, T_full)  -- log-amplitude CQT of the entire signal
    """
    if _CQT_FB_OK and _CQT_FILTERBANK is not None:
        # Fast path: STFT + filterbank matrix multiply (scipy_stft imported at module level)
        _, _, Zxx = scipy_stft(
            audio_np.astype(np.float32),
            fs=DR_SR,
            nperseg=_CQT_N_FFT,
            noverlap=_CQT_N_FFT - DR_HOP_CQT,
            nfft=_CQT_N_FFT,
            boundary=None,
            padded=False,
        )  # Zxx: (n_fft//2+1, T)
        stft_mag = np.abs(Zxx).astype(np.float32)   # (n_fft//2+1, T)
        cqt = _CQT_FILTERBANK @ stft_mag             # (84, T)
    else:
        # Fallback: slow librosa CQT
        cqt = np.abs(librosa.cqt(
            audio_np.astype(np.float32),
            sr=DR_SR, hop_length=DR_HOP_CQT, fmin=DR_FMIN,
            n_bins=DR_N_BINS, bins_per_octave=12,
        ))

    cqt_db = librosa.amplitude_to_db(cqt + 1e-10, ref=np.max)
    return cqt_db.astype(np.float32)   # (84, T_full)




def compute_lfcc_full(audio_np: np.ndarray) -> torch.Tensor:
    """
    Compute LFCC (base + Δ + ΔΔ) on the ENTIRE audio signal (one call).

    Computing deltas on the full signal gives correct boundary values for all
    interior windows — no boundary artefacts from per-window computation.

    Returns:
        feat : (3, 20, T_full)  — 3 channels, 20 coefficients, T_full frames
    """
    wf = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)  # (1, T)
    with torch.no_grad():
        base = _DR_LFCC_TRANSFORM(wf).squeeze(0)      # (20, T_full)
        d1   = torchaudio.functional.compute_deltas(base)
        d2   = torchaudio.functional.compute_deltas(d1)
        feat = torch.stack([base, d1, d2], dim=0)     # (3, 20, T_full)
    return feat   # (3, 20, T_full)


def slice_window_cqt(cqt_full: np.ndarray, start_s: float) -> torch.Tensor:
    """
    Slice a 4-second CQT window from the pre-computed full-audio CQT and
    apply per-window z-score normalisation (matches training behaviour).

    Args:
        cqt_full : (84, T_full) — log-amp CQT of full audio
        start_s  : start time of this window in seconds

    Returns:
        (1, 84, 125) — model-ready CQT tensor for this window
    """
    start_frame = int(start_s * DR_SR / DR_HOP_CQT)
    end_frame   = start_frame + DR_CQT_T
    t_full      = cqt_full.shape[-1]

    if end_frame <= t_full:
        win = cqt_full[:, start_frame:end_frame].copy()    # (84, 125)
    else:
        # Pad if at the very end of the audio
        win = np.pad(cqt_full[:, start_frame:], ((0, 0), (0, end_frame - t_full)), mode="edge")

    # Per-window z-score (matches training)
    mu  = win.mean()
    sig = win.std() + 1e-8
    win = (win - mu) / sig

    return torch.tensor(win, dtype=torch.float32).unsqueeze(0)  # (1, 84, 125)


def slice_window_lfcc(lfcc_full: torch.Tensor, start_s: float) -> torch.Tensor:
    """
    Slice a 4-second LFCC window from the pre-computed full-audio LFCC and
    apply per-channel z-score normalisation (matches training behaviour).

    Args:
        lfcc_full : (3, 20, T_full) — full-audio LFCC (3 channels)
        start_s   : start time of this window in seconds

    Returns:
        (3, 20, 400) — model-ready LFCC tensor for this window
    """
    start_frame = int(start_s * DR_SR / DR_HOP_LFCC)
    end_frame   = start_frame + DR_LFCC_T
    t_full      = lfcc_full.shape[-1]

    if end_frame <= t_full:
        win = lfcc_full[:, :, start_frame:end_frame].clone()   # (3, 20, 400)
    else:
        pad_len = end_frame - t_full
        win     = torch.nn.functional.pad(
            lfcc_full[:, :, start_frame:], (0, pad_len), mode="replicate"
        )

    # Per-channel z-score (matches training)
    for ch in range(3):
        m = win[ch].mean()
        s = win[ch].std() + 1e-8
        win[ch] = (win[ch] - m) / s

    return win.float()   # (3, 20, 400)


def compute_full_audio_features(audio_np: np.ndarray, sr: int = DR_SR) -> dict:
    """
    FAST PATH: Compute CQT and LFCC once on the full audio in parallel threads.

    Returns:
        {
            "cqt_full"  : np.ndarray (84, T_full)     — log-amp full CQT
            "lfcc_full" : torch.Tensor (3, 20, T_full) — full LFCC 3-channel
        }
    """
    if sr != DR_SR:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=DR_SR)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_cqt  = pool.submit(compute_cqt_full,  audio_np)
        fut_lfcc = pool.submit(compute_lfcc_full, audio_np)
        cqt_full  = fut_cqt.result()   # (84, T_full)
        lfcc_full = fut_lfcc.result()  # (3, 20, T_full)

    return {"cqt_full": cqt_full, "lfcc_full": lfcc_full}


def infer_window_fast(window_info: dict, cqt_full: np.ndarray,
                      lfcc_full: torch.Tensor, session) -> dict:
    """
    Run ONNX inference for one window using pre-computed full-audio features.
    Feature extraction is just array slicing — microseconds.

    Args:
        window_info : {"chunk_id", "start_s", "end_s"}
        cqt_full    : (84, T_full) pre-computed full CQT
        lfcc_full   : (3, 20, T_full) pre-computed full LFCC
        session     : ONNX InferenceSession (already cached)

    Returns:
        inference result dict  (same schema as _infer_4s_window)
    """
    start_s = window_info["start_s"]

    # Slice features for this window — instant
    cqt_win  = slice_window_cqt(cqt_full, start_s)    # (1, 84, 125)
    lfcc_win = slice_window_lfcc(lfcc_full, start_s)   # (3, 20, 400)

    # Add batch dim for ONNX
    lfcc_np = lfcc_win.unsqueeze(0).numpy().astype(np.float32)   # (1, 3, 20, 400)
    cqt_np  = cqt_win.unsqueeze(0).numpy().astype(np.float32)    # (1, 1, 84, 125)

    logits = session.run(None, {"lfcc": lfcc_np, "cqt": cqt_np})[0]
    exp    = np.exp(logits[0] - logits[0].max())
    probs  = exp / exp.sum()
    prob   = float(probs[1])

    return {
        "type":       "chunk",
        "chunk_id":   window_info["chunk_id"],
        "start_s":    window_info["start_s"],
        "end_s":      window_info["end_s"],
        "spoof_prob": round(prob * 100, 2),
        "verdict":    "FAKE" if prob >= 0.5 else "REAL",
        # for XAI: pass the worst-window features if needed
        "_cqt_raw":   cqt_win.squeeze(0).numpy(),   # (84, 125)
        "_lfcc_raw":  lfcc_win,                      # (3, 20, 400)
    }


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY PATH  (kept for backward compatibility / unit tests)
# ─────────────────────────────────────────────────────────────────────────────

def _fix_len(x: torch.Tensor, target_t: int) -> torch.Tensor:
    """Pad (repeat) or trim last dimension to exactly target_t."""
    t = x.shape[-1]
    if t < target_t:
        reps = (target_t // t) + 1
        x = x.repeat(*([1] * (x.dim() - 1)), reps)[..., :target_t]
    elif t > target_t:
        x = x[..., :target_t]
    return x


def extract_lfcc_3ch(audio_4s: np.ndarray) -> torch.Tensor:
    """Legacy: per-window LFCC extraction. Returns (3, 20, 400)."""
    wf = torch.tensor(audio_4s, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        base = _DR_LFCC_TRANSFORM(wf).squeeze(0)
        d1   = torchaudio.functional.compute_deltas(base)
        d2   = torchaudio.functional.compute_deltas(d1)
        base = _fix_len(base, DR_LFCC_T)
        d1   = _fix_len(d1,   DR_LFCC_T)
        d2   = _fix_len(d2,   DR_LFCC_T)
        feat = torch.stack([base, d1, d2], dim=0)
        for ch in range(3):
            m = feat[ch].mean()
            s = feat[ch].std() + 1e-8
            feat[ch] = (feat[ch] - m) / s
    return feat.float()


def extract_cqt(audio_4s: np.ndarray) -> torch.Tensor:
    """Legacy: per-window CQT extraction. Returns (1, 84, 125)."""
    cqt = np.abs(librosa.cqt(
        audio_4s.astype(np.float32),
        sr=DR_SR, hop_length=DR_HOP_CQT, fmin=DR_FMIN,
        n_bins=DR_N_BINS, bins_per_octave=12, res_type="fft",
    ))
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    t = cqt_db.shape[-1]
    if t < DR_CQT_T:
        cqt_db = np.pad(cqt_db, ((0, 0), (0, DR_CQT_T - t)), mode="edge")
    elif t > DR_CQT_T:
        cqt_db = cqt_db[:, :DR_CQT_T]
    mu  = cqt_db.mean()
    sig = cqt_db.std() + 1e-8
    return torch.tensor((cqt_db - mu) / sig, dtype=torch.float32).unsqueeze(0)


def prepare_4s_audio(audio_np: np.ndarray, sr: int = DR_SR) -> np.ndarray:
    """Resample (if needed) and pad/trim to exactly 4 seconds (64000 samples)."""
    if sr != DR_SR:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=DR_SR)
    if len(audio_np) < DR_CLIP_LEN:
        audio_np = np.pad(audio_np, (0, DR_CLIP_LEN - len(audio_np)), mode="wrap")
    elif len(audio_np) > DR_CLIP_LEN:
        audio_np = audio_np[:DR_CLIP_LEN]
    return audio_np.astype(np.float32)


def extract_dual_resnet_features(audio_4s_np: np.ndarray, sr: int = DR_SR) -> dict:
    """Legacy: per-window feature extraction (sequential)."""
    audio_4s = prepare_4s_audio(audio_4s_np, sr)
    lfcc_3ch = extract_lfcc_3ch(audio_4s)
    cqt_1ch  = extract_cqt(audio_4s)
    return {
        "lfcc":      lfcc_3ch.unsqueeze(0),
        "cqt":       cqt_1ch.unsqueeze(0),
        "lfcc_raw":  lfcc_3ch,
        "cqt_raw":   cqt_1ch.squeeze(0).numpy(),
        "cqt_freqs": _CQT_FREQS,
    }


def extract_dual_resnet_features_parallel(audio_4s_np: np.ndarray,
                                          sr: int = DR_SR) -> dict:
    """Legacy: per-window feature extraction (parallel LFCC+CQT threads)."""
    audio_4s = prepare_4s_audio(audio_4s_np, sr)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_lfcc = pool.submit(extract_lfcc_3ch, audio_4s)
        fut_cqt  = pool.submit(extract_cqt,      audio_4s)
        lfcc_3ch = fut_lfcc.result()
        cqt_1ch  = fut_cqt.result()
    return {
        "lfcc":      lfcc_3ch.unsqueeze(0),
        "cqt":       cqt_1ch.unsqueeze(0),
        "lfcc_raw":  lfcc_3ch,
        "cqt_raw":   cqt_1ch.squeeze(0).numpy(),
        "cqt_freqs": _CQT_FREQS,
    }


def chunk_audio_4s(audio_np: np.ndarray, sr: int = DR_SR,
                   stride_s: float = 2.0) -> list:
    """
    Slide a 4-second window with stride_s steps over full audio.
    Returns list of dicts: [{"chunk_id", "start_s", "end_s", "audio"}]
    Note: in the FAST PATH, only "chunk_id", "start_s", "end_s" are used.
    """
    if sr != DR_SR:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=DR_SR)

    stride_samples = int(stride_s * DR_SR)
    window_samples = DR_CLIP_LEN
    total          = len(audio_np)
    chunks         = []

    start    = 0
    chunk_id = 0
    while start + window_samples <= total:
        chunks.append({
            "chunk_id": chunk_id,
            "start_s":  round(start / DR_SR, 2),
            "end_s":    round((start + window_samples) / DR_SR, 2),
            "audio":    audio_np[start: start + window_samples].copy().astype(np.float32),
        })
        start    += stride_samples
        chunk_id += 1

    # If audio shorter than 4s, add one padded chunk
    if not chunks:
        padded = prepare_4s_audio(audio_np, DR_SR)
        chunks.append({
            "chunk_id": 0,
            "start_s":  0.0,
            "end_s":    min(len(audio_np) / DR_SR, DR_CLIP_SEC),
            "audio":    padded,
        })

    return chunks
