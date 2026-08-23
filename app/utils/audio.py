"""
Audio Utilities — Load, decode, resample, normalize audio files.
Matches the training notebook's load_and_augment() preprocessing exactly.
"""

import numpy as np
import librosa
import soundfile as sf
import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


def load_audio(
    filepath: str,
    sr: int = None,
    clip_len: int = None,
    max_duration_s: float = None,
) -> np.ndarray:
    """
    Load any audio file and return a normalized float32 numpy array.

    Pipeline (matches training notebook exactly):
      1. Load audio (any format: wav, flac, mp3, ogg)
      2. Convert to mono
      3. Resample to target SR
      4. Normalize to [-1, 1]
      5. Clip/pad to exact clip_len samples

    Args:
        filepath: Path to audio file
        sr: Target sample rate (default: settings.SR = 16000)
        clip_len: Target number of samples (default: settings.CLIP_LEN = 32000)
        max_duration_s: Maximum duration to load (default: settings.MAX_AUDIO_DURATION_S)

    Returns:
        float32 numpy array of shape (clip_len,), normalized to [-1, 1]
    """
    sr = sr or settings.SR
    clip_len = clip_len or settings.CLIP_LEN
    max_duration_s = max_duration_s or settings.MAX_AUDIO_DURATION_S

    filepath = str(filepath)

    # Try soundfile first (fastest), fall back to librosa
    try:
        audio, orig_sr = sf.read(filepath, dtype='float32', always_2d=False)
    except Exception:
        try:
            audio, orig_sr = librosa.load(filepath, sr=None, mono=True)
        except Exception as e:
            logger.error(f"Failed to load audio: {filepath} — {e}")
            return np.zeros(clip_len, dtype=np.float32)

    # Convert to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=-1) if audio.shape[-1] < audio.shape[0] \
                else audio.mean(axis=0)

    # Resample to target SR
    if orig_sr != sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=sr)

    # Normalize to [-1, 1]
    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio = audio / peak

    # Clip to exactly clip_len samples (center crop for inference)
    if len(audio) < clip_len:
        audio = np.pad(audio, (0, clip_len - len(audio)))
    elif len(audio) > clip_len:
        # Center crop for inference (training uses random crop)
        start = (len(audio) - clip_len) // 2
        audio = audio[start: start + clip_len]

    return audio.astype(np.float32)


def load_audio_full(
    filepath: str,
    sr: int = None,
    max_duration_s: float = None,
) -> np.ndarray:
    """
    Load full audio without clipping (for XAI statistics analysis).

    Returns:
        float32 numpy array, normalized to [-1, 1], full length
    """
    sr = sr or settings.SR
    max_duration_s = max_duration_s or settings.MAX_AUDIO_DURATION_S

    filepath = str(filepath)

    try:
        audio, orig_sr = sf.read(filepath, dtype='float32', always_2d=False)
    except Exception:
        try:
            audio, orig_sr = librosa.load(filepath, sr=None, mono=True)
        except Exception as e:
            logger.error(f"Failed to load audio: {filepath} — {e}")
            return np.zeros(int(sr * 2), dtype=np.float32)

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=-1) if audio.shape[-1] < audio.shape[0] \
                else audio.mean(axis=0)

    # Resample
    if orig_sr != sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=sr)

    # Truncate to max duration
    max_samples = int(max_duration_s * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    # Normalize
    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio = audio / peak

    return audio.astype(np.float32)


def get_audio_info(filepath: str) -> dict:
    """Get audio file metadata without loading full audio."""
    filepath = str(filepath)
    try:
        info = sf.info(filepath)
        return {
            "duration_s": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
        }
    except Exception:
        try:
            audio, sr = librosa.load(filepath, sr=None, mono=True)
            return {
                "duration_s": len(audio) / sr,
                "sample_rate": sr,
                "channels": 1,
                "format": Path(filepath).suffix,
                "subtype": "unknown",
            }
        except Exception:
            return {
                "duration_s": 0,
                "sample_rate": 0,
                "channels": 0,
                "format": "unknown",
                "subtype": "unknown",
            }


def load_audio_chunks(
    filepath: str,
    sr: int = None,
    chunk_sec: float = None,
    overlap_sec: float = None,
    max_chunks: int = None,
) -> list:
    """
    Load full audio and split into overlapping chunks for sliding-window detection.

    Each chunk is exactly CLIP_LEN samples (2 seconds), padded if needed.
    Returns a list of dicts: [{audio_np, start_s, end_s, chunk_idx}, ...]

    Args:
        filepath: Path to audio file
        sr: Sample rate (default: settings.SR)
        chunk_sec: Duration of each chunk (default: settings.CHUNK_SEC)
        overlap_sec: Overlap between chunks (default: settings.CHUNK_OVERLAP_SEC)
        max_chunks: Maximum number of chunks (default: settings.MAX_CHUNKS)

    Returns:
        List of chunk dicts, each with:
            - audio_np: float32 array of shape (CLIP_LEN,)
            - start_s: Start time in seconds
            - end_s: End time in seconds
            - chunk_idx: 0-indexed chunk number
    """
    sr = sr or settings.SR
    chunk_sec = chunk_sec or settings.CHUNK_SEC
    overlap_sec = overlap_sec or settings.CHUNK_OVERLAP_SEC
    max_chunks = max_chunks or settings.MAX_CHUNKS

    # Load full audio
    audio_full = load_audio_full(filepath, sr=sr)
    total_samples = len(audio_full)
    total_duration = total_samples / sr

    chunk_samples = int(chunk_sec * sr)       # 32000
    hop_samples = int((chunk_sec - overlap_sec) * sr)  # 24000 (1.5s hop)

    if hop_samples <= 0:
        hop_samples = chunk_samples  # No overlap if misconfigured

    chunks = []
    idx = 0

    for start in range(0, total_samples, hop_samples):
        if idx >= max_chunks:
            logger.warning(f"Hit max_chunks={max_chunks}, stopping at {start/sr:.1f}s")
            break

        end = start + chunk_samples
        chunk = audio_full[start:end]

        # Pad if chunk is shorter than chunk_samples
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        chunks.append({
            "audio_np": chunk.astype(np.float32),
            "start_s": round(start / sr, 3),
            "end_s": round(min(end, total_samples) / sr, 3),
            "chunk_idx": idx,
        })
        idx += 1

    logger.info(
        f"Split {total_duration:.1f}s audio into {len(chunks)} chunks "
        f"({chunk_sec}s each, {overlap_sec}s overlap)"
    )
    return chunks

