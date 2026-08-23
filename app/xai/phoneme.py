"""
Whisper Phoneme Aligner — Performs transcription, word alignment,
and maps findings to specific speech segments and phonemes.
"""

import os
import logging
import numpy as np
import whisper
from app.config import settings

logger = logging.getLogger(__name__)

# Global model cache to avoid re-loading on every API call
_whisper_model = None

_CMU_DICTIONARY = {
    "hello": ["HH", "AH", "L", "OW"],
    "voice": ["V", "OY", "S"],
    "guard": ["G", "AA", "R", "D"],
    "deepfake": ["D", "IY", "P", "F", "EY", "K"],
    "test": ["T", "EH", "S", "T"],
    "audio": ["AO", "D", "IY", "OW"],
    "synthetic": ["S", "IH", "N", "TH", "EH", "T", "IH", "K"],
    "human": ["HH", "Y", "UW", "M", "AH", "N"],
    "fake": ["F", "EY", "K"],
    "real": ["R", "IY", "L"],
    "eleven": ["IH", "L", "EH", "V", "AH", "N"],
    "labs": ["L", "AE", "B", "Z"],
    "speech": ["S", "P", "IY", "CH"],
    "synthesis": ["S", "IH", "N", "TH", "AH", "S", "AH", "S"],
    "system": ["S", "IH", "S", "T", "AH", "M"],
    "security": ["S", "IH", "K", "Y", "UH", "R", "AH", "T", "IY"],
    "banking": ["B", "AE", "NG", "K", "IH", "NG"],
    "verification": ["V", "EH", "R", "AH", "F", "AH", "K", "EY", "SH", "AH", "N"],
}


def word_to_phonemes(word: str) -> list:
    """Convert word to approximate ARPAbet phonemes (offline fallback)."""
    w = word.lower().strip(".,?!;:-_\"'()[]")
    if w in _CMU_DICTIONARY:
        return _CMU_DICTIONARY[w]
    
    # Heuristic letter-to-sound rules
    vowels = "aeiouy"
    phonemes = []
    i = 0
    while i < len(w):
        ch = w[i]
        if ch == 'c' and i + 1 < len(w) and w[i+1] == 'h':
            phonemes.append("CH")
            i += 2
        elif ch == 's' and i + 1 < len(w) and w[i+1] == 'h':
            phonemes.append("SH")
            i += 2
        elif ch == 't' and i + 1 < len(w) and w[i+1] == 'h':
            phonemes.append("TH")
            i += 2
        elif ch == 'p' and i + 1 < len(w) and w[i+1] == 'h':
            phonemes.append("F")
            i += 2
        elif ch == 'n' and i + 1 < len(w) and w[i+1] == 'g':
            phonemes.append("NG")
            i += 2
        elif ch in vowels:
            if ch == 'a': phonemes.append("AE")
            elif ch == 'e': phonemes.append("EH")
            elif ch == 'i': phonemes.append("IH")
            elif ch == 'o': phonemes.append("OW")
            elif ch == 'u': phonemes.append("AH")
            elif ch == 'y': phonemes.append("IY")
            i += 1
        else:
            phonemes.append(ch.upper())
            i += 1
    return phonemes


def get_whisper_model():
    """Load and cache whisper-tiny model."""
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper-tiny model (cached)...")
        # Load tiny model (39M params, very lightweight)
        _whisper_model = whisper.load_model("tiny")
    return _whisper_model


def align_findings_to_phonemes(audio_np: np.ndarray, findings: list) -> dict:
    """
    Transcribe audio, extract word timestamps, and map findings to speech text.
    """
    try:
        model = get_whisper_model()
        
        # Ensure audio is float32 and normalized in [-1, 1]
        audio = audio_np.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
            
        logger.info("Transcribing audio for phonetic alignment...")
        result = model.transcribe(audio, word_timestamps=True)
        
        words = []
        transcript = result.get("text", "").strip()
        
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                word_clean = w["word"].strip()
                words.append({
                    "word": word_clean,
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                    "prob": round(w["probability"], 3),
                    "phonemes": word_to_phonemes(word_clean)
                })
                
        # Align findings to words/phonemes
        for f in findings:
            time_lo, time_hi = f["time_range"]
            matched_words = []
            matched_phonemes = []
            
            for w in words:
                overlap = max(w["start"], time_lo) < min(w["end"], time_hi)
                if overlap:
                    matched_words.append(w["word"])
                    matched_phonemes.extend(w["phonemes"])
                    
            if matched_words:
                f["phoneme_match"] = f"{'/'.join(matched_words)} ({'-'.join(matched_phonemes)})"
            else:
                f["phoneme_match"] = "N/A (silence/unvoiced)"
                
        return {
            "transcript": transcript,
            "words": words,
        }
    except Exception as e:
        logger.error(f"Whisper phoneme alignment failed: {e}", exc_info=True)
        return {
            "transcript": "Whisper transcription failed.",
            "words": [],
        }
