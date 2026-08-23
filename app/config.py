"""
VoiceGuard AI — Configuration
All parameters match the training notebooks exactly.
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional
import os


class Settings(BaseSettings):
    # ─── App ──────────────────────────────────────────────────────────
    APP_NAME: str = "VoiceGuard AI"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"

    # ─── Paths ────────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).parent.parent
    PROJECT_ROOT: Path = BASE_DIR.parent.parent
    MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "MODELS"))
    PDF_OUTPUT_DIR: Path = Path(os.getenv("PDF_OUTPUT_DIR", BASE_DIR / "pdfs"))
    AUDIO_TEMP_DIR: Path = Path(os.getenv("AUDIO_TEMP_DIR", BASE_DIR / "tmp_audio"))

    # ─── Model Checkpoints ────────────────────────────────────────────
    CHECKPOINT_LCNN_MFCC: str = "LCNN_MFCC/lcnn_mfcc_best.pt"
    CHECKPOINT_LCNN_LFCC: str = "LCNN_LFCC/lcnn_lfcc_best.pt"
    CHECKPOINT_AASIST: str = "AASIST/aasist_l_simplified.onnx"

    # ─── Audio Feature Parameters (MUST match notebook exactly) ──────
    SR: int = 16000
    CLIP_SEC: float = 2.0
    CLIP_LEN: int = 32000          # SR * CLIP_SEC
    MAX_FRAMES: int = 200
    HOP_LEN: int = 160             # 10ms hop at 16kHz
    WIN_LEN: int = 400             # 25ms window
    N_FFT: int = 512

    # MFCC
    N_MFCC: int = 40
    N_MELS: int = 80
    MFCC_TOTAL: int = 120          # 40 * 3 (base + delta + delta-delta)

    # LFCC
    N_LFCC: int = 60
    LFCC_TOTAL: int = 180          # 60 * 3

    # AASIST raw waveform
    AASIST_CLIP_LEN: int = 64600   # ~4 seconds at 16kHz

    # ─── Model Performance (for display in reports) ───────────────────
    EER_LCNN_MFCC: float = 1.2
    EER_LCNN_LFCC: float = 1.74

    # ─── Ensemble Weights (2-model: proportional to inverse EER) ──────
    WEIGHT_LCNN_MFCC: float = 0.55
    WEIGHT_LCNN_LFCC: float = 0.45

    # ─── XAI Parameters ───────────────────────────────────────────────
    GRADCAM_TARGET_LAYER: int = 23   # features[23] = last BatchNorm2d in Block 4
    GRADCAM_THRESHOLD: float = 0.65
    GRADCAM_MIN_AREA: int = 15

    # ─── Biometric Baselines (IndicVoices-R population statistics) ────
    JITTER_BASELINE_PCT: float = 0.48
    SHIMMER_BASELINE_DB: float = 0.35
    HNR_BASELINE_DB: float = 18.2
    F0_VARIANCE_BASELINE_HZ: float = 2.4
    F2_VELOCITY_LIMIT: float = 50.0  # Hz per 10ms frame

    # ─── Processing Limits ────────────────────────────────────────────
    MAX_AUDIO_SIZE_MB: float = 50.0
    MAX_AUDIO_DURATION_S: float = 300.0    # 5 minutes max
    XAI_TIMEOUT_S: float = 60.0

    # ─── Sliding Window (Full-Length Analysis) ────────────────────────
    CHUNK_SEC: float = 2.0          # Each chunk = 2 seconds (matches training)
    CHUNK_OVERLAP_SEC: float = 0.5  # 0.5s overlap between chunks
    MAX_CHUNKS: int = 150           # Safety cap: 150 chunks = ~5 min audio
    CHUNK_SAMPLES: int = 32000      # CHUNK_SEC * SR

    # ─── Server ───────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", 8000))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def model_post_init(self, __context):
        """Create output directories on startup."""
        self.PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.AUDIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
