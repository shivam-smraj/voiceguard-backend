# ── VoiceGuard AI — Production Backend ───────────────────────────────
#
# Compatible with:
#   - Railway.app  (PORT injected dynamically, runs as root)
#   - Hugging Face Spaces (PORT=7860, USER 1000)
#   - Local Docker: docker run -p 8000:8000 -e PORT=8000 voiceguard-backend
# ─────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /code

# ── System Dependencies ───────────────────────────────────────────────
# ffmpeg      : needed by torchaudio for audio format decoding
# libsndfile1 : needed by soundfile / librosa — CRASH without this
# libgomp1    : OpenMP threading used by ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# ── Python Dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application Code ──────────────────────────────────────────────────
COPY app/ ./app/

# ── Model Weights ─────────────────────────────────────────────────────
COPY MODELS/ ./MODELS/

# ── Environment — Production Overrides ────────────────────────────────
ENV ENVIRONMENT=production
ENV DEBUG=false
ENV HOST=0.0.0.0

# PORT: Railway injects $PORT dynamically; default 8000 for local/HF
# Set PORT=7860 in HF Spaces secrets, Railway sets it automatically
ENV PORT=8000

# Explicit paths (overrides config.py relative-path logic)
ENV MODELS_DIR=/code/MODELS
ENV PDF_OUTPUT_DIR=/code/pdfs
ENV AUDIO_TEMP_DIR=/code/tmp_audio

# CORS: restrict to your Vercel frontend in production
# Override via Railway/HF environment variables
ENV CORS_ORIGINS=*

# ── Runtime Directories ───────────────────────────────────────────────
RUN mkdir -p /code/pdfs /code/tmp_audio /code/enrolled_voices && \
    chmod -R 777 /code

# ── Port ──────────────────────────────────────────────────────────────
# EXPOSE is informational — actual port comes from $PORT env var
EXPOSE 8000

# ── Start Server ──────────────────────────────────────────────────────
# Uses shell form so $PORT is expanded at runtime
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 300 --log-level info
