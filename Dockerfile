# ── VoiceGuard AI — Hugging Face Spaces Backend ──────────────────────
#
# Build:  docker build -t voiceguard-backend .
# Run:    docker run -p 7860:7860 voiceguard-backend
#
# HF Spaces requirements:
#   - EXPOSE 7860  (mandatory)
#   - USER 1000    (mandatory — HF runs containers as non-root UID 1000)
# ─────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /code

# ── System Dependencies ───────────────────────────────────────────────
# ffmpeg      : needed by torchaudio for audio format decoding
# libsndfile1 : needed by soundfile / librosa — CRASH on startup without this
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
# These are stored in the HF Space repo via git-lfs
COPY MODELS/ ./MODELS/

# ── Environment — Production Overrides ────────────────────────────────
ENV ENVIRONMENT=production
ENV DEBUG=false
ENV HOST=0.0.0.0
ENV PORT=7860

# Explicit paths for Docker layout (overrides config.py relative-path logic)
ENV MODELS_DIR=/code/MODELS
ENV PDF_OUTPUT_DIR=/code/pdfs
ENV AUDIO_TEMP_DIR=/code/tmp_audio

# CORS: allow your Vercel frontend (also accepts * for dev/demo)
# To restrict: set CORS_ORIGINS=https://voiceguardfrontend.vercel.app in HF Secrets
ENV CORS_ORIGINS=*

# ── Runtime Directories ───────────────────────────────────────────────
# Must exist and be writable by UID 1000 (HF non-root user)
RUN mkdir -p /code/pdfs /code/tmp_audio /code/enrolled_voices && \
    chmod -R 777 /code

# ── HF Spaces: run as UID 1000 (mandatory) ────────────────────────────
USER 1000

# ── Port (mandatory on HF Spaces) ─────────────────────────────────────
EXPOSE 7860

# ── Start Server ──────────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--timeout-keep-alive", "300", \
     "--log-level", "info"]
