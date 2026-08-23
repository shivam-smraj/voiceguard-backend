---
title: VoiceGuard AI Backend
emoji: 🔐
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Audio deepfake detection API - LCNN ensemble + XAI forensics
---

# VoiceGuard AI — Backend API

Production-grade audio deepfake / voice-cloning detection API for banking and call-centre fraud prevention.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/health` | Model status check |
| POST | `/api/v1/upload` | Upload audio → get `audit_id` for streaming |
| GET | `/api/v1/stream/{audit_id}` | SSE stream of progressive analysis results |
| POST | `/api/v1/analyze` | Full blocking analysis (single request) |
| WS | `/ws/live` | Real-time WebSocket live mic analysis |
| GET | `/api/v1/report/{id}/pdf` | Download forensic PDF report |
| POST | `/api/v1/enroll` | Register speaker voiceprint |
| GET | `/docs` | Interactive API docs (Swagger UI) |

## Models

- **LCNN-MFCC** — EER 1.2% — weight 0.55
- **LCNN-LFCC** — EER 1.74% — weight 0.45
- **AASIST-L** (ONNX) — raw waveform detector

## Frontend

Deployed at: https://voiceguardfrontend.vercel.app
