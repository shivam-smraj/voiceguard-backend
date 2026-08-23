"""
CQT Visualization and Harmonic Integrity Metrics.

Provides:
  - CQT heatmap with Hz y-axis labels (base64 PNG)
  - Harmonic regularity, spectral flux, sub-harmonic metrics
  - Vocoder fingerprint heuristics from CQT patterns
"""

import io
import base64
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# CQT constants (must match dual_resnet_features.py)
DR_N_BINS       = 84
DR_FMIN         = 32.7
DR_HOP_CQT      = 512
DR_SR           = 16000


def render_cqt_heatmap(
    cqt_raw: np.ndarray,         # (84, 125) — log-amplitude
    cqt_freqs: np.ndarray,       # (84,) Hz labels
    gradcam: Optional[np.ndarray] = None,   # (84, 125) 0–1 attention
    title: str = "CQT Evidence Map",
) -> str:
    """
    Render a CQT spectrogram with Hz y-axis labels.
    Optionally overlays a Grad-CAM attention map in red.

    Returns base64-encoded PNG string.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, axes = plt.subplots(
        1, 2 if gradcam is not None else 1,
        figsize=(14 if gradcam is not None else 7, 5),
        dpi=120,
    )
    fig.patch.set_facecolor("#0d0d1a")

    if gradcam is None:
        axes = [axes]

    # ── Panel 1: CQT spectrogram ─────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0d0d1a")
    im = ax.imshow(
        cqt_raw,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    # Y-axis: Hz labels (sample every 12 bins = 1 octave)
    ytick_indices = list(range(0, DR_N_BINS, 12))
    ytick_labels  = [f"{int(cqt_freqs[i])} Hz" for i in ytick_indices]
    ax.set_yticks(ytick_indices)
    ax.set_yticklabels(ytick_labels, color="#aaaaaa", fontsize=8)
    ax.set_xlabel("Time Frame (~32ms each)", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Frequency (Hz, log scale)", color="#aaaaaa", fontsize=9)
    ax.set_title("CQT Spectrogram", color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#aaaaaa")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("dB (normalized)", color="#aaaaaa", fontsize=7)
    cbar.ax.tick_params(labelcolor="#aaaaaa", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#333333")

    # ── Panel 2: CQT + Grad-CAM overlay ─────────────────────────────
    if gradcam is not None:
        ax2 = axes[1]
        ax2.set_facecolor("#0d0d1a")
        # Show CQT as grayscale base
        ax2.imshow(cqt_raw, aspect="auto", origin="lower",
                   cmap="gray", alpha=0.6, interpolation="nearest")
        # Overlay Grad-CAM in red-yellow
        cam_resize = gradcam
        if cam_resize.shape != cqt_raw.shape:
            from PIL import Image
            pil = Image.fromarray((cam_resize * 255).astype(np.uint8))
            pil = pil.resize((cqt_raw.shape[1], cqt_raw.shape[0]),
                             Image.BILINEAR)
            cam_resize = np.array(pil) / 255.0

        ax2.imshow(cam_resize, aspect="auto", origin="lower",
                   cmap="hot", alpha=0.65, interpolation="nearest")
        ax2.set_yticks(ytick_indices)
        ax2.set_yticklabels(ytick_labels, color="#aaaaaa", fontsize=8)
        ax2.set_xlabel("Time Frame (~32ms each)", color="#aaaaaa", fontsize=9)
        ax2.set_title("CQT Artifact Evidence (Grad-CAM)", color="white",
                      fontsize=10, fontweight="bold")
        ax2.tick_params(colors="#aaaaaa")
        for sp in ax2.spines.values():
            sp.set_color("#333333")

    fig.suptitle(title, color="white", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def compute_harmonic_metrics(
    cqt_raw: np.ndarray,   # (84, 125) log-amplitude
    cqt_freqs: np.ndarray, # (84,) Hz
) -> dict:
    """
    Compute CQT-based harmonic integrity metrics.

    Returns a dict with scores, interpretations, and vocoder hints.
    """
    # ── 1. Spectral Flux (frame-to-frame CQT energy change) ──────────
    #    Human speech: high flux (dynamic, unpredictable)
    #    TTS speech:   low flux (over-smooth transitions)
    diff          = np.diff(cqt_raw, axis=1)          # (84, 124)
    spectral_flux = float(np.mean(np.abs(diff)))
    # Baseline: human ~0.15–0.25 (normalized dB/frame)
    flux_score = min(100, int((1 - min(spectral_flux / 0.20, 1)) * 100))
    flux_label = "MACHINE_SMOOTH" if spectral_flux < 0.05 else \
                 "SUSPICIOUS"     if spectral_flux < 0.10 else "NATURAL"

    # ── 2. Harmonic Regularity (energy variance across harmonic bins) ─
    #    Bins in CQT at 12 bins/octave: harmonics land at predictable indices.
    #    Natural: harmonic amplitude varies unpredictably (jitter)
    #    TTS:     harmonic amplitude is overly regular (machine rhythm)
    bin_energy_std = float(np.std(cqt_raw.mean(axis=1)))
    # Higher std = more spectral variation = more natural
    harm_regularity = min(100, int((1 - min(bin_energy_std / 3.0, 1)) * 100))
    harm_label = "TOO_REGULAR" if harm_regularity > 80 else \
                 "SUSPICIOUS"  if harm_regularity > 60 else "NATURAL"

    # ── 3. Sub-harmonic energy (below F0 region, bins 0–6 ≈ 32–65 Hz) ─
    #    GAN vocoders introduce sub-harmonic artifacts from upsampling errors
    sub_harm_energy   = float(cqt_raw[:6].mean())
    total_energy       = float(cqt_raw.mean())
    sub_harm_ratio     = abs(sub_harm_energy / (total_energy + 1e-8))
    sub_harm_anomaly   = sub_harm_ratio > 0.85  # unusually high

    # ── 4. High-frequency energy cutoff (codec bandwidth detection) ──
    #    Find first bin from top where energy drops sharply → codec cutoff
    bin_energies = cqt_raw.mean(axis=1)   # (84,)
    hf_bins  = bin_energies[60:]           # bins 60–84 ≈ 2–8 kHz
    lf_mean  = float(bin_energies[:60].mean())
    hf_mean  = float(hf_bins.mean())
    hf_ratio = abs(hf_mean / (lf_mean + 1e-8))

    # Estimate cutoff frequency
    cutoff_hz = None
    for i in range(83, 59, -1):
        if bin_energies[i] > lf_mean * 0.1:
            cutoff_hz = float(cqt_freqs[i])
            break

    # ── 5. Vocoder Fingerprint (heuristic) ───────────────────────────
    vocoder_hint = "Unknown"
    vocoder_conf = {}

    hifigan_score = 0
    bvgan_score   = 0
    encodec_score = 0
    wavernn_score = 0

    # HiFi-GAN: aliasing at 4–6 kHz (bins ~66–72), low sub-harmonic
    hifi_band = float(cqt_raw[66:72].mean()) if cqt_raw.shape[0] > 72 else 0
    if hifi_band > lf_mean * 0.3 and not sub_harm_anomaly:
        hifigan_score += 40
    if spectral_flux < 0.08:
        hifigan_score += 20
    if harm_regularity > 75:
        hifigan_score += 20

    # BigVGAN: more uniform HF energy (better anti-aliasing)
    hf_uniformity = float(np.std(hf_bins))
    if hf_uniformity < 0.5 and hf_mean > lf_mean * 0.15:
        bvgan_score += 35
    if spectral_flux < 0.12:
        bvgan_score += 20

    # EnCodec: hard cutoff visible, quantization steps
    if cutoff_hz and cutoff_hz < 4000:
        encodec_score += 50
    elif cutoff_hz and cutoff_hz < 6000:
        encodec_score += 25
    quantization_steps = float(np.std(np.diff(cqt_raw, axis=0)))
    if quantization_steps < 0.3:
        encodec_score += 20

    # WaveRNN: stuttering in time dimension
    time_energy_std = float(np.std(cqt_raw.mean(axis=0)))
    if time_energy_std > 2.0:
        wavernn_score += 30

    total_vc = hifigan_score + bvgan_score + encodec_score + wavernn_score
    if total_vc > 0:
        vocoder_conf = {
            "HiFi-GAN class":  round(hifigan_score / total_vc * 100),
            "BigVGAN class":   round(bvgan_score   / total_vc * 100),
            "EnCodec class":   round(encodec_score  / total_vc * 100),
            "WaveRNN class":   round(wavernn_score  / total_vc * 100),
        }
        vocoder_hint = max(vocoder_conf, key=vocoder_conf.get)

    # ── 6. Overall CQT Anomaly Score (0–100, higher = more suspicious) ─
    anomaly_score = min(100, int(
        flux_score * 0.35 +
        harm_regularity * 0.35 +
        (50 if sub_harm_anomaly else 0) * 0.15 +
        (100 if cutoff_hz and cutoff_hz < 5000 else 0) * 0.15
    ))

    return {
        # Spectral Flux
        "spectral_flux":         round(spectral_flux, 4),
        "spectral_flux_score":   flux_score,
        "spectral_flux_label":   flux_label,
        "flux_baseline":         "0.15–0.25 (natural speech)",

        # Harmonic Regularity
        "harmonic_regularity":   harm_regularity,
        "harmonic_label":        harm_label,
        "harm_baseline":         "<60/100 (natural speech)",

        # Sub-harmonic
        "sub_harmonic_anomaly":  sub_harm_anomaly,
        "sub_harmonic_ratio":    round(sub_harm_ratio, 4),

        # HF Cutoff
        "hf_energy_ratio":       round(hf_ratio, 4),
        "estimated_cutoff_hz":   int(cutoff_hz) if cutoff_hz else None,

        # Vocoder
        "vocoder_hint":          vocoder_hint,
        "vocoder_confidence":    vocoder_conf,

        # Overall
        "cqt_anomaly_score":     anomaly_score,
        "anomaly_label":         (
            "HIGH SUSPICION" if anomaly_score >= 70 else
            "MODERATE"       if anomaly_score >= 40 else
            "LOW"
        ),
    }
