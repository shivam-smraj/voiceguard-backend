"""
Spectrogram & Waveform Renderers — Visual forensic evidence images.

Generates:
  1. Mel spectrogram PNG (frequency × time)
  2. Audio waveform PNG (amplitude × time)
  3. Biometric gauge chart PNG (jitter/shimmer/HNR/F0 radar)
  4. F0 contour plot PNG (pitch track over time)
"""

import io
import base64
import logging
import numpy as np

logger = logging.getLogger(__name__)


def render_spectrogram(audio_np: np.ndarray, sr: int = 16000, title: str = "Mel Spectrogram") -> str:
    """
    Render mel spectrogram as base64 PNG.

    Args:
        audio_np: float32 audio array
        sr: sample rate
        title: plot title

    Returns:
        base64-encoded PNG string
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=120)
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')

    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=audio_np, sr=sr, n_mels=128, n_fft=512, hop_length=160)
    S_dB = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(
        S_dB, sr=sr, hop_length=160, x_axis='s', y_axis='mel',
        ax=ax, cmap='magma'
    )
    fig.colorbar(img, ax=ax, format='%+2.0f dB', pad=0.02)
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('Time (seconds)', color='#aaaaaa', fontsize=9)
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    ax.set_ylabel('Frequency (Hz)', color='#aaaaaa', fontsize=9)

    for spine in ax.spines.values():
        spine.set_color('#333333')

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def render_waveform(audio_np: np.ndarray, sr: int = 16000, title: str = "Audio Waveform") -> str:
    """Render audio waveform as base64 PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 2.5), dpi=120)
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')

    t = np.arange(len(audio_np)) / sr
    ax.plot(t, audio_np, color='#3498db', linewidth=0.4, alpha=0.9)
    ax.fill_between(t, audio_np, alpha=0.15, color='#3498db')

    # RMS envelope
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    rms = []
    rms_t = []
    for i in range(0, len(audio_np) - frame_len, hop):
        rms.append(np.sqrt(np.mean(audio_np[i:i+frame_len]**2)))
        rms_t.append((i + frame_len // 2) / sr)
    if rms:
        ax.plot(rms_t, rms, color='#e74c3c', linewidth=1.5, alpha=0.8, label='RMS Envelope')
        ax.plot(rms_t, [-r for r in rms], color='#e74c3c', linewidth=1.5, alpha=0.8)

    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel('Time (s)', color='#aaaaaa', fontsize=9)
    ax.set_ylabel('Amplitude', color='#aaaaaa', fontsize=9)
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    ax.set_xlim(0, len(audio_np) / sr)
    ax.legend(loc='upper right', fontsize=7, facecolor='#1a1a2e', edgecolor='#333',
              labelcolor='#aaaaaa')

    for spine in ax.spines.values():
        spine.set_color('#333333')

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def render_f0_contour(audio_np: np.ndarray, sr: int = 16000) -> str:
    """Render F0 (pitch) contour as base64 PNG — shows pitch stability."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import librosa

    fig, ax = plt.subplots(1, 1, figsize=(10, 3), dpi=120)
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')

    # Extract F0 using pYIN (use hop_length consistent with times_like)
    hop_length = 512  # pYIN default
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio_np, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=sr, hop_length=hop_length,
    )

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # Plot all F0 (faint)
    ax.plot(times, f0, color='#555555', linewidth=1, alpha=0.5, label='All F0')

    # Highlight voiced segments
    voiced_f0 = np.where(voiced_flag, f0, np.nan)
    ax.plot(times, voiced_f0, color='#2ecc71', linewidth=2, alpha=0.9, label='Voiced F0')

    # Mark unvoiced regions
    unvoiced_mask = ~voiced_flag & ~np.isnan(f0)
    if np.any(unvoiced_mask):
        ax.scatter(times[unvoiced_mask], f0[unvoiced_mask], color='#e74c3c',
                   s=8, alpha=0.6, zorder=5, label='Unvoiced')

    # Human F0 range band
    valid_f0 = f0[~np.isnan(f0)]
    if len(valid_f0) > 0:
        mean_f0 = np.nanmean(valid_f0)
        ax.axhline(y=mean_f0, color='#f39c12', linestyle='--', linewidth=1, alpha=0.6,
                   label=f'Mean F0: {mean_f0:.0f} Hz')

    ax.set_title('Fundamental Frequency (F0) Contour', color='white', fontsize=11,
                 fontweight='bold', pad=6)
    ax.set_xlabel('Time (s)', color='#aaaaaa', fontsize=9)
    ax.set_ylabel('Frequency (Hz)', color='#aaaaaa', fontsize=9)
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    ax.set_xlim(0, len(audio_np) / sr)
    ax.legend(loc='upper right', fontsize=7, facecolor='#1a1a2e', edgecolor='#333',
              labelcolor='#aaaaaa')

    for spine in ax.spines.values():
        spine.set_color('#333333')

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def render_biometric_radar(biometrics: dict) -> str:
    """Render biometric measurements as a radar/spider chart (6 axes)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    categories = ['Jitter', 'Shimmer', 'HNR', 'F0 Var', 'Pause E.', 'Form. F2']

    # Normalize: all axes → >1.0 means suspicious, <1.0 means normal
    # For jitter/shimmer: LOW is suspicious → invert (baseline/measured)
    jitter = biometrics.get('jitter_pct', 0.48)
    shimmer = biometrics.get('shimmer_db', 0.35)
    hnr = biometrics.get('hnr_db', 18.2)
    f0_var = biometrics.get('f0_variance_hz', 2.4)
    pause_rms = biometrics.get('pause_energy', {}).get('min_rms', 0.001)
    formant_vel = biometrics.get('formant_velocity', {}).get('max_hz_per_frame', 0)

    # Jitter: low = synthetic → show ratio as baseline/measured (>1 = suspicious)
    jitter_norm = min(2.0, 0.48 / max(jitter, 0.01)) if jitter < 0.48 else min(2.0, jitter / 0.48)
    # Shimmer: low = synthetic
    shimmer_norm = min(2.0, 0.35 / max(shimmer, 0.01)) if shimmer < 0.35 else min(2.0, shimmer / 0.35)
    # HNR: HIGH = synthetic (too clean) → show measured/baseline
    hnr_norm = min(2.0, hnr / 18.2)
    # F0: LOW = synthetic (too static) → invert
    f0_norm = min(2.0, 2.4 / max(f0_var, 0.01)) if f0_var < 2.4 else min(2.0, f0_var / 2.4)
    # Pause: LOW min_rms = digital silence → invert
    pause_norm = min(2.0, 0.001 / max(pause_rms, 1e-8)) if pause_rms < 0.001 else 0.5
    # Formant: HIGH velocity = inhuman speed
    formant_norm = min(2.0, formant_vel / 50.0) if formant_vel > 0 else 0.3

    values = [jitter_norm, shimmer_norm, hnr_norm, f0_norm, pause_norm, formant_norm]
    baselines = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    baselines += baselines[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=120, subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')

    # Human baseline ring (1.0 = normal)
    ax.plot(angles, baselines, color='#27ae60', linewidth=2, linestyle='--',
            alpha=0.7, label='Human Baseline (1.0x)')
    ax.fill(angles, baselines, color='#27ae60', alpha=0.08)

    # Measured values
    ax.plot(angles, values, color='#e74c3c', linewidth=2.5, marker='o',
            markersize=6, label='Measured')
    ax.fill(angles, values, color='#e74c3c', alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, color='#cccccc', fontweight='bold')
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(['0.5x', '1.0x', '1.5x', '2.0x'], fontsize=7, color='#888888')
    ax.set_ylim(0, 2.2)
    ax.set_title('Biometric Radar (6-test)\n>1.0 = Suspicious', color='white', fontsize=11,
                 fontweight='bold', pad=20, y=1.08)
    ax.legend(loc='upper right', fontsize=7, facecolor='#1a1a2e', edgecolor='#333',
              labelcolor='#cccccc', bbox_to_anchor=(1.15, 1.15))

    ax.spines['polar'].set_color('#333333')
    ax.grid(color='#333333', alpha=0.5)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

