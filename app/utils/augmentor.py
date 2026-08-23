"""
VoiceGuard Inference-Time Audio Augmentor
=========================================
Provides self-contained mathematical DSP simulations for:
  1. Room impulse response (reverb) using a synthetic exponentially decaying noise RIR.
  2. Ambient background noise (pink/white noise) mixed at selectable SNR.
  3. 4-stage telephony network degradation (PSTN, VoIP, VoLTE, AMR-WB).
No local database files (MUSAN/RIR) are required.
"""

import os
import random
import subprocess
import logging
import torch
import numpy as np
import torchaudio
import torchaudio.functional as TAF

logger = logging.getLogger(__name__)

TARGET_SR = 16000


def mix_at_snr(signal: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Mix signal and noise at the desired SNR:
    SNR_dB = 10 * log10(P_signal / P_noise)
    """
    sig_power = signal.pow(2).mean().clamp(min=1e-12)
    noise_power = noise.pow(2).mean().clamp(min=1e-12)
    desired_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    scale = torch.sqrt(desired_noise_power / noise_power)
    return signal + scale * noise


def add_synthetic_noise(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Generate filtered noise to simulate background environment and mix it."""
    # Generate white noise
    noise = torch.randn_like(waveform)
    
    # Simple low-pass filter to shape the noise to pink-ish noise (hiss/rumble)
    # y[n] = 0.5 * x[n] + 0.5 * y[n-1]
    noise_np = noise.squeeze().numpy()
    filtered_noise = np.zeros_like(noise_np)
    last = 0.0
    for i in range(len(noise_np)):
        val = 0.6 * noise_np[i] + 0.4 * last
        filtered_noise[i] = val
        last = val
        
    filtered_tensor = torch.from_numpy(filtered_noise).unsqueeze(0).float()
    return mix_at_snr(waveform, filtered_tensor, snr_db)


def apply_synthetic_reverb(waveform: torch.Tensor, sr: int = TARGET_SR) -> torch.Tensor:
    """
    Apply a synthetic Room Impulse Response (RIR) to simulate room acoustics.
    Uses Schroeder's model of decaying Gaussian noise.
    """
    try:
        # RT60: time to decay by 60 dB (10^-3 amplitude)
        rt60 = random.uniform(0.15, 0.45)
        alpha = 6.91 / rt60  # decay coefficient
        
        # Time array for RIR
        t = torch.arange(0, int(rt60 * sr), dtype=torch.float32) / sr
        envelope = torch.exp(-alpha * t)
        
        # Exponentially decaying noise as impulse response
        rir = torch.randn_like(envelope) * envelope
        
        # Peak normalize RIR
        rir = rir / (rir.abs().max() + 1e-8)
        
        # Frequency-domain convolution
        time_len = waveform.shape[-1]
        rir_len = rir.shape[-1]
        n = time_len + rir_len - 1
        
        waveform_fft = torch.fft.rfft(waveform, n=n)
        rir_tensor = rir.unsqueeze(0)
        rir_fft = torch.fft.rfft(rir_tensor, n=n)
        
        out = torch.fft.irfft(waveform_fft * rir_fft, n=n)
        out = out[..., :time_len]
        
        # Restore peak volume level
        orig_peak = waveform.abs().max()
        out_peak = out.abs().max()
        if out_peak > 1e-8 and orig_peak > 1e-8:
            out = out / out_peak * orig_peak
            
        return out
    except Exception as exc:
        logger.warning(f"Synthetic reverb convolution failed: {exc}")
        return waveform


def apply_cng(waveform: torch.Tensor, sr: int = TARGET_SR,
              silence_threshold: float = 0.01) -> torch.Tensor:
    """Comfort Noise Generation (CNG) — scan silent parts and fill with comfort noise."""
    waveform = waveform.clone()
    n_samples = waveform.shape[1]
    window_sz = sr // 50  # 20ms analysis frame

    silent_starts = []
    for start in range(0, n_samples - window_sz, window_sz):
        rms = waveform[:, start:start + window_sz].pow(2).mean().sqrt().item()
        if rms < silence_threshold:
            silent_starts.append(start)

    if not silent_starts:
        return waveform

    # CNG on a random silent region
    cng_start = random.choice(silent_starts)
    cng_len = random.randint(int(0.020 * sr), min(int(0.120 * sr), n_samples - cng_start))
    
    local_rms = waveform[:, cng_start:cng_start + window_sz].pow(2).mean().sqrt().item()
    cng_std = max(local_rms, 1e-5)
    cng_noise = torch.randn(1, cng_len) * cng_std * 0.4
    waveform[:, cng_start:cng_start + cng_len] = cng_noise
    return waveform


def apply_burst_packet_loss(waveform: torch.Tensor, sr: int = TARGET_SR) -> torch.Tensor:
    """Simulate packet drop and concealment (stutter, comfort noise fill, or fade out/in)."""
    waveform = waveform.clone()
    n_samples = waveform.shape[1]

    n_drops = random.randint(1, 3)
    drop_min = int(0.020 * sr)  # 20 ms
    drop_max = int(0.080 * sr)  # 80 ms

    for _ in range(n_drops):
        drop_len = random.randint(drop_min, drop_max)
        plc_type = random.choice(['stutter', 'comfort', 'fade'])

        if plc_type == 'stutter':
            if n_samples > drop_len * 2:
                start = random.randint(drop_len, n_samples - drop_len)
                waveform[:, start:start + drop_len] = waveform[:, start - drop_len:start].clone()

        elif plc_type == 'comfort':
            history_len = 320  # 20ms at 16kHz
            if n_samples > drop_len + history_len:
                start = random.randint(history_len, n_samples - drop_len)
                local_power = waveform[:, start - history_len:start].pow(2).mean().sqrt()
                comfort = torch.randn(1, drop_len) * local_power * 0.3
                waveform[:, start:start + drop_len] = comfort

        elif plc_type == 'fade':
            start = random.randint(0, max(0, n_samples - drop_len - 1))
            actual_len = min(drop_len, n_samples - start)
            half = actual_len // 2
            rem = actual_len - half
            fade_out = torch.linspace(1.0, 0.0, half)
            fade_in = torch.linspace(0.0, 1.0, rem)
            envelope = torch.cat([fade_out, fade_in])
            waveform[:, start:start + actual_len] *= envelope

    return waveform


def apply_amr_wb_ffmpeg(waveform: torch.Tensor, sr: int = TARGET_SR,
                        bitrate: str = '12.65k') -> torch.Tensor:
    """Simulate AMR-WB CELP codec using system ffmpeg binary."""
    waveform_1d = waveform.squeeze()
    audio_np = (waveform_1d.numpy() * 32767.0).astype(np.int16)

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 's16le', '-ar', str(sr), '-ac', '1', '-i', 'pipe:0',
        '-c:a', 'amr_wb', '-b:a', bitrate,
        '-f', 's16le', '-ar', str(sr), '-ac', '1', 'pipe:1'
    ]
    try:
        proc = subprocess.Popen(cmd,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out_bytes, err_bytes = proc.communicate(input=audio_np.tobytes(), timeout=5)

        if proc.returncode != 0 or not out_bytes:
            return None

        reconstructed = np.frombuffer(out_bytes, dtype=np.int16).copy().astype(np.float32) / 32767.0
        min_len = min(waveform_1d.shape[0], reconstructed.shape[0])
        return torch.from_numpy(reconstructed[:min_len]).unsqueeze(0)
    except Exception:
        return None


def apply_telephony(waveform: torch.Tensor, sr: int = TARGET_SR,
                    mode: str = 'random') -> tuple:
    """4-stage telephony network simulation."""
    waveform = waveform.clone()
    
    if mode == 'random':
        mode = random.choice(['pstn', 'voip', 'volte', 'amr_wb'])
    resolved_mode = mode

    try:
        if mode == 'pstn':
            # Analog line hiss
            analog_hiss = torch.randn_like(waveform) * 0.003
            waveform = waveform + analog_hiss
            # Bandpass filtering
            waveform = TAF.highpass_biquad(waveform, sr, cutoff_freq=300.0)
            waveform = TAF.lowpass_biquad(waveform, sr, cutoff_freq=3400.0)
            # 8-bit µ-law quantization
            encoded = TAF.mu_law_encoding(waveform, quantization_channels=256)
            waveform = TAF.mu_law_decoding(encoded, quantization_channels=256)
            sat_prob = 0.45

        elif mode == 'voip':
            # Pre-filter hiss
            digital_hiss = torch.randn_like(waveform) * 0.001
            waveform = waveform + digital_hiss
            # Wideband filtering
            waveform = TAF.highpass_biquad(waveform, sr, cutoff_freq=50.0)
            waveform = TAF.lowpass_biquad(waveform, sr, cutoff_freq=7000.0)
            # Channel packet loss simulation
            if random.random() < 0.5:
                waveform = apply_burst_packet_loss(waveform, sr)
            # 9-bit µ-law quantization
            encoded = TAF.mu_law_encoding(waveform, quantization_channels=512)
            waveform = TAF.mu_law_decoding(encoded, quantization_channels=512)
            sat_prob = 0.25

        elif mode == 'volte':
            # Brickwall filter
            waveform = TAF.lowpass_biquad(waveform, sr, cutoff_freq=7500.0)
            # Comfort noise generation during silences
            if random.random() < 0.5:
                waveform = apply_cng(waveform, sr)
            # Packet loss
            if random.random() < 0.3:
                waveform = apply_burst_packet_loss(waveform, sr)
            # 10-bit µ-law quantization
            encoded = TAF.mu_law_encoding(waveform, quantization_channels=1024)
            waveform = TAF.mu_law_decoding(encoded, quantization_channels=1024)
            sat_prob = 0.10

        elif mode == 'amr_wb':
            # Bandpass
            waveform = TAF.highpass_biquad(waveform, sr, cutoff_freq=50.0)
            waveform = TAF.lowpass_biquad(waveform, sr, cutoff_freq=7000.0)
            # Pre-codec noise
            pre_noise = torch.randn_like(waveform) * 0.0005
            waveform = waveform + pre_noise
            
            bitrates = ['6.6k', '8.85k', '12.65k', '15.85k', '23.05k']
            bitrate = random.choice(bitrates)
            result = apply_amr_wb_ffmpeg(waveform, sr, bitrate=bitrate)

            if result is None:
                # Fallback to VoLTE 10-bit µ-law proxy
                encoded = TAF.mu_law_encoding(waveform, quantization_channels=1024)
                waveform = TAF.mu_law_decoding(encoded, quantization_channels=1024)
                resolved_mode = 'amr_wb_to_volte_fallback'
            else:
                orig_len = waveform.shape[1]
                if result.shape[1] < orig_len:
                    result = torch.nn.functional.pad(result, (0, orig_len - result.shape[1]))
                else:
                    result = result[:, :orig_len]
                waveform = result
                resolved_mode = f'amr_wb@{bitrate}'

            # CNG & Packet loss
            if random.random() < 0.5:
                waveform = apply_cng(waveform, sr)
            if random.random() < 0.4:
                waveform = apply_burst_packet_loss(waveform, sr)
            sat_prob = 0.10
        else:
            raise ValueError(f"Unknown telephony mode: {mode}")

        # Tanh microphone saturation (overdriven analog clipping)
        if random.random() < sat_prob:
            gain = random.uniform(1.5, 3.0)
            waveform = torch.tanh(waveform * gain)

    except Exception as exc:
        logger.error(f"Telephony augmentation error ({mode}): {exc}")

    return waveform, resolved_mode


def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """Peak-normalize to [-1, 1] and hard clamp."""
    peak = waveform.abs().max()
    if peak > 1e-8:
        waveform = waveform / peak
    return torch.clamp(waveform, -1.0, 1.0)


def run_inference_augmentation(
    audio_np: np.ndarray,
    sr: int = TARGET_SR,
    telephony: bool = False,
    telephony_mode: str = "random",
    noise: bool = False,
    reverb: bool = False,
) -> tuple:
    """
    Converts full-audio numpy array to torch tensor, runs all selected augmentations,
    and returns (augmented_audio_np, resolved_aug_logs).
    """
    w = torch.from_numpy(audio_np).unsqueeze(0).float()
    logs = []

    # 1. Reverb (acoustic space)
    if reverb:
        w = apply_synthetic_reverb(w, sr=sr)
        logs.append("Simulated Room Reverberation")

    # 2. Noise (background environment)
    if noise:
        # standard SNR = 15-25 dB
        snr_db = random.uniform(15.0, 25.0)
        w = add_synthetic_noise(w, snr_db=snr_db)
        logs.append(f"Simulated Background Noise (SNR {snr_db:.1f} dB)")

    # 3. Telephony (cellular codec & network degradation)
    if telephony:
        w, mode_used = apply_telephony(w, sr=sr, mode=telephony_mode)
        logs.append(f"Simulated Telephony Degradation (mode={mode_used})")

    # 4. Final peak normalization
    w = normalize_waveform(w)
    
    augmented_np = w.squeeze(0).numpy().astype(np.float32)
    return augmented_np, logs
