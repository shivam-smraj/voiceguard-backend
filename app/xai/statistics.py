"""
Audio Statistics Analyzer — Jitter, Shimmer, HNR, F0 variance.
Pure physics-based biometric analysis (no ML model needed).

These measurements detect machine-generated speech by checking if the
voice's micro-characteristics match human population baselines from
IndicVoices-R (Indian speaker dataset).
"""

import numpy as np
import librosa
from scipy.signal import hilbert, lfilter
from typing import Dict
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class AudioStatisticsAnalyzer:
    """
    Measures voice biometric markers that distinguish human from synthetic speech.

    Real human voice has natural variability in:
      - Jitter (pitch period perturbation): ~0.48% ± 0.21%
      - Shimmer (amplitude perturbation): ~0.35 dB ± 0.12 dB
      - HNR (harmonic-to-noise ratio): ~18.2 dB ± 4.1 dB
      - F0 variance: ~2.4 Hz standard deviation

    Synthetic voice tends to be "too perfect" — low jitter, low shimmer, high HNR.
    """

    def __init__(self):
        self.sr = settings.SR
        self.baselines = {
            "jitter_pct": settings.JITTER_BASELINE_PCT,
            "shimmer_db": settings.SHIMMER_BASELINE_DB,
            "hnr_db": settings.HNR_BASELINE_DB,
            "f0_variance_hz": settings.F0_VARIANCE_BASELINE_HZ,
        }

    def analyze(self, audio_np: np.ndarray) -> Dict:
        """
        Run all biometric analyses on audio.

        Args:
            audio_np: float32 array, mono, 16kHz, normalized to [-1,1]

        Returns:
            BiometricDrift dict with all measurements and status flags
        """
        try:
            # F0 (pitch) extraction using pYIN
            f0, voiced_flag, voiced_prob = librosa.pyin(
                audio_np,
                fmin=librosa.note_to_hz('C2'),   # ~65 Hz
                fmax=librosa.note_to_hz('C7'),   # ~2093 Hz
                sr=self.sr,
            )

            # Only use voiced frames
            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
            voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

            if len(voiced_f0) < 5:
                logger.warning("Too few voiced frames for biometric analysis")
                return self._empty_result()

        except Exception as e:
            logger.error(f"F0 extraction failed: {e}")
            return self._empty_result()

        # ── Jitter (pitch period perturbation) ────────────────────────
        jitter_pct = self._compute_jitter(voiced_f0)

        # ── F0 variance ───────────────────────────────────────────────
        f0_variance_hz = float(np.std(voiced_f0))

        # ── Shimmer (amplitude perturbation) ──────────────────────────
        shimmer_db = self._compute_shimmer(audio_np)

        # ── HNR (Harmonic-to-Noise Ratio) ─────────────────────────────
        hnr_db = self._compute_hnr(audio_np)

        # ── Pause energy analysis ──────────────────────────────────────
        pause_energy = self._compute_pause_energy(audio_np)

        # ── Formant F2 velocity ───────────────────────────────────────
        formant_velocity = self._compute_formant_velocity(audio_np)

        # ── Status assessment ─────────────────────────────────────────
        jitter_z = (jitter_pct - self.baselines["jitter_pct"]) / 0.21
        jitter_deviation = abs(jitter_pct - self.baselines["jitter_pct"]) / self.baselines["jitter_pct"] * 100

        # F0 variance: human ~2.4 Hz std dev
        f0_status = "NORMAL"
        if f0_variance_hz < 0.8:
            f0_status = "TOO_STATIC"
        elif f0_variance_hz > 8.0:
            f0_status = "TOO_ERRATIC"

        # Jitter: human ~0.48% ± 0.21%
        # TOO LOW (<0.20%) = machine profile, TOO HIGH (>1.5%) = unstable/degraded
        jitter_status = "NORMAL"
        if jitter_pct < 0.20:
            jitter_status = "MACHINE_PROFILE"
        elif jitter_pct > 1.5:
            jitter_status = "EXCESS"

        # Shimmer: human ~0.35 dB ± 0.12 dB
        # TOO LOW (<0.15 dB) = linear grid, TOO HIGH (>1.0 dB) = degraded
        shimmer_status = "NORMAL"
        if shimmer_db < 0.15:
            shimmer_status = "LINEAR_GRID"
        elif shimmer_db > 1.0:
            shimmer_status = "EXCESS"

        # HNR: human ~18.2 dB ± 4.1 dB
        # TOO HIGH (>24 dB) = unnaturally clean, TOO LOW (<8 dB) = degraded
        hnr_status = "NORMAL"
        hnr_warning = None
        if hnr_db > 24.0:
            hnr_status = "TOO_CLEAN"
        elif hnr_db < 1.0:
            hnr_status = "TOO_NOISY"
            hnr_warning = (
                "HNR computation may be unreliable for this audio sample — "
                "possible background noise contamination or computation error. "
                "Manual review recommended."
            )
        elif hnr_db < 8.0:
            hnr_status = "TOO_NOISY"

        pause_status = "NORMAL"
        # Issue 12: Only flag DIGITAL_SILENCE if > 0.5% of frames have near-zero RMS
        pct_silent = pause_energy.get('pct_silent', 0)
        if pct_silent > 0.5 and pause_energy.get('suspicious', False):
            pause_status = "DIGITAL_SILENCE"

        formant_status = "NORMAL"
        if formant_velocity.get('suspicious', False):
            formant_status = "INHUMAN_SPEED"

        # ── Overall biometric score (0=human, 100=machine) ────────────
        # Deviation-based scoring: even "NORMAL" values contribute if
        # they deviate significantly from baseline.
        suspicious_count = sum([
            f0_status != "NORMAL",
            jitter_status != "NORMAL",
            shimmer_status != "NORMAL",
            hnr_status != "NORMAL",
            pause_status != "NORMAL",
            formant_status != "NORMAL",
        ])

        # Base: each flagged test = ~16.7 points
        biometric_score = suspicious_count * (100.0 / 6)

        # Continuous deviation bonuses (even if status is NORMAL)
        # Jitter deviation from baseline
        jitter_dev_pct = abs(jitter_pct - self.baselines["jitter_pct"]) / self.baselines["jitter_pct"]
        if jitter_dev_pct > 0.3:  # >30% off baseline
            biometric_score += min(10, jitter_dev_pct * 10)

        # Shimmer deviation from baseline
        shimmer_dev_pct = abs(shimmer_db - self.baselines["shimmer_db"]) / self.baselines["shimmer_db"]
        if shimmer_dev_pct > 0.3:
            biometric_score += min(10, shimmer_dev_pct * 10)

        # HNR deviation from baseline
        hnr_dev_pct = abs(hnr_db - self.baselines["hnr_db"]) / self.baselines["hnr_db"]
        if hnr_dev_pct > 0.2:
            biometric_score += min(10, hnr_dev_pct * 15)

        # F0 variance deviation
        f0_dev_pct = abs(f0_variance_hz - self.baselines["f0_variance_hz"]) / self.baselines["f0_variance_hz"]
        if f0_dev_pct > 0.3:
            biometric_score += min(10, f0_dev_pct * 8)

        biometric_score = round(min(100.0, biometric_score), 1)

        return {
            "f0_variance_hz": round(f0_variance_hz, 4),
            "f0_variance_baseline": self.baselines["f0_variance_hz"],
            "f0_status": f0_status,

            "jitter_pct": round(jitter_pct, 4),
            "jitter_baseline": self.baselines["jitter_pct"],
            "jitter_z_score": round(jitter_z, 2),
            "jitter_deviation_pct": round(jitter_deviation, 1),
            "jitter_status": jitter_status,

            "shimmer_db": round(shimmer_db, 4),
            "shimmer_baseline": self.baselines["shimmer_db"],
            "shimmer_status": shimmer_status,

            "hnr_db": round(hnr_db, 2),
            "hnr_baseline": self.baselines["hnr_db"],
            "hnr_status": hnr_status,
            "hnr_warning": hnr_warning,

            "pause_energy": pause_energy,
            "pause_status": pause_status,

            "formant_velocity": formant_velocity,
            "formant_status": formant_status,

            "overall_biometric_score": round(biometric_score, 1),
            "n_suspicious": suspicious_count,
        }

    def _compute_jitter(self, voiced_f0: np.ndarray) -> float:
        """Compute jitter (% perturbation in pitch periods)."""
        if len(voiced_f0) < 3:
            return 0.0
        periods = 1.0 / voiced_f0  # pitch periods in seconds
        diffs = np.abs(np.diff(periods))
        jitter = (diffs.mean() / periods.mean()) * 100
        return float(jitter)

    def _compute_shimmer(self, audio: np.ndarray) -> float:
        """Compute shimmer (dB perturbation in amplitude)."""
        try:
            # Get amplitude envelope using Hilbert transform
            analytic = hilbert(audio)
            envelope = np.abs(analytic)

            # Compute frame-level amplitudes
            frame_len = int(0.025 * self.sr)  # 25ms
            hop = int(0.010 * self.sr)        # 10ms

            amplitudes = []
            for i in range(0, len(envelope) - frame_len, hop):
                frame = envelope[i:i+frame_len]
                rms = np.sqrt(np.mean(frame**2))
                if rms > 1e-6:
                    amplitudes.append(rms)

            if len(amplitudes) < 3:
                return 0.0

            amplitudes = np.array(amplitudes)
            # Shimmer in dB
            db_amps = 20 * np.log10(amplitudes + 1e-10)
            shimmer = np.mean(np.abs(np.diff(db_amps)))
            return float(shimmer)

        except Exception:
            return 0.0

    def _compute_hnr(self, audio: np.ndarray) -> float:
        """Compute Harmonic-to-Noise Ratio in dB."""
        try:
            # Autocorrelation method
            n = len(audio)
            if n < 1000:
                return 0.0

            # Use center 1 second for stability
            center = n // 2
            half_sec = self.sr // 2
            segment = audio[max(0, center-half_sec):min(n, center+half_sec)]

            # Autocorrelation
            autocorr = np.correlate(segment, segment, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)

            # Find first peak after initial decay (pitch period)
            min_lag = int(self.sr / 500)  # 500 Hz max
            max_lag = int(self.sr / 50)   # 50 Hz min

            if max_lag >= len(autocorr):
                max_lag = len(autocorr) - 1

            search_region = autocorr[min_lag:max_lag]
            if len(search_region) == 0:
                return 15.0

            peak_val = search_region.max()
            peak_val = max(peak_val, 1e-6)  # avoid log(0)

            # HNR = 10 * log10(r / (1 - r))
            if peak_val >= 1.0:
                hnr = 40.0  # very harmonic
            else:
                hnr = 10 * np.log10(peak_val / (1 - peak_val + 1e-10))

            return float(np.clip(hnr, 0, 45))

        except Exception:
            return 15.0

    def _compute_pause_energy(self, audio: np.ndarray) -> Dict:
        """Analyze energy in pause regions — digital silence indicates synthesis."""
        try:
            frame_len = int(0.020 * self.sr)  # 20ms frames
            hop = int(0.010 * self.sr)  # 10ms hop

            rms_values = []
            for i in range(0, len(audio) - frame_len, hop):
                frame = audio[i:i+frame_len]
                rms = float(np.sqrt(np.mean(frame**2)))
                rms_values.append(rms)

            if len(rms_values) < 5:
                return {"min_rms": 0.0, "baseline_min": 0.0008, "suspicious": False,
                        "n_silent_frames": 0, "pct_silent": 0.0}

            rms_arr = np.array(rms_values)
            # "Pause frames" = bottom 30% energy percentile
            threshold = np.percentile(rms_arr, 30)
            pause_frames = rms_arr[rms_arr <= threshold]
            min_rms = float(np.min(rms_arr))

            # Baseline: human minimum pause energy ~0.0008
            baseline_min = 0.0008
            # Digital silence: 3x quieter than the human minimum
            suspicious = min_rms < baseline_min * 0.3

            # Count near-zero frames (true digital silence)
            n_silent = int(np.sum(rms_arr < 1e-6))
            pct_silent = round(n_silent / len(rms_arr) * 100, 1)

            return {
                "min_rms": round(min_rms, 8),
                "mean_pause_rms": round(float(np.mean(pause_frames)), 8),
                "baseline_min": baseline_min,
                "suspicious": suspicious,
                "n_silent_frames": n_silent,
                "pct_silent": pct_silent,
            }
        except Exception:
            return {"min_rms": 0.0, "baseline_min": 0.0008, "suspicious": False,
                    "n_silent_frames": 0, "pct_silent": 0.0}

    def _compute_formant_velocity(self, audio: np.ndarray) -> Dict:
        """Analyze formant F2 transition velocity — inhuman speed indicates TTS."""
        try:
            from scipy.signal import lfilter

            frame_len = int(0.025 * self.sr)  # 25ms
            hop = int(0.010 * self.sr)  # 10ms
            lpc_order = 8

            f2_values = []
            for i in range(0, len(audio) - frame_len, hop):
                frame = audio[i:i+frame_len]
                # Apply pre-emphasis
                frame = lfilter([1, -0.97], [1], frame)
                # Hamming window
                frame = frame * np.hamming(len(frame))

                try:
                    # LPC analysis
                    a = librosa.lpc(frame, order=lpc_order)
                    # Find formants from LPC roots
                    roots = np.roots(a)
                    roots = roots[np.imag(roots) > 0]  # positive frequencies only
                    if len(roots) == 0:
                        f2_values.append(np.nan)
                        continue

                    # Convert roots to frequencies
                    angles = np.angle(roots)
                    freqs = sorted(angles * (self.sr / (2 * np.pi)))

                    # F2 is typically the second formant (1000-3000 Hz range)
                    f2_candidates = [f for f in freqs if 800 < f < 4000]
                    if len(f2_candidates) >= 2:
                        f2_values.append(f2_candidates[1])  # second formant
                    elif len(f2_candidates) == 1:
                        f2_values.append(f2_candidates[0])
                    else:
                        f2_values.append(np.nan)
                except Exception:
                    f2_values.append(np.nan)

            f2_arr = np.array(f2_values, dtype=float)
            # Remove NaN for velocity computation
            valid_mask = ~np.isnan(f2_arr)
            if np.sum(valid_mask) < 5:
                return {"max_hz_per_frame": 0.0, "mean_velocity": 0.0,
                        "n_violations": 0, "suspicious": False, "limit": 50.0}

            # Compute velocity (Hz per 10ms frame)
            valid_f2 = f2_arr[valid_mask]
            velocity = np.abs(np.diff(valid_f2))

            max_vel = float(np.max(velocity)) if len(velocity) > 0 else 0.0
            mean_vel = float(np.mean(velocity)) if len(velocity) > 0 else 0.0

            # Human F2 velocity limit: ~50 Hz per 10ms frame
            limit = 50.0
            n_violations = int(np.sum(velocity > limit * 2))  # 2x the limit
            suspicious = max_vel > limit * 2 or n_violations > 3

            return {
                "max_hz_per_frame": round(max_vel, 1),
                "mean_velocity": round(mean_vel, 1),
                "n_violations": n_violations,
                "suspicious": suspicious,
                "limit": limit,
            }
        except Exception:
            return {"max_hz_per_frame": 0.0, "mean_velocity": 0.0,
                    "n_violations": 0, "suspicious": False, "limit": 50.0}

    def _empty_result(self) -> Dict:
        """Return empty biometric result when analysis fails."""
        return {
            "f0_variance_hz": 0.0,
            "f0_variance_baseline": self.baselines["f0_variance_hz"],
            "f0_status": "UNKNOWN",
            "jitter_pct": 0.0,
            "jitter_baseline": self.baselines["jitter_pct"],
            "jitter_z_score": 0.0,
            "jitter_deviation_pct": 0.0,
            "jitter_status": "UNKNOWN",
            "shimmer_db": 0.0,
            "shimmer_baseline": self.baselines["shimmer_db"],
            "shimmer_status": "UNKNOWN",
            "hnr_db": 0.0,
            "hnr_baseline": self.baselines["hnr_db"],
            "hnr_status": "UNKNOWN",
            "pause_energy": {"min_rms": 0.0, "baseline_min": 0.0008,
                            "suspicious": False, "n_silent_frames": 0, "pct_silent": 0.0},
            "pause_status": "UNKNOWN",
            "formant_velocity": {"max_hz_per_frame": 0.0, "mean_velocity": 0.0,
                                "n_violations": 0, "suspicious": False, "limit": 50.0},
            "formant_status": "UNKNOWN",
            "overall_biometric_score": 0.0,
            "n_suspicious": 0,
        }
