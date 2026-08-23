"""
GradientSHAP Explainer — Computes SHAP feature attributions.
Uses a G.711 µ-law codec simulation as baseline features.
"""

import io
import base64
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend
import matplotlib.pyplot as plt
import logging
from captum.attr import GradientShap
import librosa
from app.config import settings

logger = logging.getLogger(__name__)


def simulate_codec_degradation(audio_np: np.ndarray, sr=16000) -> np.ndarray:
    """
    Simulate G.711 µ-law codec degradation.
    Applies 8kHz resampling (low-pass/bandpass filtering) and 8-bit µ-law quantization.
    """
    try:
        # 1. Resample to 8kHz to simulate G.711 sampling rate limit (band-limiting/low-pass)
        audio_8k = librosa.resample(audio_np, orig_sr=sr, target_sr=8000)
        
        # 2. µ-law companding (compress)
        mu = 255
        x = np.clip(audio_8k, -1.0, 1.0)
        comp = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
        
        # 3. 8-bit quantization
        quant = np.round((comp + 1.0) * 127.5 - 127.5) / 127.5
        
        # 4. µ-law expansion (decompress)
        recon = np.sign(quant) * ((1 + mu)**np.abs(quant) - 1) / mu
        
        # 5. Resample back to 16kHz
        audio_16k = librosa.resample(recon, orig_sr=8000, target_sr=sr)
        
        # Ensure exact length match
        if len(audio_16k) < len(audio_np):
            audio_16k = np.pad(audio_16k, (0, len(audio_np) - len(audio_16k)))
        else:
            audio_16k = audio_16k[:len(audio_np)]
        return audio_16k.astype(np.float32)
    except Exception as e:
        logger.error(f"Codec degradation simulation failed: {e}")
        return audio_np.astype(np.float32)


def explain_features_with_shap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    baseline_tensor: torch.Tensor,
    title: str = "GradientSHAP Feature Attribution",
    figsize: tuple = (10, 4),
    dpi: int = 120,
) -> str:
    """
    Compute GradientSHAP attributions for the LCNN model.
    
    Args:
        model: loaded PyTorch module (LCNN)
        input_tensor: (1, 1, n_coeff, max_frames)
        baseline_tensor: (1, 1, n_coeff, max_frames)
        title: title of the plotted output
        figsize: plot figure dimensions
        dpi: resolution of the plot
        
    Returns:
        Base64-encoded PNG string of the attribution heatmap
    """
    try:
        model.eval()
        # Initialize GradientShap explainer
        gs = GradientShap(model)
        
        # Prepare inputs and baselines
        inputs = input_tensor.clone().detach().requires_grad_(True)
        baselines = baseline_tensor.clone().detach()
        
        # Compute attribution for target=1 (FAKE class)
        attributions = gs.attribute(
            inputs,
            baselines=baselines,
            target=1,
            n_samples=25,       # Kept moderate for CPU performance
            stdevs=0.01,
        )
        
        # Squeeze down to (n_coeff, max_frames) for plotting
        attr_np = attributions.detach().cpu().squeeze().numpy()
        
        # Center the diverging colormap at zero
        max_abs = np.max(np.abs(attr_np))
        if max_abs == 0:
            max_abs = 1.0
            
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        
        im = ax.imshow(
            attr_np, aspect='auto', origin='lower',
            cmap='coolwarm', vmin=-max_abs, vmax=max_abs,
            interpolation='bilinear'
        )
        
        ax.set_title(title, color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Frame', color='white', fontsize=9)
        ax.set_ylabel('Coefficient Index', color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#444')
            
        # Add colorbar with white text
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white', fontsize=8)
        cbar.set_label('Attribution (Red = Fake, Blue = Real)', color='white', fontsize=8)
        
        plt.tight_layout()
        
        # Convert figure to base64 PNG
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return b64
        
    except Exception as e:
        logger.error(f"GradientSHAP explanation failed: {e}", exc_info=True)
        plt.close('all')
        return ""
