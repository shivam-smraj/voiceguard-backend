"""
Heatmap Renderer — Creates the 3-panel visualization:
  Panel 1: Original spectrogram/feature map
  Panel 2: Grad-CAM heatmap (jet colormap)
  Panel 3: Overlay (original + heatmap)

Returns base64-encoded PNG string for embedding in JSON responses.
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import logging

logger = logging.getLogger(__name__)


def render_heatmap_trio(
    feature_map: np.ndarray,
    heatmap: np.ndarray,
    title: str = "Grad-CAM Analysis",
    findings: list = None,
    evidence_type: str = None,
    figsize: tuple = (18, 5),
    dpi: int = 120,
) -> str:
    """
    Render 3-panel heatmap visualization.

    Args:
        feature_map: (n_coeff, max_frames) — raw LFCC/MFCC features
        heatmap: (n_coeff, max_frames) — Grad-CAM output [0,1]
        title: figure title
        findings: list of artifact findings to annotate
        evidence_type: 'gradcam_mfcc' or 'gradcam_lfcc'
        figsize: matplotlib figure size
        dpi: output resolution

    Returns:
        Base64-encoded PNG string
    """
    try:
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
        fig.suptitle(title, fontsize=14, fontweight='bold', color='white')
        fig.patch.set_facecolor('#1a1a2e')

        # Panel 1: Original feature map
        ax1 = axes[0]
        ax1.set_title('Original Features', color='white', fontsize=11)
        im1 = ax1.imshow(
            feature_map, aspect='auto', origin='lower',
            cmap='viridis', interpolation='bilinear',
        )
        ax1.set_xlabel('Time Frame', color='white')
        ax1.set_ylabel('Coefficient', color='white')
        ax1.tick_params(colors='white')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Panel 2: Grad-CAM heatmap
        ax2 = axes[1]
        ax2.set_title('Grad-CAM Heatmap', color='white', fontsize=11)
        # Custom red-orange-yellow colormap
        colors_list = ['#000033', '#0000ff', '#00ffff', '#ffff00', '#ff8800', '#ff0000']
        custom_cmap = LinearSegmentedColormap.from_list('forensic', colors_list, N=256)
        im2 = ax2.imshow(
            heatmap, aspect='auto', origin='lower',
            cmap=custom_cmap, vmin=0, vmax=1,
            interpolation='bilinear',
        )
        ax2.set_xlabel('Time Frame', color='white')
        ax2.set_ylabel('Coefficient', color='white')
        ax2.tick_params(colors='white')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Panel 3: Overlay
        ax3 = axes[2]
        ax3.set_title('Overlay (Evidence)', color='white', fontsize=11)
        ax3.imshow(
            feature_map, aspect='auto', origin='lower',
            cmap='gray', interpolation='bilinear',
        )
        im3 = ax3.imshow(
            heatmap, aspect='auto', origin='lower',
            cmap=custom_cmap, alpha=0.6, vmin=0, vmax=1,
            interpolation='bilinear',
        )
        ax3.set_xlabel('Time Frame', color='white')
        ax3.set_ylabel('Coefficient', color='white')
        ax3.tick_params(colors='white')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Annotate findings on the overlay panel if provided
        if findings and evidence_type:
            for f in findings:
                if f.get("evidence_type") == evidence_type:
                    bbox = f.get("bbox")
                    if bbox:
                        row_min, col_min, height, width = bbox
                        x_center = col_min + width / 2.0
                        y_center = row_min + height / 2.0
                        rank = f.get("rank", "?")
                        # Plot filled white circle with thin black outline
                        ax3.plot(
                            x_center, y_center, marker='o',
                            markersize=14, color='white',
                            markeredgecolor='black', markeredgewidth=1.5
                        )
                        # Overlay rank number inside
                        ax3.text(
                            x_center, y_center, str(rank),
                            color='black', fontsize=8, fontweight='bold',
                            ha='center', va='center'
                        )

        # Style all axes
        for ax in axes:
            ax.set_facecolor('#16213e')
            for spine in ax.spines.values():
                spine.set_color('#444')

        plt.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')

        return b64

    except Exception as e:
        logger.error(f"Heatmap rendering failed: {e}")
        plt.close('all')
        return ""

