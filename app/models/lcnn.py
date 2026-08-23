"""
LCNN Architecture — Copied VERBATIM from the training notebook.

Architecture: ASVspoof 2019/2021 LCNN baseline + MFM activation.
  Conv(1→32) → MFM → MaxPool
  Conv(16→64) → MFM → BN → Conv(32→64) → MFM → MaxPool → BN
  Conv(32→96) → MFM → BN → Conv(48→96) → MFM → MaxPool → BN
  Conv(48→128) → MFM → BN → Conv(64→128) → MFM → MaxPool → BN
  AdaptiveAvgPool(4,4) → Flatten → MFMLinear(1024,256) → Dropout(0.75) → Linear(256,2)

Grad-CAM target: features[23] (BatchNorm2d after last conv block)
Embedding layer: classifier[1] (MFMLinear, outputs 256-dim)

670,562 parameters per model variant.
"""

import torch
import torch.nn as nn


# ── Max-Feature-Map (MFM) Activation ─────────────────────────────────

class MFM2D(nn.Module):
    """
    Max-Feature-Map activation for 2D convolutional layers.
    Splits 2N input channels into two groups of N, takes element-wise max.
    Output channels = input channels // 2.
    """
    def forward(self, x):
        # x: (B, 2N, H, W) → two halves → max → (B, N, H, W)
        x1, x2 = torch.chunk(x, 2, dim=1)
        return torch.max(x1, x2)


class MFMLinear(nn.Module):
    """
    MFM activation for fully-connected layers.
    Equivalent to FC(in, 2*out) → MFM → (B, out)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 2)

    def forward(self, x):
        out = self.fc(x)
        x1, x2 = torch.chunk(out, 2, dim=1)
        return torch.max(x1, x2)


# ── LCNN Architecture ──────────────────────────────────────────────────

class LCNN(nn.Module):
    """
    Light CNN for audio deepfake detection.

    Based on the ASVspoof 2019/2021 official LCNN baseline.
    Uses MFM activation throughout — the key differentiator from standard CNNs.

    Input : (B, 1, n_coeff_with_deltas, max_frames)
             LFCC: (B, 1, 180, 200)
             MFCC: (B, 1, 120, 200)
    Output: (B, 2)  — [logit_real, logit_fake]
    """

    def __init__(self, n_coeff, n_frames=200):
        """
        Args:
            n_coeff  : number of input coefficient rows (180 for LFCC, 120 for MFCC)
            n_frames : number of time frames (200)
        """
        super().__init__()

        self.features = nn.Sequential(
            # ── Block 1 ───────────────────────────────────── Index 0-2
            # Conv(1→32) → MFM → (B,16,C,T) → MaxPool → (B,16,C/2,T/2)
            nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=False),   # [0]
            MFM2D(),                              # 32→16 channels    # [1]
            nn.MaxPool2d(kernel_size=2, stride=2),                    # [2]

            # ── Block 2 ───────────────────────────────────── Index 3-9
            # 1×1 conv: channel mixing without spatial change
            nn.Conv2d(16, 64, kernel_size=1, bias=False),             # [3]
            MFM2D(),                              # 64→32             # [4]
            nn.BatchNorm2d(32),                                       # [5]
            # 3×3 conv: spatial feature extraction
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),  # [6]
            MFM2D(),                              # 64→32             # [7]
            nn.MaxPool2d(kernel_size=2, stride=2),                    # [8]
            nn.BatchNorm2d(32),                                       # [9]

            # ── Block 3 ───────────────────────────────────── Index 10-16
            nn.Conv2d(32, 96, kernel_size=1, bias=False),             # [10]
            MFM2D(),                              # 96→48             # [11]
            nn.BatchNorm2d(48),                                       # [12]
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),  # [13]
            MFM2D(),                              # 96→48             # [14]
            nn.MaxPool2d(kernel_size=2, stride=2),                    # [15]
            nn.BatchNorm2d(48),                                       # [16]

            # ── Block 4 ───────────────────────────────────── Index 17-23
            nn.Conv2d(48, 128, kernel_size=1, bias=False),            # [17]
            MFM2D(),                              # 128→64            # [18]
            nn.BatchNorm2d(64),                                       # [19]
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), # [20]
            MFM2D(),                              # 128→64            # [21]
            nn.MaxPool2d(kernel_size=2, stride=2),                    # [22]
            nn.BatchNorm2d(64),                                       # [23] ← GRAD-CAM TARGET

            # ── Global pooling ───────────────────────────── Index 24
            nn.AdaptiveAvgPool2d((4, 4)),          # (B, 64, 4, 4)    # [24]
        )

        # ── Classifier with MFM-FC ───────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                          # (B, 64*4*4 = 1024)  # [0]
            MFMLinear(1024, 256),                  # (B, 256)            # [1] ← EMBEDDING
            nn.Dropout(0.75),                      # high dropout        # [2]
            nn.Linear(256, 2),                     # (B, 2) logits       # [3]
        )

    def forward(self, x):
        """
        Args: x — (B, 1, n_coeff, max_frames)
        Returns: logits — (B, 2)  [logit_real, logit_fake]
        """
        feat   = self.features(x)          # (B, 64, 4, 4)
        logits = self.classifier(feat)     # (B, 2)
        return logits

    def get_embedding(self, x):
        """Returns 256-d embedding for UMAP/ensemble fusion (before final FC)."""
        feat = self.features(x)
        feat = nn.Flatten()(feat)
        emb  = self.classifier[1](feat)   # MFMLinear output
        return emb
