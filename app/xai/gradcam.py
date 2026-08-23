"""
Grad-CAM for LCNN — Visual explanation of which frequency-time regions
triggered the deepfake detection.

Hooks into model.features[23] (BatchNorm2d, last layer of Block 4).
Produces a heatmap showing where the model "looked" when deciding FAKE.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)


class LCNNGradCAM:
    """
    Grad-CAM implementation for LCNN deepfake detection.

    Usage:
        gcam = LCNNGradCAM(model, target_layer_idx=23)
        heatmap = gcam.generate(input_tensor, target_class=1)
        gcam.cleanup()  # MUST call to avoid memory leaks
    """

    def __init__(self, model, target_layer_idx: int = None):
        """
        Args:
            model: LCNN model instance (eval mode)
            target_layer_idx: Index into model.features Sequential
                             Default: settings.GRADCAM_TARGET_LAYER = 23
        """
        self.model = model
        self.target_layer_idx = target_layer_idx or settings.GRADCAM_TARGET_LAYER
        self.target_layer = model.features[self.target_layer_idx]

        # Storage for hooks
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook: save the activation map."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: save the gradient."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
        upsample: bool = True,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, 1, n_coeff, max_frames) — single sample batch
            target_class: 1 = FAKE (we want to see why it thinks it's fake)
            upsample: If True, resize heatmap to match input dimensions

        Returns:
            numpy array of shape (n_coeff, max_frames), values in [0, 1]
        """
        self.model.eval()

        # Enable gradients for this pass
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)  # (1, 2)

        # Zero all gradients
        self.model.zero_grad()

        # Backward pass for target class
        target_score = logits[0, target_class]
        target_score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            logger.error("Grad-CAM hooks did not capture activations/gradients")
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((h, w), dtype=np.float32)

        # Global Average Pooling of gradients → channel weights
        # gradients: (1, C, H', W') → weights: (1, C, 1, 1)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted sum of activation maps
        # activations: (1, C, H', W') × weights: (1, C, 1, 1) → sum over C
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')

        # ReLU — only positive contributions
        cam = F.relu(cam)

        # Upsample to input resolution
        if upsample:
            target_h = input_tensor.shape[2]
            target_w = input_tensor.shape[3]
            cam = F.interpolate(
                cam,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False,
            )

        # Remove batch and channel dims explicitly: (1,1,H,W) -> (H,W)
        # Using [0, 0] index is more reliable than squeeze() which can leave
        # unexpected dims when any spatial dim happens to be size 1.
        cam_np = cam.detach().cpu()
        # Handle any shape: strip leading dims until 2D
        while cam_np.dim() > 2:
            cam_np = cam_np[0]
        cam = cam_np.numpy().astype(np.float32)

        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        return cam

    def cleanup(self):
        """Remove hooks to prevent memory leaks. MUST call when done."""
        if self._fwd_hook:
            self._fwd_hook.remove()
        if self._bwd_hook:
            self._bwd_hook.remove()
        self.activations = None
        self.gradients = None


def generate_gradcam_for_model(
    model,
    input_tensor: torch.Tensor,
    target_layer_idx: int = None,
) -> np.ndarray:
    """
    Convenience function: generate Grad-CAM and clean up automatically.

    Args:
        model: LCNN model
        input_tensor: (1, 1, n_coeff, max_frames)
        target_layer_idx: layer index (default from settings)

    Returns:
        heatmap numpy array (n_coeff, max_frames), values [0, 1]
    """
    gcam = LCNNGradCAM(model, target_layer_idx)
    try:
        heatmap = gcam.generate(input_tensor, target_class=1)
    finally:
        gcam.cleanup()
    return heatmap
