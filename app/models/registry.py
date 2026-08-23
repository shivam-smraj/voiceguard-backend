"""
Model Registry — Central catalog of all detection models.

Hook-based pattern:
  - Set active=True and provide a weight for each model in the ensemble
  - Weights auto-normalize to sum=1.0
  - AASIST can be dropped in later by placing the checkpoint and setting active=True
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np

from app.models.lcnn import LCNN
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """A single model in the registry."""
    name: str
    model_type: str          # "lcnn" or "aasist_onnx"
    feature_mode: str        # "mfcc", "lfcc", "raw_waveform"
    checkpoint_path: str
    active: bool = True
    weight: float = 1.0
    eer: float = 0.0
    n_coeff: int = 0         # for LCNN: 120 (MFCC) or 180 (LFCC)

    # Runtime — populated after load
    model: Optional[nn.Module] = field(default=None, repr=False)
    onnx_session: Optional[ort.InferenceSession] = field(default=None, repr=False)
    loaded: bool = False


class ModelRegistry:
    """
    Manages all detection models.
    Loads checkpoints, runs inference, fuses scores.
    """

    def __init__(self):
        self.entries: Dict[str, ModelEntry] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register the default model catalog."""

        # LCNN-MFCC (best model, EER=1.2%)
        self.entries["lcnn_mfcc"] = ModelEntry(
            name="LCNN-MFCC",
            model_type="lcnn",
            feature_mode="mfcc",
            checkpoint_path=str(settings.MODELS_DIR / settings.CHECKPOINT_LCNN_MFCC),
            active=True,
            weight=0.40,
            eer=settings.EER_LCNN_MFCC,
            n_coeff=settings.MFCC_TOTAL,    # 120
        )

        # LCNN-LFCC (second best, EER=1.74%)
        self.entries["lcnn_lfcc"] = ModelEntry(
            name="LCNN-LFCC",
            model_type="lcnn",
            feature_mode="lfcc",
            checkpoint_path=str(settings.MODELS_DIR / settings.CHECKPOINT_LCNN_LFCC),
            active=True,
            weight=0.30,
            eer=settings.EER_LCNN_LFCC,
            n_coeff=settings.LFCC_TOTAL,    # 180
        )

        # AASIST-L (ONNX)
        self.entries["aasist"] = ModelEntry(
            name="AASIST-L",
            model_type="aasist_onnx",
            feature_mode="raw_waveform",
            checkpoint_path=str(settings.MODELS_DIR / settings.CHECKPOINT_AASIST),
            active=True,
            weight=0.30,
            eer=4.03,
        )

    def load_all(self):
        """Load all active model checkpoints into memory."""
        for key, entry in self.entries.items():
            if not entry.active:
                logger.info(f"Skipping inactive model: {entry.name}")
                continue

            try:
                self._load_model(entry)
                logger.info(
                    f"[OK] Loaded {entry.name} | "
                    f"EER={entry.eer}% | weight={entry.weight}"
                )
            except Exception as e:
                logger.error(f"[FAIL] Failed to load {entry.name}: {e}")
                entry.active = False

        self._normalize_weights()
        active = [e.name for e in self.entries.values() if e.active and e.loaded]
        logger.info(f"Active ensemble: {active}")

    def _load_model(self, entry: ModelEntry):
        """Load a single model from its checkpoint."""
        path = Path(entry.checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if entry.model_type == "lcnn":
            model = LCNN(n_coeff=entry.n_coeff, n_frames=settings.MAX_FRAMES)

            # Handle numpy._core compatibility (checkpoints saved with numpy 2.x)
            import sys
            if not hasattr(np, '_core'):
                # numpy 1.x doesn't have _core — patch it
                sys.modules['numpy._core'] = np.core
                sys.modules['numpy._core.multiarray'] = np.core.multiarray

            # Try safe globals approach first (PyTorch 2.6+)
            try:
                import numpy._core.multiarray as _npcm
                torch.serialization.add_safe_globals([_npcm.scalar])
                checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
            except Exception:
                checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

            # Handle both formats: dict with 'model_state' or raw state_dict
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
                logger.info(
                    f"  Loaded from epoch {checkpoint.get('epoch', '?')}, "
                    f"best_eer={checkpoint.get('best_eer', '?')}"
                )
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            entry.model = model
            entry.loaded = True

        elif entry.model_type == "aasist_onnx":
            session = ort.InferenceSession(
                str(path),
                providers=["CPUExecutionProvider"]
            )
            entry.onnx_session = session
            entry.loaded = True

    def _normalize_weights(self):
        """Ensure active model weights sum to 1.0."""
        active = [e for e in self.entries.values() if e.active and e.loaded]
        if not active:
            return
        total = sum(e.weight for e in active)
        if total > 0:
            for e in active:
                e.weight = e.weight / total

    def get_active_models(self):
        """Return list of active, loaded model entries."""
        return [e for e in self.entries.values() if e.active and e.loaded]

    def predict_single(self, entry: ModelEntry, tensor_input) -> dict:
        """
        Run inference on a single model.

        Args:
            entry: ModelEntry to run
            tensor_input: For LCNN: (1, 1, n_coeff, frames) tensor
                         For AASIST ONNX: (1, clip_len) numpy array

        Returns:
            dict with spoof_probability, logits, embedding
        """
        if entry.model_type == "lcnn":
            return self._predict_lcnn(entry, tensor_input)
        elif entry.model_type == "aasist_onnx":
            return self._predict_aasist_onnx(entry, tensor_input)
        else:
            raise ValueError(f"Unknown model type: {entry.model_type}")

    def _predict_lcnn(self, entry: ModelEntry, tensor_input: torch.Tensor) -> dict:
        """Run LCNN inference. Returns spoof probability and embedding."""
        with torch.no_grad():
            logits = entry.model(tensor_input)            # (1, 2)
            probs = torch.softmax(logits, dim=1)          # (1, 2)
            embedding = entry.model.get_embedding(tensor_input)  # (1, 256)

        spoof_prob = float(probs[0, 1])   # P(fake)
        return {
            "spoof_probability": spoof_prob,
            "logits": logits[0].numpy().tolist(),
            "embedding": embedding[0].numpy().tolist(),
        }

    def _predict_aasist_onnx(self, entry: ModelEntry, waveform_np: np.ndarray) -> dict:
        """Run AASIST ONNX inference."""
        audio_input = waveform_np.astype(np.float32).reshape(1, -1)
        
        # Ensure it is exactly 32000 samples for the ONNX model
        if audio_input.shape[1] < 32000:
            audio_input = np.pad(audio_input, ((0, 0), (0, 32000 - audio_input.shape[1])), mode='constant')
        elif audio_input.shape[1] > 32000:
            audio_input = audio_input[:, :32000]

        outputs = entry.onnx_session.run(None, {"waveform": audio_input})
        logits = outputs[0]  # shape: (1, 2)

        # Log-likelihood ratio — positive = spoof
        exp = np.exp(logits[0] - np.max(logits[0]))
        probs = exp / exp.sum()
        spoof_prob = float(probs[1])

        return {
            "spoof_probability": spoof_prob,
            "logits": logits[0].tolist(),
            "embedding": [],  # AASIST simplified ONNX model does not output embedding
        }

    def ensemble_predict(self, features: dict) -> dict:
        """
        Run all active models and produce weighted ensemble score.

        Args:
            features: dict with keys matching model feature_modes
                      e.g. {"mfcc": tensor, "lfcc": tensor, "raw_waveform": np_array}

        Returns:
            dict with ensemble_score, verdict, per_model_results, confidence_label
        """
        active_models = self.get_active_models()
        if not active_models:
            raise RuntimeError("No active models loaded")

        per_model = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for entry in active_models:
            input_data = features.get(entry.feature_mode)
            if input_data is None:
                logger.warning(f"No features for {entry.name} ({entry.feature_mode})")
                continue

            result = self.predict_single(entry, input_data)
            per_model[entry.name] = {
                "spoof_probability": result["spoof_probability"],
                "weight": entry.weight,
                "eer": entry.eer,
            }
            weighted_sum += result["spoof_probability"] * entry.weight
            total_weight += entry.weight

        if total_weight == 0:
            raise RuntimeError("No model produced results")

        ensemble_score = (weighted_sum / total_weight) * 100  # 0-100 scale
        verdict = "FAKE" if ensemble_score >= 50.0 else "REAL"

        # Confidence label
        if ensemble_score >= 85 or ensemble_score <= 15:
            confidence_label = "CRITICAL"
        elif ensemble_score >= 70 or ensemble_score <= 30:
            confidence_label = "HIGH"
        elif ensemble_score >= 55 or ensemble_score <= 45:
            confidence_label = "MEDIUM"
        else:
            confidence_label = "LOW"

        return {
            "ensemble_score": round(ensemble_score, 2),
            "verdict": verdict,
            "confidence_label": confidence_label,
            "per_model": per_model,
        }


# Singleton
registry = ModelRegistry()
