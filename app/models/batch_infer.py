"""
Batch inference support for maximum CPU throughput.

Instead of running one chunk at a time through each model,
we stack N chunks into a single tensor and run ONE forward pass.

Batch size = 8 gives ~6-8x speedup on CPU (PyTorch internal threading
over the batch dimension is very efficient).
"""

import logging
from typing import List, Dict
import numpy as np
import torch
import onnxruntime as ort

logger = logging.getLogger(__name__)


def batch_predict_lcnn(model, mfcc_batch: torch.Tensor) -> List[float]:
    """
    Run LCNN on a batch of MFCC/LFCC feature tensors in one forward pass.

    Args:
        model: LCNN model (already in eval mode)
        mfcc_batch: (B, 1, n_coeff, MAX_FRAMES) — B chunks stacked

    Returns:
        List of B spoof probabilities (P(fake) per chunk)
    """
    with torch.no_grad():
        logits = model(mfcc_batch)              # (B, 2)
        probs  = torch.softmax(logits, dim=1)   # (B, 2)
    return probs[:, 1].tolist()                 # P(fake) per chunk


def batch_predict_aasist(session: ort.InferenceSession,
                          waveforms: List[np.ndarray]) -> List[float]:
    """
    Run AASIST ONNX on multiple waveforms.
    ONNX Runtime doesn't support dynamic batching easily for AASIST,
    so we parallelize with threads instead.

    Args:
        session: ONNX Runtime session
        waveforms: List of (32000,) numpy arrays

    Returns:
        List of spoof probabilities
    """
    from concurrent.futures import ThreadPoolExecutor

    def _infer_one(wav: np.ndarray) -> float:
        inp = wav.astype(np.float32).reshape(1, -1)
        if inp.shape[1] < 32000:
            inp = np.pad(inp, ((0, 0), (0, 32000 - inp.shape[1])))
        elif inp.shape[1] > 32000:
            inp = inp[:, :32000]
        out = session.run(None, {"waveform": inp})[0][0]  # (2,)
        exp = np.exp(out - out.max())
        probs = exp / exp.sum()
        return float(probs[1])

    with ThreadPoolExecutor(max_workers=min(len(waveforms), 4)) as ex:
        results = list(ex.map(_infer_one, waveforms))
    return results


def batch_ensemble_predict(registry, features_list: List[dict]) -> List[dict]:
    """
    Run full ensemble on a batch of feature dicts.

    This is the core speed improvement: instead of N sequential calls
    to ensemble_predict(), we stack all features and do ONE pass per model.

    Args:
        registry: ModelRegistry instance
        features_list: List of feature dicts (output of extract_all_features)

    Returns:
        List of result dicts matching the format of ensemble_predict()
    """
    B = len(features_list)
    if B == 0:
        return []

    active = registry.get_active_models()
    per_chunk_weighted = [0.0] * B
    total_weight = 0.0
    per_model_results: List[Dict] = [{} for _ in range(B)]

    for entry in active:
        mode = entry.feature_mode

        # ── LCNN batch ──────────────────────────────────────────
        if entry.model_type == "lcnn" and entry.model is not None:
            # Stack: (B, 1, n_coeff, MAX_FRAMES)
            tensors = []
            valid_indices = []
            for i, feat in enumerate(features_list):
                t = feat.get(mode)
                if t is not None:
                    tensors.append(t)   # already (1, 1, n, T)
                    valid_indices.append(i)

            if tensors:
                batch = torch.cat(tensors, dim=0)   # (B, 1, n, T)
                probs = batch_predict_lcnn(entry.model, batch)
                for j, idx in enumerate(valid_indices):
                    p = probs[j]
                    per_chunk_weighted[idx] += p * entry.weight
                    per_model_results[idx][entry.name] = {
                        "spoof_probability": p,
                        "weight": entry.weight,
                        "eer": entry.eer,
                    }

        # ── AASIST ONNX (parallelized per-sample) ───────────────
        elif entry.model_type == "aasist_onnx" and entry.onnx_session is not None:
            wavs = []
            valid_indices = []
            for i, feat in enumerate(features_list):
                w = feat.get("raw_waveform")
                if w is not None:
                    wavs.append(w)
                    valid_indices.append(i)

            if wavs:
                probs = batch_predict_aasist(entry.onnx_session, wavs)
                for j, idx in enumerate(valid_indices):
                    p = probs[j]
                    per_chunk_weighted[idx] += p * entry.weight
                    per_model_results[idx][entry.name] = {
                        "spoof_probability": p,
                        "weight": entry.weight,
                        "eer": entry.eer,
                    }

        total_weight += entry.weight

    # Build final result list
    results = []
    for i in range(B):
        score = (per_chunk_weighted[i] / max(total_weight, 1e-9)) * 100
        verdict = "FAKE" if score >= 50.0 else "REAL"
        if score >= 85 or score <= 15:
            conf = "CRITICAL"
        elif score >= 70 or score <= 30:
            conf = "HIGH"
        elif score >= 55 or score <= 45:
            conf = "MEDIUM"
        else:
            conf = "LOW"
        results.append({
            "ensemble_score": round(score, 2),
            "verdict": verdict,
            "confidence_label": conf,
            "per_model": per_model_results[i],
        })

    return results
