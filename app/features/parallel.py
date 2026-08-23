"""
Parallel Feature Extraction — extract MFCC/LFCC for all chunks simultaneously.

On a 4-core CPU:
  Sequential:   133 chunks × 0.3s = ~40s just for feature extraction
  Parallel (4): 133 chunks / 4 cores × 0.3s = ~10s

Uses ThreadPoolExecutor (not ProcessPoolExecutor) because:
  - PyTorch/torchaudio release the GIL during transform ops
  - No pickle overhead for large arrays
  - Shared memory — no data copying between processes
"""

import logging
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# Use all available CPU cores for feature extraction
_N_FEATURE_WORKERS = max(2, (os.cpu_count() or 4))

logger.info(f"Feature extractor: {_N_FEATURE_WORKERS} parallel workers")


def parallel_extract_all(
    chunks: List[dict],
    max_workers: int = None,
) -> List[Tuple[int, dict]]:
    """
    Extract features for ALL chunks simultaneously using a thread pool.

    Args:
        chunks: List of chunk dicts from load_audio_chunks()
                Each has: {chunk_idx, audio_np, start_s, end_s}
        max_workers: Override thread count (default = CPU count)

    Returns:
        List of (chunk_idx, features_dict) tuples, IN ORDER of chunk_idx.
        features_dict is output of extract_all_features().
    """
    from app.features.pipeline import extract_all_features

    n_workers = max_workers or _N_FEATURE_WORKERS

    results: Dict[int, dict] = {}
    errors: Dict[int, Exception] = {}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(extract_all_features, chunk["audio_np"]): chunk["chunk_idx"]
            for chunk in chunks
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Feature extraction failed for chunk {idx}: {e}")
                errors[idx] = e

    # Return in original order, skip failed chunks
    ordered = []
    for chunk in chunks:
        idx = chunk["chunk_idx"]
        if idx in results:
            ordered.append((idx, results[idx]))
        else:
            logger.warning(f"Skipping chunk {idx} — extraction failed: {errors.get(idx)}")

    logger.info(f"Extracted features: {len(ordered)}/{len(chunks)} chunks OK")
    return ordered
