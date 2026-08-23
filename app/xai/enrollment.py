"""
Voice Enrollment — Handles speaker enrollment and comparison.
Extracts 256-dim embeddings from the active LCNN model, stores them in
enrolled_voices/ JSON files, and computes cosine similarity.
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
import torch
from app.config import settings

logger = logging.getLogger(__name__)

ENROLLED_DIR = settings.BASE_DIR / "enrolled_voices"
ENROLLED_DIR.mkdir(parents=True, exist_ok=True)


def _get_active_lcnn_and_features(audio_np: np.ndarray, registry) -> tuple:
    """Helper to extract features and model for embedding extraction."""
    from app.features.pipeline import extract_all_features
    from app.xai.embedding import extract_embedding
    
    # Try LFCC model first, then MFCC
    entry = registry.entries.get("lcnn_lfcc")
    feat_key = "lfcc"
    if not entry or not entry.loaded or not entry.model:
        entry = registry.entries.get("lcnn_mfcc")
        feat_key = "mfcc"
        
    if not entry or not entry.loaded or not entry.model:
        raise RuntimeError("No active LCNN model loaded in registry")
        
    features = extract_all_features(audio_np)
    feat_tensor = features.get(feat_key)
    if feat_tensor is None:
        raise ValueError(f"Features not extracted for {feat_key}")
        
    return entry.model, feat_tensor


def enroll_speaker(speaker_name: str, audio_np: np.ndarray, registry) -> dict:
    """Extract embedding and enroll speaker voiceprint."""
    try:
        from app.xai.embedding import extract_embedding
        model, feat_tensor = _get_active_lcnn_and_features(audio_np, registry)
        
        # Extract embedding
        embedding = extract_embedding(model, feat_tensor)
        
        # Save to JSON
        safe_name = speaker_name.replace(" ", "_").replace("/", "").replace("\\", "")
        file_path = ENROLLED_DIR / f"{safe_name}.json"
        
        data = {
            "speaker_name": speaker_name,
            "embedding": embedding.tolist()
        }
        with open(file_path, "w") as f:
            json.dump(data, f)
            
        logger.info(f"Successfully enrolled speaker: {speaker_name}")
        return {"status": "success", "speaker_name": speaker_name}
    except Exception as e:
        logger.error(f"Failed to enroll speaker {speaker_name}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


def list_enrolled_speakers() -> list:
    """List all enrolled speakers."""
    speakers = []
    try:
        for p in ENROLLED_DIR.glob("*.json"):
            with open(p, "r") as f:
                data = json.load(f)
                speakers.append({
                    "id": p.stem,
                    "speaker_name": data.get("speaker_name", p.stem)
                })
    except Exception as e:
        logger.error(f"Failed to list enrolled speakers: {e}")
    return speakers


def delete_enrolled_speaker(speaker_id: str) -> bool:
    """Delete enrolled speaker voiceprint."""
    try:
        file_path = ENROLLED_DIR / f"{speaker_id}.json"
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted enrolled speaker: {speaker_id}")
            return True
    except Exception as e:
        logger.error(f"Failed to delete speaker {speaker_id}: {e}")
    return False


def compare_speaker(audio_np: np.ndarray, registry) -> dict:
    """
    Compare query audio embedding against all enrolled speaker voiceprints.
    Computes cosine similarity. Threshold of 0.75 defines a match.
    """
    try:
        from app.xai.embedding import extract_embedding
        
        enrolled = list_enrolled_speakers()
        if not enrolled:
            return {"match_found": False, "speaker_name": "No Enrolled Speakers", "similarity": 0.0, "all_comparisons": []}
            
        model, feat_tensor = _get_active_lcnn_and_features(audio_np, registry)
        query_emb = extract_embedding(model, feat_tensor)
        
        best_match = None
        best_score = -1.0
        comparisons = []
        
        # Normalize query embedding
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            query_norm = 1e-8
            
        for sp in enrolled:
            file_path = ENROLLED_DIR / f"{sp['id']}.json"
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    ref_emb = np.array(data["embedding"])
                    
                ref_norm = np.linalg.norm(ref_emb)
                if ref_norm == 0: ref_norm = 1e-8
                
                # Cosine similarity
                sim = float(np.dot(query_emb, ref_emb) / (query_norm * ref_norm))
                
                comparisons.append({
                    "speaker_name": sp["speaker_name"],
                    "similarity": round(sim, 4)
                })
                
                if sim > best_score:
                    best_score = sim
                    best_match = sp["speaker_name"]
            except Exception as e:
                logger.error(f"Error loading reference embedding for {sp['id']}: {e}")
                
        # Define biometric voice match threshold (0.75 is standard)
        match_found = best_score >= 0.75
        
        return {
            "match_found": match_found,
            "speaker_name": best_match if match_found else "No Match Found",
            "similarity": round(max(0.0, best_score), 4),
            "all_comparisons": sorted(comparisons, key=lambda x: x["similarity"], reverse=True)
        }
    except Exception as e:
        logger.error(f"Biometric voice comparison failed: {e}", exc_info=True)
        return {
            "match_found": False,
            "speaker_name": "Comparison Error",
            "similarity": 0.0,
            "all_comparisons": []
        }
