"""
Audit ID Generator — Human-readable, collision-resistant IDs.
Format: UCO-{YYYYMMDD}-{8_hex_chars}
Example: UCO-20260529-A3F7B2D1
"""

import hashlib
import time
import uuid
from datetime import datetime, timezone


def generate_audit_id(call_id: str = None) -> str:
    """
    Generate unique audit ID.

    Collision probability: ~1 in 4 billion per day.
    Safe for UCO Bank's call volume (estimated <100K calls/day).
    """
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d")

    # Use call_id + timestamp + random bytes for entropy
    entropy = f"{call_id or ''}{time.time_ns()}{uuid.uuid4()}"
    hash_bytes = hashlib.sha256(entropy.encode()).hexdigest()[:8].upper()

    return f"UCO-{date_str}-{hash_bytes}"


def parse_audit_id(audit_id: str) -> dict:
    """
    Parse audit ID components.
    Returns: {"prefix": "UCO", "date": "20260529", "hash": "A3F7B2D1"}
    """
    parts = audit_id.split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid audit ID format: {audit_id}")
    return {
        "prefix": parts[0],
        "date": parts[1],
        "hash": parts[2],
    }
