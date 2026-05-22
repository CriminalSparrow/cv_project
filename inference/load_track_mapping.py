"""Загрузка track2idx и idx2track из checkpoint."""

from typing import Dict, Tuple


def load_track_mappings(checkpoint) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Загружает track2idx и idx2track из checkpoint.

    Ожидается, что checkpoint содержит ключ "track2idx".
    """
    if not isinstance(checkpoint, dict) or "track2idx" not in checkpoint:
        raise RuntimeError(
            "Track mappings not found in checkpoint. "
            "Checkpoint must contain 'track2idx'."
        )

    track2idx = {
        str(track_id): int(idx)
        for track_id, idx in checkpoint["track2idx"].items()
    }

    idx2track = {
        idx: track_id
        for track_id, idx in track2idx.items()
    }

    return track2idx, idx2track
