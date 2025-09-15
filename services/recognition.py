"""Utilities for recognizing faces using a configured FAISS index."""

from __future__ import annotations

from typing import Any, Dict, List

from utils.config_loader import load_config
from utils.search_actor import search_actor


def recognize(image_path: str, top_k: int | None = None) -> Dict[str, Any]:
    """Recognize a face image using the configured index.

    Always returns the top-K nearest characters. The ``is_unknown`` flag is
    determined by comparing the best distance against a threshold.

    Parameters
    ----------
    image_path:
        Path to the image containing the query face.

    Returns
    -------
    dict
        ``is_unknown`` – whether the query falls below the threshold.
        ``candidates`` – list of top-K matches including image paths.
    """

    cfg = load_config()
    search_cfg = cfg.get("search", {})
    threshold = float(search_cfg.get("threshold", 0.5))
    if top_k is None:
        top_k = int(search_cfg.get("top_k", 5))

    matches: List[Dict[str, Any]] = search_actor(image_path, k=top_k)
    if not matches:
        return {"is_unknown": True, "candidates": []}

    best_score = matches[0].get("distance", 0.0)
    is_unknown = best_score < threshold
    return {"is_unknown": is_unknown, "candidates": matches}
