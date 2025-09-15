"""Utilities for recognizing faces using a configured FAISS index."""

from __future__ import annotations

from typing import Any, Dict, List

from utils.config_loader import load_config
from utils.search_actor import search_actor


def recognize(image_path: str) -> Dict[str, Any]:
    """Recognize a face image using the configured index.

    Parameters
    ----------
    image_path:
        Path to the image containing the query face.

    Returns
    -------
    dict
        Dictionary containing two keys:
        ``is_unknown`` (bool) – whether no match satisfied the threshold.
        ``matches`` (list) – top-k matches with their scores when available.
    """

    cfg = load_config()
    search_cfg = cfg.get("search", {})
    threshold = float(search_cfg.get("threshold", 0.5))
    top_k = int(search_cfg.get("top_k", 5))

    matches: List[Dict[str, Any]] = search_actor(image_path, k=top_k)

    if not matches:
        return {"is_unknown": True, "matches": []}

    best_score = matches[0].get("distance", 0.0)
    if best_score < threshold:
        return {"is_unknown": True, "matches": []}

    return {"is_unknown": False, "matches": matches}
