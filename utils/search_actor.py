"""Utilities for searching characters using face embeddings."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from utils.config_loader import load_config
from utils.indexer import load_index
from utils.vector_utils import l2_normalize


def _query_index(index: Any, emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Query a FAISS or Annoy index."""
    if hasattr(index, "search"):
        distances, indices = index.search(emb, k)
        return distances[0], indices[0]
    if hasattr(index, "get_nns_by_vector"):
        indices, distances = index.get_nns_by_vector(
            emb[0], k, include_distances=True
        )
        return np.array(distances), np.array(indices)
    raise TypeError("Unsupported index type")


def search_actor(image_path: str, k: int = 5) -> List[Dict[str, Any]]:
    """Find the closest characters in the index based on an image."""
    cfg = load_config()
    emb_cfg = cfg["embedding"]
    storage_cfg = cfg["storage"]

    index, id_map = load_index()

    app = FaceAnalysis(name=emb_cfg["model"], providers=emb_cfg["providers"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    img = cv2.imread(image_path)
    if img is None:
        return []

    faces = app.get(img)
    if not faces:
        return []

    faces.sort(key=lambda f: f.det_score, reverse=True)
    emb = faces[0].embedding
    if emb_cfg.get("l2_normalize", True):
        emb = l2_normalize(emb)

    emb = np.asarray([emb], dtype="float32")
    distances, indices = _query_index(index, emb, k)

    with open(storage_cfg["characters_json"], "r", encoding="utf-8") as f:
        characters = json.load(f)

    results: List[Dict[str, Any]] = []
    for dist, idx in zip(distances, indices):
        char_id = str(id_map.get(int(idx), idx))
        movies = characters.get(char_id, {}).get("movies", [])
        results.append(
            {
                "character_id": char_id,
                "movies": movies,
                "distance": float(dist),
            }
        )

    return results
