"""Utilities for searching characters using face embeddings."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Union

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


def search_actor(
    image_path: str,
    k: int = 5,
    min_count: int = 0,
    return_emb: bool = False,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Find the closest characters in the index based on an image."""
    cfg = load_config()
    emb_cfg = cfg["embedding"]
    storage_cfg = cfg["storage"]
    pca_cfg = cfg.get("pca", {})
    index_cfg = cfg.get("index", {})

    index, id_map = load_index()
    index_type = str(index_cfg.get("type", "")).lower()

    app = FaceAnalysis(name=emb_cfg["model"], providers=emb_cfg["providers"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    img = cv2.imread(image_path)
    if img is None:
        return {} if return_emb else []

    faces = app.get(img)
    if not faces:
        return {} if return_emb else []

    # lấy mặt có det_score cao nhất
    faces.sort(key=lambda f: f.det_score, reverse=True)
    emb = faces[0].embedding.astype("float32")

    # normalize trước PCA (embedding gốc thường đã chuẩn)
    if emb_cfg.get("l2_normalize", True):
        emb = l2_normalize(emb)

    # nếu index đang ở chiều khác, cố gắng áp PCA để khớp
    index_dim = getattr(index, "d", emb.shape[0])  # FAISS có .d; Annoy không có
    if pca_cfg.get("enable", False) or emb.shape[0] != index_dim:
        try:
            from joblib import load
            pca_path = storage_cfg.get("pca_model", "models/pca_model.joblib")
            if os.path.exists(pca_path):
                pca = load(pca_path)
                emb = pca.transform(emb.reshape(1, -1)).astype("float32")[0]
        except Exception:
            # nếu có lỗi khi load/transform PCA, giữ nguyên emb
            pass

    # sau PCA, norm bị thay đổi -> normalize lại nếu dùng IP/cosine
    if "ip" in index_type:
        emb = l2_normalize(emb)

    emb = emb.reshape(1, -1).astype("float32")

    with open(storage_cfg["characters_json"], "r", encoding="utf-8") as f:
        characters = json.load(f)

    def _search_func(
        query_emb: np.ndarray, top_k: int = k, min_count: int = min_count
    ) -> List[Dict[str, Any]]:
        """Search the loaded index using the provided embedding."""
        q = np.asarray(query_emb, dtype="float32")
        if q.ndim == 1:
            q = q[None, :]

        # đảm bảo normalize khi index là inner-product (cosine)
        if "ip" in index_type:
            # normalize theo hàng
            norms = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
            q = q / norms

        distances, indices = _query_index(index, q, top_k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances, indices):
            char_id = str(id_map.get(int(idx), idx))
            char_info = characters.get(char_id, {})
            movies = char_info.get("movies", [])
            count = char_info.get("count", 0)
            if count < min_count:
                continue

            rep_image = char_info.get("rep_image", {})
            preview_paths = char_info.get("preview_paths", [])
            # ensure preview paths are absolute
            previews_root = storage_cfg.get("cluster_previews_root", "")
            preview_paths = [
                p if os.path.isabs(p) else os.path.join(previews_root, p)
                for p in preview_paths
            ]

            results.append(
                {
                    "character_id": char_id,
                    "movies": movies,
                    "distance": float(dist),
                    "count": count,
                    "rep_image": rep_image,
                    "preview_paths": preview_paths,
                }
            )
        return results

    if return_emb:
        return {
            "embedding": emb[0].tolist(),
            "search_func": _search_func,
        }

    return _search_func(emb, min_count=min_count)
