import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from prefect import task
from utils.config_loader import load_config
from utils.indexer import build_index
from utils.vector_utils import _mean_vector
from tasks.filter_clusters_task import filter_clusters_task

@task(name="Build Character Profiles Task")
def character_task():
    """
    Hợp nhất các cụm nhỏ và tạo hồ sơ nhân vật cuối cùng.
    """
    print("\n--- Starting Character Profile Task ---")
    cfg = load_config()
    # (FIX) Đọc tất cả các đường dẫn từ mục "storage" duy nhất
    storage_cfg = cfg.get("storage", {})
    post_merge_cfg = cfg.get("post_merge", {})

    clusters_path = storage_cfg["warehouse_clusters"]
    output_json_path = storage_cfg["characters_json"]
    merged_parquet_path = storage_cfg.get("clusters_merged_parquet")  # Tùy chọn

    # === BƯỚC 1: Load dữ liệu ===
    print(f"[Character] Loading clustered data from {clusters_path}...")
    df = pd.read_parquet(clusters_path)
    if df.empty:
        print("[Character] No clustered data to process. Skipping task.")
        return None

    # === BƯỚC 2: Tính centroid cho từng cụm gốc ===
    print("[Character] Calculating initial centroids...")
    centroids_df = (
        df.groupby("cluster_id")["emb"]
        .apply(lambda embs: _mean_vector(embs.tolist()))
        .reset_index()
    )
    base_cluster_ids = centroids_df["cluster_id"].tolist()
    centroid_vectors = np.stack(centroids_df["emb"].values)

    # === BƯỚC 3: Gom cụm tầng hai (Post-merge) ===
    if post_merge_cfg.get("enable", False):
        print("[Character] Post-merging enabled. Clustering centroids...")
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(post_merge_cfg.get("distance_threshold", 0.35)),
            metric=post_merge_cfg.get("metric", "cosine"),
            linkage=post_merge_cfg.get("linkage", "average"),
        )
        super_labels = clusterer.fit_predict(centroid_vectors)
        mapping = dict(zip(base_cluster_ids, super_labels))
        df["final_character_id"] = df["cluster_id"].map(mapping)
    else:
        print("[Character] Post-merging disabled. Using original cluster IDs.")
        df["final_character_id"] = df["cluster_id"]

    # --- Lưu file Parquet trung gian để debug ---
    if merged_parquet_path:
        os.makedirs(os.path.dirname(merged_parquet_path), exist_ok=True)
        df.to_parquet(merged_parquet_path, index=False)
        print(f"[Character] Saved merged cluster data to {merged_parquet_path}")

    # === BƯỚC 4: Tạo hồ sơ nhân vật cuối cùng ===
    print("[Character] Building final character profiles...")
    characters = {}
    for char_id, group in df.groupby("final_character_id"):
        if pd.isna(char_id):
            continue

        rep = group.loc[group["det_score"].idxmax()]

        characters[str(int(char_id))] = {
            "count": int(len(group)),
            "movies": sorted(group["movie"].unique().tolist()),
            "rep_image": {
                "movie": rep["movie"],
                "frame": rep["frame"],
                "bbox": rep["bbox"] if isinstance(rep["bbox"], list) else rep["bbox"].tolist(),
                "det_score": float(rep["det_score"]),
            },
        }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(characters, f, indent=2, ensure_ascii=False)

    print(f"[Character] Saved {len(characters)} character profiles to {output_json_path}")

    # Lọc các cụm kém chất lượng trước khi xây dựng index
    output_json_path = filter_clusters_task()

    # Xây dựng index cho các nhân vật
    index_path = storage_cfg.get("index_path")
    if index_path:
        print(f"[Character] Building index at {index_path}...")
        build_index(output_json_path, index_path)
    print("[Character] Task completed successfully ✅")
    return output_json_path
