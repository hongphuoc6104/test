import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("prefect")
pytest.importorskip("yaml")

from sklearn.cluster import AgglomerativeClustering
from prefect import task
from utils.config_loader import load_config

import numpy as np
import pandas as pd


def filter_clusters(df: 'pd.DataFrame', min_det: float = 0.5, min_size: int = 3) -> 'pd.DataFrame':
    stats = (
        df.groupby("cluster_id")
        .agg(median_det=("det_score", "median"), size=("track_id", "size"))
    )
    valid = stats[(stats["median_det"] >= min_det) & (stats["size"] >= min_size)].index
    removed = len(stats) - len(valid)
    if removed > 0:
        print(f"[INFO] Filtered out {removed} low-quality clusters.")
    return df[df["cluster_id"].isin(valid)].copy()


@task(name="Cluster Faces Task")
def cluster_task():
    """
    Gom cụm embeddings bằng thuật toán được cấu hình và bảo toàn metadata.
    """

    config = load_config()
    storage_cfg = config["storage"]
    clustering_cfg = config["clustering"]
    algo = clustering_cfg.get("algo", "agglomerative").lower()
    pca_cfg = config.get("pca", {})

    print(f"\n--- Starting Cluster Task ({algo.capitalize()}) ---")

    # Logic chọn file input một cách thông minh
    if pca_cfg.get("enable", False):
        embeddings_path = storage_cfg["warehouse_embeddings_pca"]
        print("[INFO] Clustering on PCA-reduced data.")
    else:
        embeddings_path = storage_cfg["warehouse_embeddings"]
        print("[INFO] Clustering on original 512-dim data.")

    df = pd.read_parquet(embeddings_path)

    if "track_centroid" not in df.columns:
        raise ValueError("[Cluster] Input parquet must contain column: track_centroid")

    # Xác định cột dùng để gom cụm theo phim
    if "movie_id" in df.columns:
        group_col = "movie_id"
    elif "movie" in df.columns:
        group_col = "movie"
        print("[INFO] movie_id column not found. Grouping by movie.")
    else:
        print("[WARN] No movie information found. Treating entire dataset as one movie.")
        group_col = "_movie_tmp"
        df[group_col] = 0

    track_scores = (
        df.groupby([group_col, "track_id"])["det_score"].median().rename("det_score").reset_index()
    )
    df_tracks = (
        df.drop_duplicates(subset=[group_col, "track_id"]).drop(columns=["det_score"])
        .merge(track_scores, on=[group_col, "track_id"], how="left")
    )

    print(f"[INFO] Đã load {len(df_tracks)} track centroids để gom cụm.")

    # Gom cụm theo từng phim
    results = []
    for movie_key, group in df_tracks.groupby(group_col):
        print(
            f"Running {algo.capitalize()} Clustering for {group_col}={movie_key}..."
        )
        emb_matrix = np.array(group["track_centroid"].tolist(), dtype=np.float32)

        if len(group) > 1:
            if algo == "hdbscan":
                import hdbscan

                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=int(
                        clustering_cfg.get("min_cluster_size", 5)
                    ),
                    metric=clustering_cfg.get("metric", "euclidean"),
                )
                labels = clusterer.fit_predict(emb_matrix)
            else:
                clusterer = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=float(
                        clustering_cfg.get("distance_threshold", 0.7)
                    ),
                    metric="cosine",
                    linkage="complete",
                )
                labels = clusterer.fit_predict(emb_matrix)
        else:
            labels = np.array([0])

        group = group.copy()
        group["cluster_id"] = [f"{movie_key}_{lbl}" for lbl in labels]
        results.append(group)

    # Giữ lại TẤT CẢ các cột khi lưu kết quả
    clusters_df = pd.concat(results, ignore_index=True)
    clusters_df = filter_clusters(clusters_df)

    # Logic thống kê
    unique_labels, counts = np.unique(clusters_df["cluster_id"], return_counts=True)
    print("\n=== KẾT QUẢ CLUSTERING ===")
    print(f"Số cụm nhân vật tìm được: {len(unique_labels)}")
    for label, count in zip(unique_labels, counts):
        print(f"  → Cụm {label}: {count} khuôn mặt")
    print("==========================")

    # Lưu kết quả
    clusters_path = storage_cfg["warehouse_clusters"]

    clusters_df.to_parquet(clusters_path, index=False)

    print(f"\n[INFO] Đã lưu {len(clusters_df)} cluster records -> {clusters_path}")
    return clusters_path
