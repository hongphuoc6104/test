from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from prefect import task
from utils.config_loader import load_config

def filter_clusters(
    df: pd.DataFrame,
    min_track_size: int = 1,
    min_cluster_size: int = 1,
) -> pd.DataFrame:
    """Filter out tracks and clusters that are too small."""

    original_len = len(df)

    if min_track_size > 1 and "track_size" in df.columns:
        df = df[df["track_size"] >= min_track_size]

    if min_cluster_size > 1:
        counts = df.groupby("cluster_id")["track_id"].count()
        valid_clusters = counts[counts >= min_cluster_size].index
        df = df[df["cluster_id"].isin(valid_clusters)]

    removed = original_len - len(df)
    if removed > 0:
        print(f"[INFO] Filtered out {removed} tracks/clusters based on config.")

    return df.copy()


@task(name="Cluster Faces Task")
def cluster_task():
    """
    Gom cụm embeddings bằng thuật toán được cấu hình và bảo toàn metadata.
    """

    config = load_config()
    storage_cfg = config["storage"]
    clustering_cfg = config["clustering"]
    algo = clustering_cfg.get("algo", "auto").lower()
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

    df_tracks = df.drop_duplicates(subset=[group_col, "track_id"])

    print(f"[INFO] Đã load {len(df_tracks)} track centroids để gom cụm.")

    metric = clustering_cfg.get("metric", "cosine")

    # Gom cụm theo từng phim
    results = []
    for movie_key, group in df_tracks.groupby(group_col):
        print(f"Processing {group_col}={movie_key}...")
        emb_matrix = np.array(group["track_centroid"].tolist(), dtype=np.float32)

        if len(group) > 1:
            # Determine distance threshold
            auto_percentile = clustering_cfg.get("auto_distance_percentile")
            dist_matrix = pairwise_distances(emb_matrix, metric=metric)
            if auto_percentile is not None:
                triu = dist_matrix[np.triu_indices(len(dist_matrix), k=1)]
                dist_th = float(np.percentile(triu, auto_percentile))
                print(
                    f"[INFO] Inferred distance_threshold={dist_th:.4f} at p{auto_percentile}"
                )
            else:
                if pca_cfg.get("enable", False):
                    dist_cfg = clustering_cfg.get(
                        "distance_threshold_pca",
                        clustering_cfg.get("distance_threshold", 0.7),
                    )
                else:
                    dist_cfg = clustering_cfg.get("distance_threshold", 0.7)
                if isinstance(dist_cfg, dict):
                    default_th = float(dist_cfg.get("default", 0.7))
                    per_movie = dist_cfg.get("per_movie", {})
                    dist_th = float(per_movie.get(str(movie_key), default_th))
                else:
                    dist_th = float(dist_cfg)
                print(f"[INFO] Using configured distance_threshold={dist_th:.4f}")

            # Baseline Agglomerative clustering
            aggl_clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=dist_th,
                metric=metric,
                linkage=clustering_cfg.get("linkage", "complete"),
            )
            aggl_labels = aggl_clusterer.fit_predict(emb_matrix)
            n_aggl = len(set(aggl_labels))
            sil_score = (
                silhouette_score(emb_matrix, aggl_labels, metric=metric)
                if n_aggl > 1
                else -1
            )
            print(
                f"[DEBUG] Agglomerative produced {n_aggl} clusters (silhouette={sil_score:.3f})"
            )

            chosen_labels = aggl_labels
            final_algo = "agglomerative"
            final_clusters = n_aggl

            if algo in {"auto", "hdbscan"}:
                import hdbscan

                hdbscan_cfg = clustering_cfg.get("hdbscan", {})
                if hdbscan_cfg.get("use_precomputed", False):
                    dist_hdb = 1 - cosine_similarity(emb_matrix)
                    hdb_clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=int(
                            hdbscan_cfg.get("min_cluster_size", 5)
                        ),
                        min_samples=int(hdbscan_cfg.get("min_samples", 5)),
                        metric="precomputed",
                    )
                    hdb_labels = hdb_clusterer.fit_predict(dist_hdb)
                else:
                    hdb_clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=int(
                            hdbscan_cfg.get("min_cluster_size", 5)
                        ),
                        min_samples=int(hdbscan_cfg.get("min_samples", 5)),
                        metric=metric,
                    )
                    hdb_labels = hdb_clusterer.fit_predict(emb_matrix)

                n_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
                outlier_ratio = float(np.mean(hdb_labels == -1))
                print(
                    f"[DEBUG] HDBSCAN produced {n_hdb} clusters (outliers={outlier_ratio:.2%})"
                )

                if algo == "hdbscan" or (
                    algo == "auto"
                    and (n_aggl <= 1 or sil_score < 0.2)
                    and n_hdb > 0
                    and outlier_ratio < 0.5
                ):
                    chosen_labels = hdb_labels
                    final_algo = "hdbscan"
                    final_clusters = n_hdb

            if algo == "auto":
                print(
                    f"[INFO] Cluster count change: {n_aggl} -> {final_clusters} using {final_algo}"
                )
        else:
            chosen_labels = np.array([0])
            final_clusters = 1
            final_algo = algo

        group = group.copy()
        group["cluster_id"] = [f"{movie_key}_{lbl}" for lbl in chosen_labels]
        results.append(group)

    # Giữ lại TẤT CẢ các cột khi lưu kết quả và lọc các cụm chất lượng thấp
    clusters_df = pd.concat(results, ignore_index=True)
    filter_cfg = config.get("filter", {})
    clusters_df = filter_clusters(
        clusters_df,
        min_track_size=int(filter_cfg.get("min_track_size", 1)),
        min_cluster_size=int(filter_cfg.get("min_cluster_size", 1)),
    )

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
