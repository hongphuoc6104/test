import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from prefect import task
from utils.config_loader import load_config

@task(name="Cluster Faces Task")
def cluster_task():
    """
    Gom cụm embeddings bằng Agglomerative Clustering và bảo toàn metadata.
    """
    print("\n--- Starting Cluster Task (Agglomerative) ---")

    config = load_config()
    # (FIX) Đọc tất cả các đường dẫn từ mục "storage" duy nhất
    storage_cfg = config["storage"]
    clustering_cfg = config["clustering"]
    pca_cfg = config.get("pca", {})

    # Logic chọn file input một cách thông minh
    if pca_cfg.get("enable", False):
        embeddings_path = storage_cfg["warehouse_embeddings_pca"]
        print("[INFO] Clustering on PCA-reduced data.")
    else:
        embeddings_path = storage_cfg["warehouse_embeddings"]
        print("[INFO] Clustering on original 512-dim data.")

    df = pd.read_parquet(embeddings_path)
    required_cols = {"emb", "movie_id"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"[Cluster] Input parquet must contain columns: {', '.join(missing_cols)}"
        )

    print(f"[INFO] Đã load {len(df)} embeddings để gom cụm.")

    # Gom cụm theo từng movie_id
    results = []
    for movie_id, group in df.groupby("movie_id"):
        print(f"Running Agglomerative Clustering for movie_id={movie_id}...")
        emb_matrix = np.array(group["emb"].tolist(), dtype=np.float32)

        if len(group) > 1:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=clustering_cfg["distance_threshold"],
                metric=clustering_cfg["metric"],
                linkage=clustering_cfg["linkage"],
            )
            labels = clusterer.fit_predict(emb_matrix)
        else:
            labels = np.array([0])

        group = group.copy()
        group["cluster_id"] = [f"{movie_id}_{lbl}" for lbl in labels]
        results.append(group)

    # Giữ lại TẤT CẢ các cột khi lưu kết quả
    clusters_df = pd.concat(results, ignore_index=True)

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
