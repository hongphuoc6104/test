import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from prefect import task
import os
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
    if "emb" not in df.columns:
        raise ValueError("[Cluster] Input parquet must contain 'emb' column.")

    print(f"[INFO] Đã load {len(df)} embeddings để gom cụm.")
    embeddings_matrix = np.array(df["emb"].tolist(), dtype=np.float32)

    # Chạy Agglomerative Clustering
    print("Running Agglomerative Clustering...")
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=clustering_cfg["distance_threshold"],
        metric=clustering_cfg["metric"],
        linkage=clustering_cfg["linkage"],
    )
    labels = clusterer.fit_predict(embeddings_matrix)
    df["cluster_id"] = labels

    # Giữ lại TẤT CẢ các cột khi lưu kết quả
    clusters_df = df.copy()


    # Logic thống kê
    n_clusters = len(set(labels))
    print("\n=== KẾT QUẢ CLUSTERING ===")
    print(f"Số cụm nhân vật tìm được: {n_clusters}")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  → Cụm {label}: {count} khuôn mặt")
    print("==========================")

    # Lưu kết quả
    clusters_path = storage_cfg["warehouse_clusters"]

    clusters_df.to_parquet(clusters_path, index=False)


    print(f"\n[INFO] Đã lưu {len(clusters_df)} cluster records -> {clusters_path}")
    return clusters_path