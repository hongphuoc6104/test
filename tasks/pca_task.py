import os

import joblib
import numpy as np
import pandas as pd
from prefect import task
from sklearn.decomposition import PCA

from utils.config_loader import load_config  # giả sử bạn đã có sẵn
from utils.vector_utils import l2_normalize

@task
def pca_task():
    """
    Fit PCA trên embeddings, lưu model và tạo file embeddings_pca.parquet.
    """
    # 1. Load config
    cfg = load_config()
    pca_cfg = cfg.get("pca", {})
    storage_cfg = cfg.get("storage", {})

    if not pca_cfg.get("enable", False):
        print("[PCA] PCA is disabled in config. Skipping...")
        return

    embeddings_path = storage_cfg["warehouse_embeddings"]
    embeddings_pca_path = storage_cfg["warehouse_embeddings_pca"]
    pca_model_path = storage_cfg["pca_model"]

    print(f"[PCA] Loading embeddings from {embeddings_path} ...")

    # 2. Load embeddings
    df = pd.read_parquet(embeddings_path, engine="pyarrow")

    required_cols = {"emb", "global_id", "track_centroid"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            "[PCA] Input parquet must contain columns: 'global_id', 'emb', and 'track_centroid'."
        )

    X = np.array(df["emb"].tolist())  # shape: (N, D)
    C = np.array(df["track_centroid"].tolist())  # shape: (N, D)
    print(f"[PCA] Loaded {X.shape[0]} embeddings with dimension {X.shape[1]}.")

    # 3. Fit PCA
    n_components = pca_cfg.get("n_components", 128)
    whiten = pca_cfg.get("whiten", False)

    print(f"[PCA] Fitting PCA with n_components={n_components}, whiten={whiten} ...")
    pca_model = PCA(n_components=n_components, whiten=whiten, random_state=42)
    pca_model.fit(X)

    # 4. Save PCA model
    os.makedirs(os.path.dirname(pca_model_path), exist_ok=True)
    joblib.dump(pca_model, pca_model_path)
    print(f"[PCA] PCA model saved to {pca_model_path}")

    # 5. Transform embeddings and centroids
    X_pca = pca_model.transform(X)
    C_pca = pca_model.transform(C)
    print(f"[PCA] Transformed embeddings shape: {X_pca.shape}")

    # 6. L2-normalize transformed vectors
    X_pca = np.array([l2_normalize(v) for v in X_pca])
    C_pca = np.array([l2_normalize(v) for v in C_pca])

    # 7. Save reduced embeddings
    print("[PCA] Creating new DataFrame with reduced embeddings...")

    # Copy DataFrame gốc để giữ lại tất cả metadata, và xóa các cột vector cũ
    df_pca = df.drop(columns=["emb", "track_centroid"]).copy()

    # Gán lại các cột bằng dữ liệu đã được giảm chiều và chuẩn hóa
    df_pca["emb"] = X_pca.tolist()
    df_pca["track_centroid"] = C_pca.tolist()

    os.makedirs(os.path.dirname(embeddings_pca_path), exist_ok=True)
    df_pca.to_parquet(embeddings_pca_path, index=False)

    print(f"[PCA] Reduced embeddings saved to {embeddings_pca_path}")
    print("[PCA] Task completed successfully ✅")

if __name__ == "__main__":
    pca_task.fn()  # dùng .fn() để gọi hàm bên trong task Prefect
