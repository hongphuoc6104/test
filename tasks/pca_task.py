
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib
from prefect import task
from utils.config_loader import load_config  # giả sử bạn đã có sẵn
import os

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

    if "emb" not in df.columns or "global_id" not in df.columns:
        raise ValueError("[PCA] Input parquet must contain 'global_id' and 'emb' columns.")

    X = np.array(df["emb"].tolist())  # shape: (N, D)
    print(f"[PCA] Loaded {X.shape[0]} embeddings with dimension {X.shape[1]}.")

    # 3. Fit PCA
    n_components = pca_cfg.get("n_components", 128)
    whiten = pca_cfg.get("whiten", True)

    print(f"[PCA] Fitting PCA with n_components={n_components}, whiten={whiten} ...")
    pca_model = PCA(n_components=n_components, whiten=whiten, random_state=42)
    pca_model.fit(X)

    # 4. Save PCA model
    os.makedirs(os.path.dirname(pca_model_path), exist_ok=True)
    joblib.dump(pca_model, pca_model_path)
    print(f"[PCA] PCA model saved to {pca_model_path}")

    # 5. Transform embeddings
    X_pca = pca_model.transform(X)
    print(f"[PCA] Transformed embeddings shape: {X_pca.shape}")

    # 6. Save reduced embeddings
    print("[PCA] Creating new DataFrame with reduced embeddings...")

    # (FIX) Copy DataFrame gốc để giữ lại tất cả metadata, và xóa cột 'emb' cũ
    df_pca = df.drop(columns=['emb']).copy()

    # Gán lại cột 'emb' bằng dữ liệu đã được giảm chiều
    df_pca['emb'] = X_pca.tolist()

    os.makedirs(os.path.dirname(embeddings_pca_path), exist_ok=True)
    df_pca.to_parquet(embeddings_pca_path, index=False)

    print(f"[PCA] Reduced embeddings saved to {embeddings_pca_path}")
    print("[PCA] Task completed successfully ✅")

if __name__ == "__main__":
    pca_task.fn()  # dùng .fn() để gọi hàm bên trong task Prefect
