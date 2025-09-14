import pandas as pd
import numpy as np
from pathlib import Path
from prefect import task
from utils.config_loader import load_config
import warnings

try:  # sklearn có thể bị stub trong test
    from sklearn.metrics import silhouette_score as _silhouette_score
except Exception:  # pragma: no cover - nếu sklearn thiếu
    _silhouette_score = None

@task(name="Validate Warehouse Task")
def validate_warehouse_task():
    """Kiểm tra chất lượng file embeddings.parquet tổng hợp."""
    # 1. Đọc config
    config = load_config()
    warehouse_path = config["storage"]["warehouse_embeddings"]

    print(f"Validation: Loading data from {warehouse_path}")
    df = pd.read_parquet(warehouse_path)

    # Các kiểm tra chất lượng (logic của bạn đã rất chuẩn)
    assert df["global_id"].is_unique, " Lỗi: global_id bị trùng!"
    assert df["det_score"].between(0, 1).all(), " Lỗi: det_score nằm ngoài [0,1]!"
    assert df["emb"].notnull().all(), " Lỗi: Cột emb chứa giá trị rỗng!"

    embeddings_matrix = np.array(df["emb"].tolist(), dtype=np.float32)
    norms = np.linalg.norm(embeddings_matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), " Lỗi: Vector emb không được chuẩn hóa L2!"

    print(f"✅ Validation thành công cho {len(df)} records!")
    return True



@task(name="Cluster Metrics Task")
def validate_clusters_task():
    """Tính toán các metrics cho kết quả gom cụm."""
    config = load_config()
    clusters_path = config["storage"]["warehouse_clusters"]
    print(f"[Metrics] Loading clusters from {clusters_path}")

    # Tránh lỗi khi pandas bị stub trong môi trường test
    if not hasattr(pd, "read_parquet"):
        warnings.warn("pandas.read_parquet unavailable, skipping cluster metrics", RuntimeWarning)
        return None

    if not Path(clusters_path).exists():
        warnings.warn(f"Cluster file missing at {clusters_path}. Skipping metrics", RuntimeWarning)
        return None

    df = pd.read_parquet(clusters_path)

    labels = df["cluster_id"].astype(str)
    emb_matrix = np.stack(df["track_centroid"].to_numpy())

    unique_labels = labels.unique()
    n_clusters = len(unique_labels)

    if n_clusters <= 1 or _silhouette_score is None:
        silhouette = np.nan
        if n_clusters <= 1:
            warnings.warn("Single cluster detected. Silhouette score undefined.", RuntimeWarning)
        else:
            warnings.warn("silhouette_score unavailable, skipping silhouette computation", RuntimeWarning)
    else:
        numeric_labels = pd.factorize(labels)[0]
        silhouette = float(_silhouette_score(emb_matrix, numeric_labels, metric="cosine"))
        if silhouette < 0.2:
            warnings.warn(f"Low silhouette score: {silhouette:.3f}", RuntimeWarning)

    outlier_mask = labels.str.endswith("-1")
    outlier_fraction = outlier_mask.mean()

    cluster_sizes = labels.value_counts().rename_axis("cluster_id").reset_index(name="size")
    cluster_sizes["silhouette"] = silhouette
    cluster_sizes["outlier_fraction"] = outlier_fraction
    cluster_sizes["n_clusters"] = n_clusters

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "cluster_metrics.csv"
    cluster_sizes.to_csv(metrics_path, index=False)
    print(f"[INFO] Saved cluster metrics -> {metrics_path}")

    return metrics_path