import pandas as pd
import numpy as np
from prefect import task
from utils.config_loader import load_config

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
