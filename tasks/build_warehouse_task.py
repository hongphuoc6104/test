import os
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from prefect import task
from utils.config_loader import load_config

@task(name="Build Warehouse Task")
def build_warehouse_task():
    """
    Gom tất cả các file embeddings parquet (mỗi phim một file)
    thành một file embeddings.parquet duy nhất trong warehouse/parquet/.
    """
    # 1. Đọc config
    config = load_config()
    metadata_json_path = config["storage"]["metadata_json"]
    warehouse_path = config["storage"].get("embeddings_parquet") or config["storage"]["warehouse_embeddings"]
    embeddings_parquet = config["storage"]["warehouse_embeddings"]

    # Tạo thư mục warehouse/parquet nếu chưa có
    os.makedirs(os.path.dirname(embeddings_parquet), exist_ok=True)

    # 2. Đọc metadata.json
    if not os.path.exists(metadata_json_path):
        raise FileNotFoundError(f"Không tìm thấy file metadata.json: {metadata_json_path}")

    with open(metadata_json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    writer = None
    schema = None
    total_rows = 0
    print("Bắt đầu gom các file Parquet vào warehouse...")

    for movie, info in metadata.items():
        embedding_path = info.get("embedding_file_path")

        if embedding_path and os.path.exists(embedding_path):
            print(f"  -> Đang xử lý: {movie}")
            table = pq.read_table(embedding_path)

            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(warehouse_path, schema)
            writer.write_table(table)
            total_rows += len(table)

        else:
            print(f"[WARN] Không tìm thấy embeddings cho {movie} tại {embedding_path}")

    if writer:
        writer.close()
        print(f"\n[INFO] Warehouse embeddings ({total_rows} records) được lưu tại: {warehouse_path}")
    else:
        raise RuntimeError("Không có dữ liệu nào để gom. Hãy chạy embedding_task trước.")

    return True

if __name__ == "__main__":
    build_warehouse_task()