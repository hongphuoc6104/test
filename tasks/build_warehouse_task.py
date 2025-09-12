import os
import json
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

    print("Bắt đầu gom các file Parquet vào warehouse...")

    embedding_files = []
    for movie, info in metadata.items():
        embedding_path = info.get("embedding_file_path")

        if embedding_path and os.path.exists(embedding_path):
            embedding_files.append((movie, embedding_path))
        else:
            print(f"[WARN] Không tìm thấy embeddings cho {movie} tại {embedding_path}")

    if not embedding_files:
        raise RuntimeError("Không có dữ liệu nào để gom. Hãy chạy embedding_task trước.")

    first_movie, first_path = embedding_files[0]
    first_table = pq.read_table(first_path)
    schema = first_table.schema
    total_rows = 0

    with pq.ParquetWriter(warehouse_path, schema) as writer:
        print(f"  -> Đang xử lý: {first_movie}")
        writer.write_table(first_table)
        total_rows += len(first_table)

        for movie, path in embedding_files[1:]:
            print(f"  -> Đang xử lý: {movie}")
            table = pq.read_table(path)
            writer.write_table(table)
            total_rows += len(table)

    print(f"\n[INFO] Warehouse embeddings ({total_rows} records) được lưu tại: {warehouse_path}")

    return True

if __name__ == "__main__":
    build_warehouse_task()
