import os
import json
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
    warehouse_path = config["storage"]["warehouse_embeddings"]
    os.makedirs(os.path.dirname(warehouse_path), exist_ok=True)

    # 2. Đọc metadata.json
    if not os.path.exists(metadata_json_path):
        raise FileNotFoundError(
            f"Không tìm thấy file metadata.json: {metadata_json_path}"
        )

    with open(metadata_json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("Bắt đầu gom các file Parquet vào warehouse...")

    embedding_files: list[tuple[str, str]] = []
    for movie, info in metadata.items():
        embedding_path = info.get("embedding_file_path")

        if embedding_path and os.path.exists(embedding_path):
            embedding_files.append((movie, embedding_path))
        else:
            print(f"[WARN] Không tìm thấy embeddings cho {movie} tại {embedding_path}")

    if not embedding_files:
        raise RuntimeError("Không có dữ liệu nào để gom. Hãy chạy embedding_task trước.")

    # Gán movie_id cho từng phim dựa trên thứ tự xuất hiện
    embedding_files_with_id = [
        (movie, path, movie_id)
        for movie_id, (movie, path) in enumerate(embedding_files)
    ]

    # Đọc bảng đầu tiên và thêm cột movie_id
    _, first_path, first_id = embedding_files_with_id[0]
    first_table = pq.read_table(first_path)
    first_table = first_table.append_column(
        "movie_id", pa.array([first_id] * first_table.num_rows, pa.int32())
    )
    schema = first_table.schema

    # 3. Ghi các bảng vào file warehouse sử dụng ParquetWriter
    with pq.ParquetWriter(warehouse_path, schema) as writer:
        # Ghi bảng đầu tiên
        writer.write_table(first_table)

        # Ghi các bảng còn lại
        for movie, path, movie_id in embedding_files_with_id[1:]:
            table = pq.read_table(path)
            table = table.append_column(
                "movie_id", pa.array([movie_id] * table.num_rows, pa.int32())
            )
            if table.schema != schema:
                table = table.cast(schema)
            writer.write_table(table)

    print(f"Đã gom {len(embedding_files_with_id)} file vào {warehouse_path}")
    return warehouse_path
