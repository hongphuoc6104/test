from prefect import flow
from utils.config_loader import load_config  # <--- IMPORT MỚI
from tasks.ingestion_task import ingestion_task
from utils.indexer import build_index
from tasks.embedding_task import embedding_task
from tasks.build_warehouse_task import build_warehouse_task
from tasks.validation_task import validate_warehouse_task
from tasks.pca_task import pca_task
from tasks.cluster_task import cluster_task
from tasks.character_task import character_task



@flow(name="Face Discovery MVP Pipeline")
def main_pipeline():
    """
    Flow chính điều phối toàn bộ quá trình xử lý dữ liệu.
    """
    print("🚀 Starting Main Pipeline (Synchronous Mode)...")

    # (NÂNG CẤP) Đọc config ngay từ đầu flow
    cfg = load_config()
    storage = cfg.get("storage", {})

    # Chạy các task tuần tự
    ingestion_task()
    print("--- Ingestion Task Completed ---")

    embedding_task()
    print("--- Embedding Task Completed ---")

    build_warehouse_task()
    print("--- Build Warehouse Task Completed ---")

    validate_warehouse_task()
    print("--- Validation Task Completed ---")

    # (NÂNG CẤP) Thêm logic IF/ELSE để quyết định có chạy PCA không
    if cfg.get("pca", {}).get("enable", False):
        print("\n--- PCA is enabled. Running PCA Task... ---")
        pca_task()
        print("--- PCA Task Completed ---")
    else:
        print("\n--- PCA is disabled. Skipping PCA Task. ---")

    cluster_task()
    print("--- Cluster Task Completed ---")

    characters_json = character_task()
    print("--- Character Profile Task Completed ---")

    if characters_json:
        build_index(characters_json, storage["faiss_index"])
        print("--- FAISS Index Built ---")


    print("\n✅✅✅ All tasks completed successfully!")


if __name__ == "__main__":
    main_pipeline()



