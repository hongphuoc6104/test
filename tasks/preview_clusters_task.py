import os
import shutil
from typing import Optional

import pandas as pd
from prefect import task
from utils.config_loader import load_config


@task(name="Preview Clusters Task")
def preview_clusters_task(max_images_per_cluster: Optional[int] = None):
    """Tạo thư mục preview cho từng cụm ảnh.

    Đọc file ``clusters.parquet`` và với mỗi ``cluster_id`` tạo một thư mục
    ``cluster_{id}`` nằm trong ``storage.cluster_previews_root``.  Trong mỗi thư
    mục con copy hoặc tạo symlink các ảnh từ cột ``face_crop_path``.

    Args:
        max_images_per_cluster: Giới hạn số ảnh sẽ lấy cho mỗi cụm. ``None``
            nghĩa là lấy toàn bộ ảnh của cụm.
    Returns:
        Đường dẫn gốc nơi chứa các thư mục preview.
    """
    print("\n--- Starting Preview Clusters Task ---")
    cfg = load_config()
    storage_cfg = cfg["storage"]
    clusters_path = storage_cfg["warehouse_clusters"]
    previews_root = storage_cfg["cluster_previews_root"]

    os.makedirs(previews_root, exist_ok=True)
    df = pd.read_parquet(clusters_path)

    for cluster_id, group in df.groupby("cluster_id"):
        cluster_dir = os.path.join(previews_root, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)

        subset = group.head(max_images_per_cluster) if max_images_per_cluster else group
        for _, row in subset.iterrows():
            src = row.get("face_crop_path")
            if not src or not os.path.exists(src):
                continue
            dst = os.path.join(cluster_dir, os.path.basename(src))
            if os.path.exists(dst):
                continue
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                try:
                    shutil.copy(src, dst)
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] Could not copy {src} -> {dst}: {e}")

    print(f"[INFO] Cluster previews generated at {previews_root}")
    return previews_root