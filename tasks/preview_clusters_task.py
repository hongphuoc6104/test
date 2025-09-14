import os
import shutil
from typing import Optional

import cv2
import pandas as pd
from prefect import task
from utils.config_loader import load_config


@task(name="Preview Clusters Task")
def preview_clusters_task(max_images_per_cluster: Optional[int] = 3):
    """Tạo thư mục preview cho từng cụm ảnh.

    Đọc file ``clusters.parquet`` và với mỗi ``cluster_id`` tạo một thư mục
    ``cluster_{id}`` nằm trong ``storage.cluster_previews_root``. Trong mỗi thư
    mục con lưu tối đa ``max_images_per_cluster`` khung hình đại diện. Mỗi khung
    hình được lưu dưới dạng ảnh gốc và ảnh có vẽ bbox.

    Args:
        max_images_per_cluster: Số khung hình đại diện sẽ lưu cho mỗi cụm. Mặc
            định là 3.
    Returns:
        Đường dẫn gốc nơi chứa các thư mục preview.
    """
    print("\n--- Starting Preview Clusters Task ---")
    cfg = load_config()
    storage_cfg = cfg["storage"]
    clusters_path = storage_cfg["warehouse_clusters"]
    previews_root = storage_cfg["cluster_previews_root"]
    frames_root = storage_cfg["frames_root"]

    os.makedirs(previews_root, exist_ok=True)
    df = pd.read_parquet(clusters_path)

    for cluster_id, group in df.groupby("cluster_id"):
        cluster_dir = os.path.join(previews_root, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)

        subset = (
            group.sort_values("det_score", ascending=False)
            .drop_duplicates(subset=["frame"])
            .head(max_images_per_cluster)
        )
        for idx, row in subset.iterrows():
            frame_path = os.path.join(frames_root, row["movie"], row["frame"])
            if not os.path.exists(frame_path):
                continue

            base_name = f"{idx:02d}_{os.path.splitext(row['frame'])[0]}"
            orig_dst = os.path.join(cluster_dir, f"{base_name}.jpg")
            bbox_dst = os.path.join(cluster_dir, f"{base_name}_bbox.jpg")

            if not os.path.exists(orig_dst):
                try:
                    os.symlink(os.path.abspath(frame_path), orig_dst)
                except OSError:
                    try:
                        shutil.copy(frame_path, orig_dst)
                    except Exception as e:  # noqa: BLE001
                        print(f"[WARN] Could not copy {frame_path} -> {orig_dst}: {e}")

            img = cv2.imread(frame_path)
            if img is None:
                continue
            bbox = row.get("bbox")
            if bbox is None:
                continue
            if not isinstance(bbox, (list, tuple)):
                bbox = bbox.tolist()
            x1, y1, x2, y2 = map(int, bbox)
            annotated = img.copy()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(bbox_dst, annotated)

    print(f"[INFO] Cluster previews generated at {previews_root}")
    return previews_root
