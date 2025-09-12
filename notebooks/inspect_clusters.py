# notebooks/inspect_clusters.py
import os
import math
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Cấu hình ---
# (FIX) Các đường dẫn trỏ đến output cuối cùng của các task
CLUSTERS_MERGED_PARQUET = "warehouse/parquet/clusters_merged.parquet"  # File có chứa super_cluster_id
FRAMES_BASE_DIR = "Data/frames"
OUT_DIR = "warehouse/debug"
TOP_K_CLUSTERS = 10  # Xem 10 cụm lớn nhất
SAMPLES_PER_CLUSTER = 16  # Mỗi cụm hiển thị tối đa 16 ảnh

os.makedirs(OUT_DIR, exist_ok=True)


def load_df():
    """(FIX) Đơn giản hóa: chỉ cần đọc một file duy nhất đã có đủ thông tin."""
    if not os.path.exists(CLUSTERS_MERGED_PARQUET):
        print(f"Lỗi: Không tìm thấy file {CLUSTERS_MERGED_PARQUET}. Hãy chạy character_task trước.")
        return None
    df = pd.read_parquet(CLUSTERS_MERGED_PARQUET)
    return df


def crop_face(img: Image.Image, bbox: list) -> Image.Image:
    """Cắt vùng khuôn mặt từ ảnh."""
    if not isinstance(bbox, list) or len(bbox) != 4:
        return img

    W, H = img.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(W, int(x2))
    y2 = min(H, int(y2))

    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def load_image(movie: str, frame_filename: str) -> Image.Image:
    """(FIX) Xây dựng đường dẫn ảnh chính xác từ movie và tên frame."""
    image_path = os.path.join(FRAMES_BASE_DIR, movie, frame_filename)
    return Image.open(image_path).convert("RGB")


def make_gallery(faces: list, title: str, out_path: str):
    """Tạo và lưu một bộ sưu tập ảnh."""
    if not faces: return
    n = len(faces)
    cols = min(4, n)
    rows = math.ceil(n / cols)

    plt.figure(figsize=(cols * 2.0, rows * 2.0))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(faces[i])
        plt.axis("off")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.close()


def visualize(df: pd.DataFrame, group_col: str, label_prefix: str):
    """Hàm chính để trực quan hóa các cụm."""
    if group_col not in df.columns:
        print(f"Cảnh báo: Không tìm thấy cột '{group_col}' để trực quan hóa.")
        return

    sizes = df[group_col].value_counts()
    top_ids = list(sizes.head(TOP_K_CLUSTERS).index)
    print(f"\n--- Đang trực quan hóa {len(top_ids)} cụm lớn nhất cho '{label_prefix}' ---")

    for gid in top_ids:
        group_df = df[df[group_col] == gid]
        # Sắp xếp theo det_score để lấy những ảnh đại diện tốt nhất
        sample_df = group_df.sort_values("det_score", ascending=False).head(SAMPLES_PER_CLUSTER)

        faces = []
        for _, row in sample_df.iterrows():
            try:
                # (FIX) Truyền cả movie và frame vào load_image
                img = load_image(row["movie"], row["frame"])
                faces.append(crop_face(img, row["bbox"]))
            except Exception as e:
                print(f"Lỗi khi tải ảnh cho {row['movie']}/{row['frame']}: {e}")
                continue

        title = f"{label_prefix.capitalize()} ID: {gid} - ({len(group_df)} faces total)"
        out_path = os.path.join(OUT_DIR, f"{label_prefix}_{gid}.png")
        make_gallery(faces, title, out_path)
        print(f"Đã lưu bộ sưu tập cho cụm {gid} tại: {out_path}")


if __name__ == "__main__":
    df = load_df()
    if df is not None:
        # (FIX) Luôn visualize cột kết quả cuối cùng là 'final_character_id'
        visualize(df, "final_character_id", "character")
        print("\nHoàn tất! Hãy mở các file ảnh trong thư mục 'warehouse/debug/' để xem kết quả.")