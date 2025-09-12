import os
import time
import hashlib
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from prefect import task
from insightface.app import FaceAnalysis
from utils.config_loader import load_config
from utils.image_utils import calculate_blur_score, check_brightness, check_contrast


# --- Các hàm tiện ích ---
def make_global_id(movie: str, frame: str, bbox: np.ndarray) -> str:
    s = f"{movie}|{frame}|{int(bbox[0])}|{int(bbox[1])}|{int(bbox[2])}|{int(bbox[3])}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# --- Hàm xử lý từng phim ---
def process_single_movie(movie_name, movie_frames_path, app, config):
    image_files = sorted([
        os.path.join(movie_frames_path, f)
        for f in os.listdir(movie_frames_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    movie_specific_rows = []
    q_filters = config.get("quality_filters", {})

    for img_path in tqdm(image_files, desc=f"Scanning {movie_name}"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue

            # --- Resize trước khi detect ---
            scale = 1.0
            processing_img = img
            h, w, _ = processing_img.shape
            if h > config['pre_resize_dim'] or w > config['pre_resize_dim']:
                scale = config['pre_resize_dim'] / max(h, w)
                processing_img = cv2.resize(img, (int(w * scale), int(h * scale)))

            faces = app.get(processing_img)
            if not faces:
                continue

            frame_area = processing_img.shape[0] * processing_img.shape[1]
            min_face_area = config['min_face_ratio'] * frame_area

            good_quality_faces = []
            for face in faces:
                # --- Filter tầng 1 ---
                if face.det_score < config['min_det_score']:
                    continue
                x1, y1, x2, y2 = face.bbox
                if (x2 - x1) * (y2 - y1) < min_face_area:
                    continue

                # --- Crop khuôn mặt gốc ---
                orig_x1, orig_y1, orig_x2, orig_y2 = np.round(face.bbox / scale).astype(int)
                face_crop = img[orig_y1:orig_y2, orig_x1:orig_x2]
                if face_crop.size == 0:
                    continue

                # --- Filter tầng 2: brightness + contrast ---
                gray_face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                if q_filters.get("brightness", {}).get("enable") and not check_brightness(
                    gray_face_crop, q_filters["brightness"]["value_range"]
                ):
                    continue
                if q_filters.get("contrast", {}).get("enable") and not check_contrast(
                    gray_face_crop, q_filters["contrast"]["min_value"]
                ):
                    continue

                # --- Filter tầng 3: blur ---
                blur_score = calculate_blur_score(face_crop)
                if blur_score < config['min_blur_clarity']:
                    continue

                good_quality_faces.append(face)

            good_quality_faces.sort(key=lambda f: f.det_score, reverse=True)
            selected_faces = good_quality_faces[:config['max_faces_per_frame']]

            for face in selected_faces:
                if face.embedding is None:
                    continue

                original_bbox = np.round(face.bbox / scale).astype(np.int32)
                emb = face.embedding
                if config['embedding'].get('l2_normalize', True):
                    emb = l2_normalize(emb)

                row = {
                    "global_id": make_global_id(movie_name, os.path.basename(img_path), original_bbox),
                    "movie": movie_name,
                    "frame": os.path.basename(img_path),
                    "bbox": original_bbox.tolist(),
                    "det_score": float(face.det_score),
                    "emb": emb.tolist(),
                    "ts_created": int(time.time()),
                    "version": 1
                }
                movie_specific_rows.append(row)

        except Exception as e:
            print(f"\n[Error] failed to process {img_path}: {e}")
            continue

    return movie_specific_rows


# --- Task chính ---
@task(name="Embedding Task")
def embedding_task():
    cfg = load_config()

    config = {
        "embedding": cfg["embedding"],
        "storage": cfg["storage"],
        "quality_filters": cfg.get("quality_filters", {}),
        "min_det_score": cfg["quality_filters"].get("min_det_score", 0.5),
        "min_face_ratio": cfg["quality_filters"].get("min_face_ratio", 0.005),
        "min_blur_clarity": cfg["quality_filters"].get("min_blur_clarity", 70.0),
        "max_faces_per_frame": cfg.get("search", {}).get("max_faces_per_frame", 5),  # default
        "pre_resize_dim": cfg.get("pre_resize_dim", 1280)
    }
    storage_cfg = config['storage']

    print("Initializing InsightFace model...")
    app = FaceAnalysis(name=config['embedding']['model'], providers=config['embedding']['providers'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model ready.")

    metadata_filepath = storage_cfg['metadata_json']
    try:
        with open(metadata_filepath, 'r', encoding='utf-8') as f:
            all_metadata = json.load(f)
    except FileNotFoundError:
        all_metadata = {}

    frames_root = storage_cfg['frames_root']
    embeddings_folder = storage_cfg['embeddings_folder_per_movie']
    os.makedirs(embeddings_folder, exist_ok=True)

    movie_folders = [d for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d))]
    new_data_generated = False

    for movie_name in movie_folders:
        expected_parquet_path = os.path.join(embeddings_folder, f"{movie_name}.parquet")
        if os.path.exists(expected_parquet_path):
            print(f"File embedding cho phim '{movie_name}' đã tồn tại. Bỏ qua.")
            continue

        new_data_generated = True
        print(f"\nProcessing movie: {movie_name}")
        movie_frames_path = os.path.join(frames_root, movie_name)

        movie_rows = process_single_movie(movie_name, movie_frames_path, app, config)

        if movie_name not in all_metadata:
            all_metadata[movie_name] = {}

        if not movie_rows:
            print(f"⚠️ Không tìm thấy khuôn mặt nào đủ điều kiện cho phim '{movie_name}'.")
            all_metadata[movie_name]['num_faces_detected'] = 0
            all_metadata[movie_name]['embedding_file_path'] = None
            continue

        df_movie = pd.DataFrame(movie_rows)
        df_movie.to_parquet(expected_parquet_path, index=False)
        print(f"✅ Đã lưu {len(movie_rows)} embeddings cho phim '{movie_name}' tại: {expected_parquet_path}")

        all_metadata[movie_name]['num_faces_detected'] = len(movie_rows)
        all_metadata[movie_name]['embedding_file_path'] = expected_parquet_path

    if new_data_generated:
        with open(metadata_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Cập nhật thành công file {metadata_filepath}")
    else:
        print("\nKhông có phim mới nào để xử lý.")

    return True


if __name__ == '__main__':
    embedding_task()
