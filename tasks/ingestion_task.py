import cv2 as cv
import os
import json
from glob import glob  # (CẢI TIẾN) Thư viện để tìm file theo mẫu
from prefect import task
from utils.config_loader import load_config


def extract_frames(video_path, root_output_folder):
    """Trích xuất khung hình và trả về thông tin metadata của video."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder_for_movie = os.path.join(root_output_folder, video_name)

    if not os.path.exists(output_folder_for_movie):
        os.makedirs(output_folder_for_movie)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video '{video_path}'")
        return None, None

    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps if fps > 0 else 0

    video_metadata = {
        "video_path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": round(duration_seconds, 2)
    }

    frame_interval = int(fps) if fps > 0 else 30
    frame_count = 0
    saved_frame_count = 0

    print(f"Bắt đầu trích xuất khung hình cho '{video_name}'...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(
                output_folder_for_movie, f"frame_{saved_frame_count:07d}.jpg"
            )
            cv.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_count += 1

    cap.release()
    print(
        f"-> Hoàn tất! Đã lưu {saved_frame_count} khung hình tại '{output_folder_for_movie}'"
    )
    return video_name, video_metadata


@task
def ingestion_task():
    """
    Hàm chính để điều khiển toàn bộ quá trình xử lý hàng loạt.
    """
    cfg = load_config()

    video_folder = cfg["storage"]["video_root"]
    frames_folder = cfg["storage"]["frames_root"]
    metadata_filepath = cfg["storage"]["metadata_json"]

    # --- (CẢI TIẾN) Tải metadata hiện có để kiểm tra ---
    try:
        with open(metadata_filepath, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
    except FileNotFoundError:
        all_metadata = {}

    # --- (CẢI TIẾN) Tìm tất cả các file video trong thư mục ---
    # Tìm tất cả các file có đuôi .mp4,
    video_paths = glob(os.path.join(video_folder, "*.[mM][pP]4"))

    if not video_paths:
        print(f"Không tìm thấy video nào trong thư mục '{video_folder}'")
        return

    print(f"Tìm thấy tổng cộng {len(video_paths)} video.")

    # --- (CẢI TIẾN) Xử lý hàng loạt ---
    new_videos_processed = False
    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder_for_movie = os.path.join(frames_folder, video_name)
        # Bỏ qua nếu video đã có trong metadata
        if os.path.isdir(output_folder_for_movie) and len(os.listdir(output_folder_for_movie)) > 0:
            print(f"Video '{video_name}' có vẻ đã được trích xuất frames trước đó. Bỏ qua.")
            if video_name not in all_metadata:
                print(f"Cảnh báo: Thiếu metadata cho phim '{video_name}'. Sẽ cố gắng tạo lại.")
                cap = cv.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / fps if fps > 0 else 0
                    all_metadata[video_name] = {
                        "video_path": video_path, "fps": fps,
                        "total_frames": total_frames, "duration_seconds": round(duration, 2)
                    }
                    new_videos_processed = True
                cap.release()
            continue

        new_videos_processed = True
        result = extract_frames(video_path, frames_folder)
        if result and result[0] is not None:
            name, info = result
            all_metadata[name] = info

    # --- (CẢI TIẾN) Chỉ ghi lại file metadata nếu có thay đổi ---
    if new_videos_processed:
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Đã cập nhật thành công file metadata tại '{metadata_filepath}'")
    else:
        print("\nKhông có video mới nào để xử lý.")


if __name__ == "__main__":
    ingestion_task()
