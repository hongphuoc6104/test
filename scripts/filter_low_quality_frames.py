import argparse
import os
from glob import glob

import cv2


def is_blurry(image_path: str, threshold: float) -> bool:
    image = cv2.imread(image_path)
    if image is None:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def filter_frames(frames_dir: str, threshold: float) -> int:
    removed = 0
    for img_path in glob(os.path.join(frames_dir, "*.jpg")):
        if is_blurry(img_path, threshold):
            os.remove(img_path)
            removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove blurry frames using Laplacian variance")
    parser.add_argument("frames_dir", help="Directory containing extracted frames")
    parser.add_argument("--threshold", type=float, default=100.0, help="Variance threshold")
    args = parser.parse_args()

    removed = filter_frames(args.frames_dir, args.threshold)
    print(f"Removed {removed} low quality frames from {args.frames_dir}")


if __name__ == "__main__":
    main()
