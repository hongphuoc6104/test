# utils/image_utils.py
import cv2
import numpy as np
from typing import Sequence

def calculate_blur_score(image: np.ndarray) -> float:
    """Tính điểm sắc nét bằng variance of Laplacian. Càng cao càng nét."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def check_brightness(gray_image: np.ndarray, value_range: Sequence[float]) -> bool:
    """Kiểm tra độ sáng trung bình có nằm trong khoảng cho phép."""
    if gray_image is None or gray_image.size == 0:
        return False
    mean_val = float(gray_image.mean())
    return value_range[0] <= mean_val <= value_range[1]

def check_contrast(gray_image: np.ndarray, min_value: float) -> bool:
    """Kiểm tra độ tương phản có đủ lớn (dựa trên độ lệch chuẩn)."""
    if gray_image is None or gray_image.size == 0:
        return False
    std_val = float(gray_image.std())
    return std_val >= float(min_value)