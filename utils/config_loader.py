import yaml
import os

def load_config(cfg_path: str = "configs/config.yaml"):
    """
    Tải file cấu hình YAML một cách an toàn từ gốc dự án.
    """
    # Xác định project root dựa trên vị trí utils/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    absolute_cfg_path = os.path.join(project_root, cfg_path)

    if not os.path.exists(absolute_cfg_path):
        raise FileNotFoundError(f"Không tìm thấy file config tại: {absolute_cfg_path}")

    with open(absolute_cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)