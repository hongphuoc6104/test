import argparse
import joblib
import numpy as np
import os

from utils.search_actor import search_actor
from utils.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description="Search actor by face image")
    parser.add_argument("--image", required=True, help="Path to the query image")
    args = parser.parse_args()

    # --- Load config ---
    cfg = load_config()
    pca_cfg = cfg.get("pca", {})
    storage_cfg = cfg.get("storage", {})
    pca_model_path = storage_cfg.get("pca_model", "models/pca_model.joblib")

    # --- Get embedding and search function ---
    results = search_actor(args.image, return_emb=True)
    if not results or "embedding" not in results:
        print("No matching actors found.")
        return

    emb = np.array(results["embedding"]).reshape(1, -1)

    # --- Nếu PCA được bật thì transform ---
    if pca_cfg.get("enable", False) and os.path.exists(pca_model_path):
        print(f"[INFO] Applying PCA transform from {pca_model_path}")
        pca_model = joblib.load(pca_model_path)
        emb = pca_model.transform(emb)

    # --- Search bằng embedding đã được PCA ---
    matches = results["search_func"](emb)  # search_func được trả về từ search_actor

    if not matches:
        print("No matching actors found.")
        return

    # --- In kết quả ---
    for res in matches:
        char_id = res.get("character_id", "unknown")
        movies = ", ".join(res.get("movies", []))
        rep = res.get("rep_image", {})
        print(f"Character {char_id} - Movies: {movies}")
        if rep:
            movie = rep.get("movie", "")
            frame = rep.get("frame", "")
            bbox = rep.get("bbox", [])
            print(f"  Representative frame: {movie}/{frame} bbox={bbox}")


if __name__ == "__main__":
    main()
