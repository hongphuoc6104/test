import argparse
import joblib
import numpy as np
import os

from utils.search_actor import search_actor
from utils.config_loader import load_config


def main():
    # --- Load config for defaults ---
    cfg = load_config()
    pca_cfg = cfg.get("pca", {})
    storage_cfg = cfg.get("storage", {})
    search_cfg = cfg.get("search", {})

    default_sim_threshold = search_cfg.get("sim_threshold", 0.5)
    default_margin_threshold = search_cfg.get("margin_threshold", 0.05)
    default_top_k = max(2, search_cfg.get("knn", 5))
    pca_model_path = storage_cfg.get("pca_model", "models/pca_model.joblib")

    parser = argparse.ArgumentParser(description="Search actor by face image")
    parser.add_argument("--image", required=True, help="Path to the query image")
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=default_sim_threshold,
        help="Minimum similarity score for a valid match (default from config)",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=default_margin_threshold,
        help="Minimum margin between top matches (default from config)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=default_top_k,
        help="Number of top matches to retrieve (default from config)",
    )
    args = parser.parse_args()

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
    top_k = max(2, args.top_k)
    matches = results["search_func"](emb, top_k=top_k)  # search_func được trả về từ search_actor

    if not matches:
        print("No matching actors found.")
        return

    sims = [m.get("distance", 0.0) for m in matches]
    sim_top1 = sims[0]
    sim_top2 = sims[1] if len(sims) > 1 else float("-inf")
    valid_matches = [
        m for m in matches if m.get("distance", 0.0) >= args.sim_threshold
    ]

    if sim_top1 < args.sim_threshold:
        print("Unknown")
        return

    if (sim_top1 - sim_top2) < args.margin_threshold:
        if len(valid_matches) <= 1:
            print("Unknown")
            return
        matches_to_print = valid_matches
    else:
        matches_to_print = [valid_matches[0]]

    # --- In kết quả ---
    for res in matches_to_print:
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
