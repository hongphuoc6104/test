import argparse
import os
from typing import Any, Dict, List

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore

try:
    from utils.search_actor import search_actor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    search_actor = None  # type: ignore


def _decide_matches(
    matches: List[Dict[str, Any]],
    sim_threshold: float,
    ratio_threshold: float,
    margin_threshold: float,
) -> Dict[str, Any]:
    if not matches:
        return {"recognized": False, "matches": []}

    sims = [m.get("distance", 0.0) for m in matches]
    sim_top1 = sims[0]
    sim_top2 = sims[1] if len(sims) > 1 else float("-inf")

    if sim_top1 < sim_threshold:
        return {"recognized": False, "matches": matches}

    eps = 1e-8
    if sim_top1 / max(sim_top2, eps) < ratio_threshold:
        return {"recognized": False, "matches": matches}

    if (sim_top1 - sim_top2) < margin_threshold:
        return {"recognized": False, "matches": matches}

    return {"recognized": True, "matches": [matches[0]]}


def run(
    image_path: str,
    sim_threshold: float,
    ratio_threshold: float,
    margin_threshold: float,
    top_k: int,
    min_count: int,
) -> Dict[str, Any]:
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}

    if cv2 is None:
        return {"error": "OpenCV not installed"}
    if np is None:
        return {"error": "NumPy not installed"}
    if search_actor is None:
        return {"error": "Search utilities not available"}

    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Cannot read image: {image_path}"}

    try:
        result = search_actor(
            image_path, k=max(2, top_k), min_count=min_count, return_emb=True
        )
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Search failed: {e}"}

    if not result or "embedding" not in result:
        return {"error": "No face detected in the image."}

    emb = np.array(result["embedding"], dtype=np.float32).reshape(1, -1)
    search_func = result["search_func"]

    try:
        matches = search_func(emb, top_k=max(2, top_k), min_count=min_count)
    except Exception as e:
        return {"error": f"Search failed: {e}"}

    return _decide_matches(matches, sim_threshold, ratio_threshold, margin_threshold)


def main() -> None:
    from utils.config_loader import load_config

    cfg = load_config()
    search_cfg = cfg.get("search", {})
    storage_cfg = cfg.get("storage", {})

    parser = argparse.ArgumentParser(description="Find actor by face image")
    parser.add_argument("--image", required=True, help="Path to the query image")
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=search_cfg.get("sim_threshold", 0.5),
        help="Minimum similarity score for a valid match",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=search_cfg.get("margin_threshold", 0.05),
        help="Minimum margin between top matches",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=search_cfg.get("ratio_threshold", 1.1),
        help="Minimum ratio between top-1 and top-2 similarities",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=max(2, search_cfg.get("knn", 5)),
        help="Number of top matches to retrieve",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=search_cfg.get("min_count", 0),
        help="Minimum occurrence count for characters",
    )

    args = parser.parse_args()
    res = run(
        args.image,
        args.sim_threshold,
        args.ratio_threshold,
        args.margin_threshold,
        args.top_k,
        args.min_count,
    )

    if "error" in res:
        print(f"Error: {res['error']}")
        return

    matches = res.get("matches", [])
    if res.get("recognized") and matches:
        m = matches[0]
        movies = ", ".join(m.get("movies", []))
        print(f"Character {m.get('character_id', 'unknown')} - Movies: {movies}")
        rep = m.get("rep_image", {})
        if rep:
            movie = rep.get("movie", "")
            frame = rep.get("frame", "")
            bbox = rep.get("bbox", [])
            frames_root = storage_cfg.get("frames_root", "")
            path = (
                os.path.join(frames_root, movie, frame)
                if movie and frame
                else ""
            )
            print(f"  Representative frame: {path} bbox={bbox}")
    else:
        print("Unknown")
        if matches:
            print("Top suggestions:")
            for m in matches:
                movies = ", ".join(m.get("movies", []))
                score = m.get("distance", 0.0)
                print(
                    f"  Character {m.get('character_id', 'unknown')} "
                    f"- score: {score:.4f} - Movies: {movies}"
                )
                for p in m.get("preview_paths", [])[:3]:
                    print(f"    Preview: {p}")


if __name__ == "__main__":
    main()
