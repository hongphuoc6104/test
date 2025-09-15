import argparse
import os
from typing import Any, Dict, List

import cv2

from services.recognition import recognize
from utils.config_loader import load_config


def _display(title: str, paths: List[str]) -> None:
    """Show a list of images sequentially."""
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        cv2.imshow(title, img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize search results")
    parser.add_argument("--image", required=True, help="Query image path")
    parser.add_argument("--top-k", type=int, default=None, help="Number of candidates")
    args = parser.parse_args()

    cfg = load_config()
    frames_root = cfg["storage"].get("frames_root", "")

    result = recognize(args.image, top_k=args.top_k)
    candidates: List[Dict[str, Any]] = result.get("candidates", [])

    if result.get("is_unknown", True):
        print("Unknown face. Showing nearest candidates...")
        for cand in candidates:
            movies = ", ".join(cand.get("movies", []))
            print(f"Candidate {cand['character_id']} - Movies: {movies}")
            _display(str(cand["character_id"]), cand.get("preview_paths", []))
    else:
        print("Recognized face. Showing frames by movie...")
        for cand in candidates:
            print(f"Character {cand['character_id']}")
            images_by_movie: Dict[str, List[str]] = {}
            # representative image
            rep = cand.get("rep_image")
            if rep:
                movie = rep.get("movie")
                frame = rep.get("frame")
                if movie and frame:
                    path = os.path.join(frames_root, movie, frame)
                    images_by_movie.setdefault(movie, []).append(path)
            # group preview paths by movie using real path
            for p in cand.get("preview_paths", []):
                real = os.path.realpath(p)
                parts = real.split(os.sep)
                if len(parts) >= 2:
                    movie = parts[-2]
                    images_by_movie.setdefault(movie, []).append(real)
            for movie, imgs in images_by_movie.items():
                print(f"  Movie: {movie}")
                _display(f"{cand['character_id']}:{movie}", imgs)


if __name__ == "__main__":
    main()
