import argparse
import os
import sys

# Bảo đảm root dự án nằm trong sys.path để có thể import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.search_actor import search_actor


def main():
    parser = argparse.ArgumentParser(description="Search actor by face image")
    parser.add_argument("--image", required=True, help="Path to the query image")
    args = parser.parse_args()

    results = search_actor(args.image)

    if not results:
        print("No matching actors found.")
        return

    for res in results:
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