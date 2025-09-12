import json
import os
from typing import List, Dict


def search_actor(image_path: str) -> List[Dict]:
    """Return movie list and representative info for characters.

    This is a simple placeholder that loads the ``warehouse/characters.json``
    file and returns its contents. The input ``image_path`` is accepted for
    API compatibility but is not used in this stub implementation.
    """
    # Xác định đường dẫn tuyệt đối tới file characters.json
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_path = os.path.join(project_root, "warehouse", "characters.json")

    if not os.path.exists(abs_path):
        return []

    with open(abs_path, "r", encoding="utf-8") as f:
        characters = json.load(f)

    results = []
    for char_id, info in characters.items():
        results.append(
            {
                "character_id": char_id,
                "movies": info.get("movies", []),
                "rep_image": info.get("rep_image"),
            }
        )
    return results