from __future__ import annotations

import json
import pandas as pd
from prefect import task


@task(name="Filter Clusters Task")
def filter_clusters_task(
    clusters: pd.DataFrame,
    characters_path: str,
    min_size: int = 3,
    min_det: float = 0.6,
    min_frames: int = 5,
):
    """Remove low-quality clusters and update ``characters.json``.

    Parameters
    ----------
    clusters:
        DataFrame of tracklets with at least ``final_character_id``, ``det_score``
        and ``frame`` columns.
    characters_path:
        Path to the characters JSON file to be filtered in-place.

    Returns
    -------
    str
        The path to the cleaned ``characters.json`` file.
    """

    stats = (
        clusters.groupby("final_character_id")
        .agg(
            size=("final_character_id", "size"),
            mean_det=("det_score", "mean"),
            frames=("frame", "nunique"),
        )
        .reset_index()
    )

    valid_ids = stats[
        (stats["size"] >= min_size)
        & (stats["mean_det"] >= min_det)
        & (stats["frames"] >= min_frames)
    ]["final_character_id"].astype(str).tolist()

    with open(characters_path, "r", encoding="utf-8") as f:
        characters = json.load(f)

    filtered = {cid: characters[cid] for cid in valid_ids if cid in characters}

    with open(characters_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    return characters_path
