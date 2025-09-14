import os
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from utils.config_loader import load_config
from tasks.merge_clusters_task import merge_clusters_task
from tasks.character_task import character_task


def _cluster_new_embeddings(new_df: pd.DataFrame, existing: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Cluster only the new embeddings and assign unique cluster IDs.

    Parameters
    ----------
    new_df : pd.DataFrame
        Newly added embeddings with ``track_centroid`` and movie column.
    existing : pd.DataFrame
        Existing cluster assignments to ensure ID uniqueness.
    cfg : dict
        Loaded configuration for clustering parameters.
    """
    clustering_cfg = cfg.get("clustering", {})
    group_col = "movie_id" if "movie_id" in new_df.columns else "movie"
    results = []

    for movie_key, group in new_df.groupby(group_col):
        emb_matrix = np.stack(group["track_centroid"].to_list()).astype("float32")
        if len(group) > 1:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=float(clustering_cfg.get("distance_threshold", 0.7)),
                metric="cosine",
                linkage="complete",
            )
            labels = clusterer.fit_predict(emb_matrix)
        else:
            labels = np.array([0])

        existing_ids = existing[existing[group_col] == movie_key]["cluster_id"]
        if not existing_ids.empty:
            max_id = max(int(cid.split("_")[1]) for cid in existing_ids)
            offset = max_id + 1
        else:
            offset = 0

        group = group.copy()
        group["cluster_id"] = [f"{movie_key}_{offset + lbl}" for lbl in labels]
        results.append(group)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def update_pipeline(new_parquet: str) -> None:
    """Incrementally update the warehouse with a new parquet file.

    The function appends the new data to the embeddings store, clusters only the
    newly added rows, merges clusters across movies and rebuilds the search index.
    """
    cfg = load_config()
    storage = cfg["storage"]
    emb_path = storage["warehouse_embeddings"]
    clusters_path = storage["warehouse_clusters"]

    new_df = pd.read_parquet(new_parquet)

    # Append embeddings
    if os.path.exists(emb_path):
        base_emb = pd.read_parquet(emb_path)
        emb_df = pd.concat([base_emb, new_df], ignore_index=True)
    else:
        emb_df = new_df.copy()
    emb_df.to_parquet(emb_path, index=False)

    # Load existing clusters and cluster only new data
    existing_clusters = pd.read_parquet(clusters_path) if os.path.exists(clusters_path) else pd.DataFrame()
    new_clusters = _cluster_new_embeddings(new_df, existing_clusters, cfg)
    updated = (
        pd.concat([existing_clusters, new_clusters], ignore_index=True)
        if not existing_clusters.empty else new_clusters
    )
    updated.to_parquet(clusters_path, index=False)

    # Merge clusters across movies and rebuild index
    merge_clusters_task()
    character_task()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Incrementally update pipeline with new data")
    parser.add_argument("parquet", help="Path to newly generated embeddings parquet")
    args = parser.parse_args()
    update_pipeline(args.parquet)
