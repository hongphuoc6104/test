import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from prefect import task
from utils.config_loader import load_config


class UnionFind:
    """Simple union-find/DSU structure for merging cluster IDs."""

    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}

    def find(self, x: str) -> str:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> dict[str, list[str]]:
        from collections import defaultdict

        groups: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            groups[self.find(item)].append(item)
        return groups


@task(name="Merge Clusters Task")
def merge_clusters_task(sim_threshold: float | None = None):
    """Merge clusters across movies based on centroid similarity.

    Steps:
    1. Load clustered track data.
    2. Compute centroid for each cluster together with its ``movie_id``.
    3. Compute pairwise cosine distance between centroids and cluster them
       using Agglomerative clustering.
    4. Use Union-Find to merge ``cluster_id`` across all movies.
    5. Save the merged results back to the warehouse and log stats.
    """

    print("\n--- Starting Merge Clusters Task ---")
    cfg = load_config()
    storage_cfg = cfg["storage"]
    merge_cfg = cfg.get("merge", {})
    if sim_threshold is None:
        sim_threshold = float(merge_cfg.get("global_threshold", 0.8))
    clusters_path = storage_cfg["warehouse_clusters"]

    df = pd.read_parquet(clusters_path)
    if df.empty:
        print("[Merge] No cluster data found. Skipping merge.")
        return clusters_path

    n_before = df["cluster_id"].nunique()

    # Ensure movie_id column exists for downstream logging
    if "movie_id" not in df.columns:
        df["movie_id"] = df.get("movie", 0)

    # Compute centroid for each cluster along with movie_id
    centroids = (
        df.groupby("cluster_id")
        .agg(
            centroid=("track_centroid", lambda arrs: np.mean(np.stack(arrs.to_list()), axis=0)),
            movie_id=("movie_id", "first"),
        )
        .reset_index()
    )

    centroid_matrix = np.stack(centroids["centroid"].to_list())
    dist_matrix = 1 - cosine_similarity(centroid_matrix)

    if len(centroids) > 1:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=1 - float(sim_threshold),
        )
        labels = clusterer.fit_predict(dist_matrix)
    else:
        labels = np.zeros(len(centroids), dtype=int)

    centroids["merge_label"] = labels

    # Union-Find to merge cluster IDs with same label
    uf = UnionFind(centroids["cluster_id"].tolist())
    for _, group in centroids.groupby("merge_label"):
        ids = group["cluster_id"].tolist()
        root = ids[0]
        for cid in ids[1:]:
            uf.union(root, cid)

    groups = uf.groups()
    mapping: dict[str, int] = {}
    for new_id, ids in enumerate(groups.values()):
        for cid in ids:
            mapping[cid] = new_id

    df["final_character_id"] = df["cluster_id"].map(mapping)

    n_after = df["final_character_id"].nunique()
    print(f"[Merge] Clusters before merge: {n_before}, after merge: {n_after}")

    df.to_parquet(clusters_path, index=False)
    print(f"[Merge] Saved merged clusters to {clusters_path}")
    return clusters_path
