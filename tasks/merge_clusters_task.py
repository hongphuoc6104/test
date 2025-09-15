import os

import numpy as np
import pandas as pd
from prefect import task
from utils.config_loader import load_config
from utils.vector_utils import l2_normalize


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
    3. Build a global FAISS/Annoy index over centroids and connect
       clusters whose cosine similarity passes the configured threshold.
    4. Use Union-Find to merge ``cluster_id`` across all movies and
       assign a stable ``final_character_id``.
    5. Persist both the per-track assignments and the aggregated mapping.
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

    centroid_vectors = np.stack(centroids["centroid"].to_list()).astype("float32")
    centroid_vectors = np.array([l2_normalize(v) for v in centroid_vectors])

    n_centroids = len(centroid_vectors)
    if n_centroids == 0:
        print("[Merge] No centroids to merge. Skipping.")
        return clusters_path

    if n_centroids == 1:
        mapping = {centroids.loc[0, "cluster_id"]: 0}
        df["final_character_id"] = df["cluster_id"].map(mapping)
    else:
        knn = int(merge_cfg.get("knn", min(64, n_centroids)))
        knn = max(2, min(knn, n_centroids))

        neighbors: list[list[tuple[int, float]]] = []
        index_type = "brute-force"

        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(centroid_vectors.shape[1])
            index.add(centroid_vectors)
            sim_scores, sim_indices = index.search(centroid_vectors, knn)
            for sims, idxs in zip(sim_scores, sim_indices):
                neighbors.append(list(zip(idxs.tolist(), sims.tolist())))
            index_type = "FAISS"
        except Exception as exc:
            print(f"[Merge] Failed to build FAISS index ({exc}). Trying Annoy...")
            try:
                from annoy import AnnoyIndex  # type: ignore

                dim = centroid_vectors.shape[1]
                annoy_trees = int(merge_cfg.get("annoy_trees", 50))
                annoy_index = AnnoyIndex(dim, "angular")
                for idx, vec in enumerate(centroid_vectors):
                    annoy_index.add_item(idx, vec)
                annoy_index.build(annoy_trees)
                for vec in centroid_vectors:
                    idxs, dists = annoy_index.get_nns_by_vector(
                        vec, knn, include_distances=True
                    )
                    sims = [1 - 0.5 * (dist ** 2) for dist in dists]
                    neighbors.append(list(zip(idxs, sims)))
                index_type = "Annoy"
            except Exception as ann_exc:
                print(
                    "[Merge] Failed to build Annoy index "
                    f"({ann_exc}). Falling back to brute-force search."
                )
                neighbors = []

        if not neighbors:
            similarity = centroid_vectors @ centroid_vectors.T
            for sims in similarity:
                idxs = np.argsort(-sims)[:knn]
                neighbors.append(list(zip(idxs.tolist(), sims[idxs].tolist())))

        print(
            f"[Merge] Built {index_type} index for {n_centroids} centroids (k={knn})."
        )

        uf = UnionFind(centroids["cluster_id"].tolist())
        centroid_ids = centroids["cluster_id"].tolist()
        for src_idx, nbrs in enumerate(neighbors):
            src_id = centroid_ids[src_idx]
            for dst_idx, sim in nbrs:
                if dst_idx < 0 or dst_idx >= n_centroids:
                    continue
                if dst_idx == src_idx:
                    continue
                if sim < sim_threshold:
                    continue
                dst_id = centroid_ids[dst_idx]
                uf.union(src_id, dst_id)

        root_to_final: dict[str, int] = {}
        mapping = {}
        next_id = 0
        for cid in centroid_ids:
            root = uf.find(cid)
            if root not in root_to_final:
                root_to_final[root] = next_id
                next_id += 1
            mapping[cid] = root_to_final[root]

    centroids["final_character_id"] = centroids["cluster_id"].map(mapping)
    centroids["centroid"] = centroids["centroid"].apply(
        lambda v: v.tolist() if isinstance(v, np.ndarray) else v
    )

    df["final_character_id"] = df["cluster_id"].map(mapping)

    n_after = df["final_character_id"].nunique()
    print(f"[Merge] Clusters before merge: {n_before}, after merge: {n_after}")

    df.to_parquet(clusters_path, index=False)
    print(f"[Merge] Saved merged clusters to {clusters_path}")

    merged_output = storage_cfg.get("clusters_merged_parquet")
    if merged_output:
        os.makedirs(os.path.dirname(merged_output), exist_ok=True)
        centroids.to_parquet(merged_output, index=False)
        print(f"[Merge] Saved centroid mapping to {merged_output}")

    return clusters_path
