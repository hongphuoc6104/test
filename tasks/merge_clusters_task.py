import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from prefect import task
from utils.config_loader import load_config


@task(name="Merge Clusters Task")
def merge_clusters_task(sim_threshold: float = 0.80):
    """Merge clusters across movies based on centroid similarity.

    Steps:
    1. Load clustered track data.
    2. Compute centroid for each cluster (per movie).
    3. Build a similarity graph where edges connect clusters with
       cosine similarity above ``sim_threshold``.
    4. Assign ``final_character_id`` using connected components.
    5. Save the merged results back to the warehouse.
    """
    print("\n--- Starting Merge Clusters Task ---")
    cfg = load_config()
    storage_cfg = cfg["storage"]
    clusters_path = storage_cfg["warehouse_clusters"]

    df = pd.read_parquet(clusters_path)
    if df.empty:
        print("[Merge] No cluster data found. Skipping merge.")
        return clusters_path

    # Compute centroid for each cluster
    centroids = (
        df.groupby("cluster_id")["track_centroid"]
        .apply(lambda arrs: np.mean(np.stack(arrs.to_list()), axis=0))
        .reset_index()
        .rename(columns={"track_centroid": "centroid"})
    )

    centroid_matrix = np.stack(centroids["centroid"].to_list())
    cluster_ids = centroids["cluster_id"].tolist()

    # Cosine similarity between all centroids
    sim_matrix = cosine_similarity(centroid_matrix)

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(cluster_ids)
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            if sim_matrix[i, j] >= sim_threshold:
                G.add_edge(cluster_ids[i], cluster_ids[j])

    components = list(nx.connected_components(G))
    mapping = {}
    for comp_id, nodes in enumerate(components):
        for node in nodes:
            mapping[node] = comp_id

    df["final_character_id"] = df["cluster_id"].map(mapping)

    df.to_parquet(clusters_path, index=False)
    print(f"[Merge] Saved merged clusters to {clusters_path}")
    return clusters_path
