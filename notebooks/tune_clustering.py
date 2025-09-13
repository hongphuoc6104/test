import os
from itertools import product

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from utils.config_loader import load_config

# --- Configurable hyper-parameter grids ---
DISTANCE_THRESHOLDS = [0.5, 0.7, 0.9]
METRICS = ["cosine", "euclidean"]
LINKAGES = ["single", "complete", "average"]

OUT_CSV = "warehouse/debug/clustering_tuning.csv"


def load_embeddings() -> np.ndarray:
    """Load saved embeddings from configured path."""
    cfg = load_config()
    storage = cfg["storage"]
    emb_path = storage.get("warehouse_embeddings_pca")
    if not os.path.exists(emb_path):
        emb_path = storage["warehouse_embeddings"]
    print(f"[INFO] Loading embeddings from {emb_path}")
    df = pd.read_parquet(emb_path)
    return np.array(df["emb"].tolist(), dtype=np.float32)


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    X = load_embeddings()
    results = []

    for dt, metric, linkage in product(DISTANCE_THRESHOLDS, METRICS, LINKAGES):
        if linkage == "ward" and metric != "euclidean":
            continue
        try:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=dt,
                metric=metric,
                linkage=linkage,
            )
            labels = clusterer.fit_predict(X)
            n_clusters = len(set(labels))
            sil = None
            if n_clusters > 1:
                try:
                    sil = silhouette_score(X, labels, metric=metric)
                except Exception:
                    sil = None
            print(
                f"{metric}-{linkage} dt={dt}: clusters={n_clusters}, silhouette={sil}"
            )
            results.append(
                {
                    "distance_threshold": dt,
                    "metric": metric,
                    "linkage": linkage,
                    "num_clusters": n_clusters,
                    "silhouette": sil,
                }
            )
        except Exception as e:
            print(f"[WARN] Failed {metric}-{linkage} dt={dt}: {e}")
            results.append(
                {
                    "distance_threshold": dt,
                    "metric": metric,
                    "linkage": linkage,
                    "num_clusters": None,
                    "silhouette": None,
                    "error": str(e),
                }
            )

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"Results saved to {OUT_CSV}")


if __name__ == "__main__":
    main()