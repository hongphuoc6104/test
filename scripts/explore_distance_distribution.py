import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from utils.config_loader import load_config


def analyze_distances(use_pca: bool = False, sample_size: int = 1000, metric: str = "cosine") -> None:
    """Sample pairwise distances and print basic statistics.

    Parameters
    ----------
    use_pca: bool
        Whether to analyze PCA-reduced embeddings.
    sample_size: int
        Number of embeddings to sample for pairwise distance computation.
    metric: str
        Distance metric passed to ``pairwise_distances``.
    """
    cfg = load_config()
    storage_cfg = cfg["storage"]
    emb_path = storage_cfg["warehouse_embeddings_pca"] if use_pca else storage_cfg["warehouse_embeddings"]
    df = pd.read_parquet(emb_path)

    emb = np.stack(df["track_centroid"].values)
    if sample_size and len(emb) > sample_size:
        idx = np.random.choice(len(emb), size=sample_size, replace=False)
        emb = emb[idx]

    dists = pairwise_distances(emb, metric=metric)
    triu = np.triu_indices(len(emb), k=1)
    dists = dists[triu]

    label = "PCA" if use_pca else "original"
    print(f"Distance stats for {label} embeddings from {emb_path}")
    print(f"Samples: {len(emb)}, pairwise distances: {len(dists)}")
    for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
        print(f"  quantile {q:.2f}: {np.quantile(dists, q):.3f}")
    print(f"  mean: {dists.mean():.3f}, std: {dists.std():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Explore distance distribution of embeddings")
    parser.add_argument("--pca", action="store_true", help="Use PCA embeddings instead of original")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of embeddings to sample")
    parser.add_argument("--metric", type=str, default="cosine", help="Distance metric to use")
    args = parser.parse_args()
    analyze_distances(use_pca=args.pca, sample_size=args.sample_size, metric=args.metric)


if __name__ == "__main__":
    main()