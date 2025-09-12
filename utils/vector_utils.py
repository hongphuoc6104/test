import numpy as np


def _mean_vector(vectors: list[np.ndarray]) -> np.ndarray:
    """Return the mean vector from a list of vectors."""
    return np.mean(np.stack(vectors), axis=0)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector using the L2 norm."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

