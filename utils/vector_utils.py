import numpy as np


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector using the L2 norm."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm