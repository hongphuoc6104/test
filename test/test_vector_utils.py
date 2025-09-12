import pytest

np = pytest.importorskip("numpy")
from utils.vector_utils import _mean_vector, l2_normalize


def test_l2_normalize_zero_vector():
    v = np.zeros(3, dtype=np.float32)
    result = l2_normalize(v)
    assert np.array_equal(result, v)


def test_l2_normalize_non_zero_vector():
    v = np.array([3.0, 4.0], dtype=np.float32)
    result = l2_normalize(v)
    expected = v / np.linalg.norm(v)
    assert np.allclose(result, expected)


def test_mean_vector_single():
    vec = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    result = _mean_vector(vec)
    assert np.allclose(result, vec[0])


def test_mean_vector_multiple():
    vecs = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([3.0, 4.0, 5.0], dtype=np.float32),
    ]
    expected = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    result = _mean_vector(vecs)
    assert np.allclose(result, expected)