import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
import tasks.cluster_task as ct


def test_cluster_task_groups_by_movie(monkeypatch):
    df = pd.DataFrame(
        {
            "movie": ["m1", "m1", "m2"],
            "track_id": [1, 2, 3],
            "track_centroid": [np.zeros(3), np.ones(3), np.zeros(3)],
        }
    )

    monkeypatch.setattr(ct.pd, "read_parquet", lambda path: df)

    cfg = {
        "storage": {
            "warehouse_embeddings": "in.parquet",
            "warehouse_clusters": "out.parquet",
        },
        "clustering": {},
        "pca": {},
    }
    monkeypatch.setattr(ct, "load_config", lambda: cfg)

    saved = {}

    def fake_to_parquet(self, path, index=False):
        saved["df"] = self.copy()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    class DummyClusterer:
        def __init__(self, *args, **kwargs):
            pass

        def fit_predict(self, X):
            return np.arange(len(X))

    monkeypatch.setattr(ct, "AgglomerativeClustering", DummyClusterer)

    ct.cluster_task.fn()

    result = saved["df"]
    assert set(result["cluster_id"]) == {"m1_0", "m1_1", "m2_0"}
