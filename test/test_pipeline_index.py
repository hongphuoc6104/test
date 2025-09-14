import os
import json
import sys
import types
from pathlib import Path

def test_pipeline_creates_index(tmp_path, monkeypatch):
    # Ensure repository root on path
    root = Path(__file__).resolve().parents[1]
    sys.path.append(str(root))

    # Stub external dependencies
    prefect_stub = types.ModuleType("prefect")
    def _decorator(fn=None, **kwargs):
        if fn is None:
            def wrap(f):
                return f
            return wrap
        return fn
    prefect_stub.flow = _decorator
    prefect_stub.task = _decorator
    monkeypatch.setitem(sys.modules, "prefect", prefect_stub)

    monkeypatch.setitem(sys.modules, "yaml", types.ModuleType("yaml"))
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    numpy_mod = types.ModuleType("numpy")
    numpy_mod.ndarray = object
    monkeypatch.setitem(sys.modules, "numpy", numpy_mod)
    monkeypatch.setitem(sys.modules, "pandas", types.ModuleType("pandas"))
    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda x, *a, **k: x
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_mod)
    sklearn_stub = types.ModuleType("sklearn")
    cluster_mod = types.ModuleType("sklearn.cluster")
    class DummyCluster:
        pass
    cluster_mod.AgglomerativeClustering = DummyCluster
    decomp_mod = types.ModuleType("sklearn.decomposition")
    class DummyPCA:
        pass
    decomp_mod.PCA = DummyPCA
    monkeypatch.setitem(sys.modules, "sklearn", sklearn_stub)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", cluster_mod)
    monkeypatch.setitem(sys.modules, "sklearn.decomposition", decomp_mod)
    monkeypatch.setitem(sys.modules, "pyarrow", types.ModuleType("pyarrow"))
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", types.ModuleType("pyarrow.parquet"))
    monkeypatch.setitem(sys.modules, "joblib", types.ModuleType("joblib"))
    insight_stub = types.ModuleType("insightface")
    app_mod = types.ModuleType("insightface.app")
    class DummyFaceAnalysis:
        def prepare(self, *a, **k):
            pass
    app_mod.FaceAnalysis = DummyFaceAnalysis
    monkeypatch.setitem(sys.modules, "insightface", insight_stub)
    monkeypatch.setitem(sys.modules, "insightface.app", app_mod)

    import flows.pipeline as pipeline
    import utils.indexer as indexer

    storage = {
        "characters_json": str(tmp_path / "characters.json"),
        "index_path": str(tmp_path / "index.faiss"),
        "index_map": str(tmp_path / "index_map.json"),
    }
    cfg = {"storage": storage, "pca": {"enable": False}}
    monkeypatch.setattr(pipeline, "load_config", lambda: cfg)

    dummy = lambda: None
    monkeypatch.setattr(pipeline, "ingestion_task", dummy)
    monkeypatch.setattr(pipeline, "embedding_task", dummy)
    monkeypatch.setattr(pipeline, "build_warehouse_task", dummy)
    monkeypatch.setattr(pipeline, "validate_warehouse_task", dummy)
    monkeypatch.setattr(pipeline, "pca_task", dummy)
    monkeypatch.setattr(pipeline, "cluster_task", dummy)
    monkeypatch.setattr(pipeline, "merge_clusters_task", dummy)
    monkeypatch.setattr(pipeline, "preview_clusters_task", dummy)

    def fake_build_index(chars_path, index_path):
        map_path = storage["index_map"]
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, "wb") as f:
            f.write(b"index")
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump({"0": 0}, f)
    monkeypatch.setattr(indexer, "build_index", fake_build_index)

    def fake_character_task():
        os.makedirs(os.path.dirname(storage["characters_json"]), exist_ok=True)
        with open(storage["characters_json"], "w", encoding="utf-8") as f:
            json.dump({"0": {"embedding": [0], "movies": []}}, f)
        indexer.build_index(storage["characters_json"], storage["index_path"])
        return storage["characters_json"]
    monkeypatch.setattr(pipeline, "character_task", fake_character_task)

    pipeline.main_pipeline()

    assert os.path.exists(storage["index_path"])
    assert os.path.exists(storage["index_map"])
