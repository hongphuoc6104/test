import json
from types import SimpleNamespace
import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

import utils.search_actor as sa


class DummyFaceAnalysis:
    def __init__(self, embedding):
        self.embedding = embedding

    def prepare(self, *args, **kwargs):
        pass

    def get(self, img):
        face = SimpleNamespace(det_score=1.0, embedding=self.embedding)
        return [face]


def _setup_env(tmp_path, monkeypatch):
    emb = np.array([1.0, 0.0], dtype=np.float32)
    dummy_app = DummyFaceAnalysis(emb)
    monkeypatch.setattr(sa, "FaceAnalysis", lambda *args, **kwargs: dummy_app)
    cfg = {
        "embedding": {"model": "dummy", "providers": [], "l2_normalize": False},
        "storage": {"characters_json": str(tmp_path / "characters.json")},
    }
    monkeypatch.setattr(sa, "load_config", lambda: cfg)
    monkeypatch.setattr(sa, "load_index", lambda: (object(), {0: 0}))
    monkeypatch.setattr(sa, "_query_index", lambda idx, e, k: (np.array([0.1]), np.array([0])))

    with open(tmp_path / "characters.json", "w", encoding="utf-8") as f:
        json.dump({"0": {"movies": ["movie1"]}}, f)

    img_path = tmp_path / "img.jpg"
    cv2.imwrite(str(img_path), np.zeros((5, 5, 3), dtype=np.uint8))
    return img_path, emb


def test_search_actor_return_emb(monkeypatch, tmp_path):
    img_path, emb = _setup_env(tmp_path, monkeypatch)
    results = sa.search_actor(str(img_path), k=1, return_emb=True)
    assert "embedding" in results
    assert callable(results["search_func"])
    matches = results["search_func"](np.asarray([results["embedding"]], dtype=np.float32))
    assert matches[0]["character_id"] == "0"
    assert matches[0]["movies"] == ["movie1"]


def test_search_actor_direct(monkeypatch, tmp_path):
    img_path, emb = _setup_env(tmp_path, monkeypatch)
    matches = sa.search_actor(str(img_path), k=1)
    assert matches[0]["character_id"] == "0"
    assert matches[0]["movies"] == ["movie1"]