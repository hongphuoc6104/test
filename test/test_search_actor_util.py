import json
import sys
from types import SimpleNamespace, ModuleType
import pytest

np = pytest.importorskip("numpy")

# create dummy cv2 module
cv2 = ModuleType("cv2")
cv2.imwrite = lambda path, img: True
cv2.imread = lambda path: np.zeros((5, 5, 3), dtype=np.uint8)
sys.modules["cv2"] = cv2

import utils.search_actor as sa


class DummyFaceAnalysis:
    def __init__(self, embedding):
        self.embedding = embedding

    def prepare(self, *args, **kwargs):
        pass

    def get(self, img):
        face = SimpleNamespace(det_score=1.0, embedding=self.embedding)
        return [face]


def _setup_env(tmp_path, monkeypatch, rep_image=None):
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

    data = {"0": {"movies": ["movie1"]}}
    if rep_image is not None:
        data["0"]["rep_image"] = rep_image
    with open(tmp_path / "characters.json", "w", encoding="utf-8") as f:
        json.dump(data, f)

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


def test_search_actor_returns_rep_image(monkeypatch, tmp_path):
    rep = {"movie": "movie1", "frame": "frame1.jpg", "bbox": [1, 2, 3, 4]}
    img_path, emb = _setup_env(tmp_path, monkeypatch, rep_image=rep)
    matches = sa.search_actor(str(img_path), k=1)
    assert matches[0]["rep_image"] == rep
