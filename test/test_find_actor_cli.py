import types
import pytest

import cli.find_actor as fa


class DummyArray(list):
    def reshape(self, a, b):
        return self


def test_run_recognized(monkeypatch, tmp_path):
    stub_cv2 = types.SimpleNamespace(imread=lambda path: object())
    monkeypatch.setattr(fa, "cv2", stub_cv2)

    dummy_np = types.SimpleNamespace(array=lambda x, dtype=None: DummyArray(x), float32=float)
    monkeypatch.setattr(fa, "np", dummy_np)

    img_path = tmp_path / "img.jpg"
    img_path.write_text("dummy")

    def dummy_search_actor(image_path, k, min_count, return_emb):
        assert k == 20
        assert min_count == 0
        assert return_emb is True

        def _search_func(emb, top_k, min_count):
            assert top_k == 20
            assert min_count == 0
            return [
                {"character_id": "1", "movies": ["m1"], "distance": 0.9},
                {"character_id": "2", "movies": ["m2"], "distance": 0.8},
            ]
        return {"embedding": [0.0, 0.0], "search_func": _search_func}

    monkeypatch.setattr(fa, "search_actor", dummy_search_actor)
    res = fa.run(str(img_path), 0.5, 1.1, 0.05)
    assert res["recognized"] is True
    assert len(res["matches"]) == 2
    assert [m["character_id"] for m in res["matches"]] == ["1", "2"]


def test_run_unknown(monkeypatch, tmp_path):
    stub_cv2 = types.SimpleNamespace(imread=lambda path: object())
    monkeypatch.setattr(fa, "cv2", stub_cv2)

    dummy_np = types.SimpleNamespace(
        array=lambda x, dtype=None: DummyArray(x), float32=float
    )
    monkeypatch.setattr(fa, "np", dummy_np)

    img_path = tmp_path / "img.jpg"
    img_path.write_text("dummy")

    def dummy_search_actor(image_path, k, min_count, return_emb):
        assert k == 20
        assert min_count == 0
        assert return_emb is True

        def _search_func(emb, top_k, min_count):
            assert top_k == 20
            assert min_count == 0
            return [
                {"character_id": "1", "movies": [], "distance": 0.4},
                {"character_id": "2", "movies": [], "distance": 0.39},
            ]
        return {"embedding": [0.0, 0.0], "search_func": _search_func}

    monkeypatch.setattr(fa, "search_actor", dummy_search_actor)
    res = fa.run(str(img_path), 0.5, 1.1, 0.05)
    assert res["recognized"] is False
    assert len(res["matches"]) == 2


def test_run_missing_image(monkeypatch):
    stub_cv2 = types.SimpleNamespace(imread=lambda path: None)
    monkeypatch.setattr(fa, "cv2", stub_cv2)
    res = fa.run("missing.jpg", 0.5, 1.1, 0.05)
    assert "error" in res
