import json
import os
from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import tasks.embedding_task as emb_task


class DummyFaceAnalysis:
    """Simple stand-in for insightface.app.FaceAnalysis."""

    def __init__(self, embedding):
        self.embedding = np.asarray(embedding, dtype=np.float32)

    def prepare(self, *args, **kwargs):
        pass

    def get(self, img):
        face = SimpleNamespace(
            bbox=np.array([0, 0, 5, 5], dtype=np.float32),
            det_score=1.0,
            embedding=self.embedding,
        )
        return [face]


def build_index(chars_path):
    with open(chars_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    embeddings = []
    movie_map = []
    for info in data.values():
        embeddings.append(info["embedding"])
        movie_map.append(info["movies"])
    return np.asarray(embeddings, dtype=np.float32), movie_map


def search_actor(image_path, app, index, movie_map):
    img = cv2.imread(image_path)
    face = app.get(img)[0]
    query = face.embedding.astype(np.float32)
    sims = index @ query
    best_idx = int(np.argmax(sims))
    return movie_map[best_idx]


def test_embedding_and_search(tmp_path, monkeypatch):
    # --- Setup sample frame directory ---
    movie_dir = tmp_path / "frames" / "movie1"
    movie_dir.mkdir(parents=True)
    frame_path = movie_dir / "frame1.jpg"
    cv2.imwrite(str(frame_path), np.zeros((10, 10, 3), dtype=np.uint8))

    # --- Patch FaceAnalysis and config ---
    dummy_app = DummyFaceAnalysis(np.ones(512, dtype=np.float32))
    monkeypatch.setattr(emb_task, "FaceAnalysis", lambda *args, **kwargs: dummy_app)

    storage = {
        "frames_root": str(tmp_path / "frames"),
        "metadata_json": str(tmp_path / "metadata.json"),
        "embeddings_folder_per_movie": str(tmp_path / "embeddings"),
        "characters_json": str(tmp_path / "characters.json"),
        "face_crops_root": str(tmp_path / "face_crops"),
    }
    quality_filters = {
        "min_det_score": 0.0,
        "min_face_ratio": 0.0,
        "min_blur_clarity": 0.0,
        "brightness": {"enable": False},
        "contrast": {"enable": False},
    }
    cfg = {
        "embedding": {"model": "dummy", "providers": []},
        "storage": storage,
        "quality_filters": quality_filters,
        "search": {},
    }
    monkeypatch.setattr(emb_task, "load_config", lambda: cfg)

    # --- Run embedding task ---
    assert emb_task.embedding_task.fn() is True

    # --- Build characters_json using produced embedding ---
    emb_file = os.path.join(storage["embeddings_folder_per_movie"], "movie1.parquet")
    df = pd.read_parquet(emb_file)
    emb_vec = df["emb"].iloc[0].tolist()
    characters = {"0": {"embedding": emb_vec, "movies": ["movie1"]}}
    with open(storage["characters_json"], "w", encoding="utf-8") as f:
        json.dump(characters, f)

    # --- Build index and search ---
    index, movie_map = build_index(storage["characters_json"])
    result = search_actor(str(frame_path), dummy_app, index, movie_map)
    assert "movie1" in result
