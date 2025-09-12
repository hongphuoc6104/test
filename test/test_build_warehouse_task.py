import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import tasks.build_warehouse_task as bwt


def test_build_warehouse_closes_on_error(tmp_path, monkeypatch):
    table1 = pa.table({"col": [1]})
    table2 = pa.table({"col": [2]})
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    path1 = emb_dir / "m1.parquet"
    path2 = emb_dir / "m2.parquet"
    pq.write_table(table1, path1)
    pq.write_table(table2, path2)

    metadata = {
        "m1": {"embedding_file_path": str(path1)},
        "m2": {"embedding_file_path": str(path2)},
    }
    metadata_json = tmp_path / "metadata.json"
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    warehouse_path = tmp_path / "warehouse" / "embeddings.parquet"
    storage = {
        "metadata_json": str(metadata_json),
        "warehouse_embeddings": str(warehouse_path),
    }
    cfg = {"storage": storage}
    monkeypatch.setattr(bwt, "load_config", lambda: cfg)

    orig_writer = pq.ParquetWriter
    close_called = {"val": False}

    class FailingWriter:
        def __init__(self, path, schema):
            self.writer = orig_writer(path, schema)
            self.calls = 0

        def write_table(self, table):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("boom")
            self.writer.write_table(table)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            close_called["val"] = True
            self.writer.close()
            return False

    monkeypatch.setattr(bwt.pq, "ParquetWriter", FailingWriter)

    with pytest.raises(RuntimeError, match="boom"):
        bwt.build_warehouse_task.fn()

    assert close_called["val"] is True
