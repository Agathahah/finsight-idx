"""The Space bundle contains exactly what the app needs and refuses a bad index."""

import importlib.util
from pathlib import Path

import pytest

from src.rag.indexer import DEFAULT_EMBED_MODEL
from src.rag.manifest import IndexVersionError, record_document

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_space", ROOT / "scripts" / "build_space.py")
build_space = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_space)


def _index(tmp_path, embed=DEFAULT_EMBED_MODEL):
    store = tmp_path / "vectorstore"
    store.mkdir()
    (store / "chroma.sqlite3").write_bytes(b"x")
    (store / ".DS_Store").write_bytes(b"x")
    record_document(store, "d", {"company": "BBCA", "year": 2023, "total_chunks": 10953,
                                 "total_pages": 747, "pages_ocr": [570]},
                    embed_model=embed, chunk_size=256, chunk_overlap=64)
    return store


def test_bundle_contents(tmp_path):
    out = tmp_path / "space"
    build_space.build(out, _index(tmp_path))
    build_space.summarize(out)
    files = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert {"app.py", "README.md", "requirements.txt", "data/vectorstore/chroma.sqlite3",
            "data/vectorstore/index_manifest.json", "src/rag/retriever.py"} <= files
    assert not any("__pycache__" in f or f.endswith(".DS_Store") for f in files)
    readme = (out / "README.md").read_text()
    assert readme.startswith("---\ntitle: FinSight IDX") and "sdk_version: 6.12.0" in readme
    assert "BBCA 2023 (10,953 chunks, 747 pages, OCR on 1 pages)" in readme


def test_refuses_index_with_other_embedding_model(tmp_path):
    with pytest.raises(IndexVersionError):
        build_space.build(tmp_path / "space", _index(tmp_path, embed="other-model"))


def test_refuses_index_without_manifest(tmp_path):
    store = tmp_path / "vs"
    store.mkdir()
    with pytest.raises(IndexVersionError):
        build_space.build(tmp_path / "space", store)
