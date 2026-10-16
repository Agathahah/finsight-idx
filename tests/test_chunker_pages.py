"""Regression tests for the chunker bug found in the October 2026 audit."""

from src.nlp.pdf_extractor import PAGE_BREAK
from src.rag.indexer import TextChunker


def _text():
    return "".join(f"Kredit bermasalah Kategori {i} dan nilainya relatif besar di atas Rp{i} Miliar. "
                   for i in range(40))


def test_no_runs_of_near_duplicate_chunks():
    chunks = TextChunker(chunk_size=256, overlap=64).chunk(_text())
    assert len(chunks) == len(set(chunks))
    assert min(len(c) for c in chunks[:-1]) >= 128


def test_each_step_advances_at_least_half_window_minus_overlap():
    offsets = [o for _, o in TextChunker(256, 64).chunk_with_offsets(_text())]
    assert all(b - a >= 256 // 2 - 64 for a, b in zip(offsets, offsets[1:]))


def _covered(text, chunker):
    cov = [False] * len(text)
    for piece, off in chunker.chunk_with_offsets(text):
        i = text.find(piece[:20], off)
        for k in range(i, i + len(piece)):
            cov[k] = True
    return [text[k] for k in range(len(text)) if not cov[k] and not text[k].isspace()]


def test_no_text_is_skipped():
    """v2 forced a 192-character step even when the cut came earlier, skipping text."""
    prose = "".join(f"Kalimat {i} pendek. " if i % 3 else f"Kalimat {i} yang jauh lebih panjang dari biasanya. "
                    for i in range(120))
    table = "\n".join(f"Baris {i} 1.{i:03d}.107 1.314.{i:03d} 1.228.345" for i in range(60))
    for text in (prose, table, prose + PAGE_BREAK + table):
        assert _covered(text, TextChunker(256, 64)) == []


def test_table_rows_are_not_split():
    rows = [f"Pos keuangan {i} {i}.408.107 {i}.314.732 1.228.345 1.075.570" for i in range(40)]
    for chunk in TextChunker(256, 64).chunk("\n".join(rows)):
        assert all(line in rows for line in chunk.split("\n"))


def test_offsets_map_to_pages():
    pages = ["Halaman satu. " * 30, "Halaman dua. " * 30, "Halaman tiga. " * 30]
    text = PAGE_BREAK.join(pages)
    chunker = TextChunker(256, 64)
    found = {text.count(PAGE_BREAK, 0, off) for _, off in chunker.chunk_with_offsets(text)}
    assert found == {0, 1, 2}


def test_page_break_not_leaked_into_chunk_text():
    text = PAGE_BREAK.join(["a. " * 200, "b. " * 200])
    assert all(PAGE_BREAK not in c for c in TextChunker(256, 64).chunk(text))


def test_force_reindex_deletes_existing_chunks(tmp_path, monkeypatch):
    """v1 raised NameError because `existing` was only defined when force_reindex=False."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.rag import indexer as idx

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    collection = MagicMock()
    collection.get.return_value = {"ids": ["old"]}
    rag = idx.RAGIndexer.__new__(idx.RAGIndexer)
    rag.collection_name, rag.embed_model_name = "c", "m"
    rag.chunker = TextChunker(256, 64)
    rag._get_collection = lambda: collection
    rag._store_chunks = lambda chunks: None
    rag.persist_dir = tmp_path
    fake_doc = SimpleNamespace(sections=[], total_pages=1)
    rag.extractor = SimpleNamespace(extract=lambda p: fake_doc)
    rag.index_pdf(pdf, company="BBCA", year=2023, force_reindex=True)
    collection.delete.assert_called_once()
