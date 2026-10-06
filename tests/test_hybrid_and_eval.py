"""Hybrid retrieval must surface lexical-only hits; retrieval metrics must be correct."""

from types import SimpleNamespace

from src.rag.eval_retrieval import evaluate, reciprocal_rank
from src.rag.retriever import BM25, RAGRetriever, RetrievedChunk


def _chunk(cid, text, page):
    return RetrievedChunk(chunk_id=cid, text=text, page_number=page, section_title="",
                          section_type="", source_file="", company="BBCA", year=2023)


class FakeCollection:
    def __init__(self, chunks):
        self.chunks = chunks

    def count(self):
        return len(self.chunks)

    def get(self, include, where=None):
        return {"ids": [c.chunk_id for c in self.chunks],
                "documents": [c.text for c in self.chunks],
                "metadatas": [{"page_number": c.page_number, "company": c.company, "year": c.year}
                              for c in self.chunks]}


def _retriever(chunks, semantic):
    r = RAGRetriever.__new__(RAGRetriever)
    r.semantic_candidates, r.rrf_k, r._bm25_cache = 3, 60, {}
    r._indexer = SimpleNamespace(_get_collection=lambda: FakeCollection(chunks))
    r._semantic_search = lambda q, n, f: [_chunk(c.chunk_id, c.text, c.page_number) for c in semantic]
    return r


def test_lexical_only_hit_survives_fusion():
    exact = _chunk("c9", "Laba bersih BBCA tahun 2023 sebesar Rp48,6 triliun", 112)
    others = [_chunk(f"c{i}", f"paragraf umum tata kelola nomor {i}", i) for i in range(8)]
    retriever = _retriever(others + [exact], semantic=others[:3])  # embeddings missed c9
    ids = [c.chunk_id for c in retriever.retrieve("laba bersih BBCA 2023", top_k=5)]
    assert "c9" in ids


def test_tokenizer_strips_punctuation():
    assert BM25()._tokenize("BBCA, (Rp48,6 T).") == ["bbca", "rp48", "6", "t"]


def test_reciprocal_rank_and_recall():
    assert reciprocal_rank([False, True, False]) == 0.5
    golden = [{"question": "q1", "relevant_pages": [5]}, {"question": "q2", "relevant_pages": [9]}]
    hits = {"q1": [_chunk("a", "", 1), _chunk("b", "", 5)], "q2": [_chunk("c", "", 2)]}
    result = evaluate(lambda q, it: hits[q], golden, k=5)
    assert result["recall_at_k"] == 0.5
    assert result["mrr"] == 0.25
    assert result["misses"] == ["q2"]
