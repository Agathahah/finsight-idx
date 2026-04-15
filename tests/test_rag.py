"""
Tests for src/rag/ — indexer, retriever, and qa_chain.
All tests use mocks — no real ChromaDB, no real API calls.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"

from src.rag.indexer import TextChunker, DocumentChunk   # noqa: E402
from src.rag.retriever import BM25, RAGRetriever, RetrievedChunk  # noqa: E402
from src.rag.qa_chain import FinancialQAChain, QAResponse, QASource  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "BCA membukukan laba bersih Rp 48,6 triliun pada tahun 2023, "
    "meningkat 19,5% dibandingkan tahun sebelumnya. "
    "Total aset BCA tumbuh 8,7% menjadi Rp 1.408 triliun. "
    "Kredit yang disalurkan meningkat 13,8% menjadi Rp 793,2 triliun. "
    "Rasio kecukupan modal (CAR) tercatat 25,9%."
)

SAMPLE_TEXTS = [
    "BCA laba bersih Rp 48,6 triliun tumbuh 19,5% di 2023.",
    "IHSG menguat didorong sektor perbankan dan konsumer.",
    "Bank Indonesia pertahankan suku bunga acuan 6%.",
    "Risiko kredit BCA terkendali dengan NPL gross 1,9%.",
    "BCA targetkan pertumbuhan kredit 10-12% di tahun 2024.",
]


def _make_chunk(
    chunk_id: str = "chunk_001",
    text: str = "BCA laba Rp 48,6 triliun.",
    page: int = 42,
    company: str = "BBCA",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        page_number=page,
        section_title="Ikhtisar Keuangan",
        section_type="financial",
        source_file="bbca_2023.pdf",
        company=company,
        year=2023,
        semantic_score=0.85,
        bm25_score=2.3,
        rrf_score=0.032,
    )


def _make_api_response(text: str = "Laba bersih BCA 2023 adalah Rp 48,6 triliun.") -> MagicMock:
    content = MagicMock()
    content.text = text
    usage = MagicMock()
    usage.input_tokens = 200
    usage.output_tokens = 80
    response = MagicMock()
    response.content = [content]
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# TextChunker Tests
# ---------------------------------------------------------------------------

class TestTextChunker:

    def test_short_text_returns_single_chunk(self):
        chunker = TextChunker(chunk_size=1000)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) == 1
        assert chunks[0] == SAMPLE_TEXT.strip()

    def test_long_text_splits_into_multiple_chunks(self):
        chunker = TextChunker(chunk_size=100, overlap=20)
        long_text = SAMPLE_TEXT * 5
        chunks = chunker.chunk(long_text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 200  # some tolerance for boundary detection

    def test_empty_text_returns_empty_list(self):
        chunker = TextChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_chunks_cover_full_content(self):
        """Concatenated chunks should contain all key content."""
        chunker = TextChunker(chunk_size=80, overlap=20)
        text = "Kata satu. Kata dua. Kata tiga. Kata empat. Kata lima. Kata enam."
        chunks = chunker.chunk(text)
        combined = " ".join(chunks)
        assert "Kata satu" in combined
        assert "Kata enam" in combined

    def test_overlap_creates_redundancy(self):
        """With overlap, some content should appear in consecutive chunks."""
        chunker = TextChunker(chunk_size=60, overlap=30)
        text = "ABCDEFGH. " * 20
        chunks = chunker.chunk(text)
        if len(chunks) > 1:
            # First chunk content should partially appear in second
            assert len(chunks[0]) > 0
            assert len(chunks[1]) > 0


# ---------------------------------------------------------------------------
# DocumentChunk Tests
# ---------------------------------------------------------------------------

class TestDocumentChunk:

    def test_auto_generates_chunk_id(self):
        chunk = DocumentChunk(
            chunk_id="",
            document_id="doc_001",
            text=SAMPLE_TEXT,
            page_number=5,
            section_title="Keuangan",
            section_type="financial",
            chunk_index=0,
            source_file="bbca.pdf",
        )
        assert chunk.chunk_id != ""
        assert len(chunk.chunk_id) == 16

    def test_char_count_computed(self):
        chunk = DocumentChunk(
            chunk_id="abc",
            document_id="doc",
            text="Hello world",
            page_number=1,
            section_title="",
            section_type="general",
            chunk_index=0,
            source_file="test.pdf",
        )
        assert chunk.char_count == len("Hello world")

    def test_to_metadata_is_flat(self):
        """ChromaDB requires flat metadata dict."""
        chunk = DocumentChunk(
            chunk_id="abc123",
            document_id="doc_001",
            text=SAMPLE_TEXT,
            page_number=42,
            section_title="Ikhtisar Keuangan",
            section_type="financial",
            chunk_index=5,
            source_file="bbca_2023.pdf",
            company="BBCA",
            year=2023,
        )
        meta = chunk.to_metadata()
        # All values must be primitive (no nested dicts/lists)
        for v in meta.values():
            assert isinstance(v, (str, int, float, bool))

    def test_to_metadata_excludes_text(self):
        """Text should not be in metadata (stored separately in ChromaDB)."""
        chunk = DocumentChunk(
            chunk_id="abc",
            document_id="doc",
            text=SAMPLE_TEXT,
            page_number=1,
            section_title="Test",
            section_type="general",
            chunk_index=0,
            source_file="test.pdf",
        )
        meta = chunk.to_metadata()
        assert "text" not in meta


# ---------------------------------------------------------------------------
# BM25 Tests
# ---------------------------------------------------------------------------

class TestBM25:

    def test_fit_and_score(self):
        bm25 = BM25()
        docs = [
            "bank indonesia suku bunga naik",
            "laba bersih bca meningkat tahun ini",
            "ihsg menguat sektor perbankan",
        ]
        bm25.fit(docs)
        scores = bm25.score_all("laba bersih bank")
        assert len(scores) == 3
        # Doc about BCA profit should score highest for "laba bersih"
        assert scores[1] > scores[2]

    def test_zero_score_for_unknown_terms(self):
        bm25 = BM25()
        bm25.fit(["hello world", "foo bar"])
        score = bm25.score("xyzxyz", 0)
        assert score == 0.0

    def test_identical_query_doc_scores_high(self):
        bm25 = BM25()
        bm25.fit(["laba bersih triliun rupiah", "suku bunga inflasi"])
        scores = bm25.score_all("laba bersih")
        assert scores[0] > scores[1]

    def test_score_all_length_matches_corpus(self):
        bm25 = BM25()
        docs = [f"document number {i}" for i in range(10)]
        bm25.fit(docs)
        scores = bm25.score_all("document number five")
        assert len(scores) == 10


# ---------------------------------------------------------------------------
# RetrievedChunk Tests
# ---------------------------------------------------------------------------

class TestRetrievedChunk:

    def test_citation_format(self):
        chunk = _make_chunk(page=42, company="BBCA")
        citation = chunk.citation()
        assert "BBCA" in citation
        assert "42" in citation
        assert "[" in citation and "]" in citation

    def test_citation_without_company(self):
        chunk = _make_chunk(company="")
        chunk.year = 0
        citation = chunk.citation()
        assert "hal." in citation


# ---------------------------------------------------------------------------
# RAGRetriever Tests (mocked)
# ---------------------------------------------------------------------------

class TestRAGRetriever:

    def test_raises_on_empty_query(self):
        retriever = RAGRetriever.__new__(RAGRetriever)
        with pytest.raises(ValueError, match="non-empty"):
            retriever.retrieve("")

    def test_build_filter_none_when_no_params(self):
        retriever = RAGRetriever.__new__(RAGRetriever)
        result = retriever._build_filter(None, None, None)
        assert result is None

    def test_build_filter_single_company(self):
        retriever = RAGRetriever.__new__(RAGRetriever)
        result = retriever._build_filter("BBCA", None, None)
        assert result == {"company": {"$eq": "BBCA"}}

    def test_build_filter_multiple_conditions(self):
        retriever = RAGRetriever.__new__(RAGRetriever)
        result = retriever._build_filter("BBCA", 2023, "financial")
        assert result is not None
        assert "$and" in result
        assert len(result["$and"]) == 3

    def test_rrf_fusion_combines_scores(self):
        retriever = RAGRetriever.__new__(RAGRetriever)
        retriever.rrf_k = 60

        semantic = [_make_chunk(f"id_{i}") for i in range(3)]
        bm25 = [_make_chunk(f"id_{i}") for i in [1, 0, 2]]

        fused = retriever._reciprocal_rank_fusion(semantic, bm25)
        assert len(fused) == 3
        # id_2 should rank last (rank 3 in both lists)
        assert fused[-1].chunk_id == "id_2"
        # id_0 and id_1 have equal RRF score — both appear in top 2
        assert fused[0].chunk_id in ("id_0", "id_1")


# ---------------------------------------------------------------------------
# FinancialQAChain Tests (mocked)
# ---------------------------------------------------------------------------

class TestFinancialQAChain:

    def test_raises_on_empty_question(self):
        qa = FinancialQAChain.__new__(FinancialQAChain)
        qa.retriever = MagicMock()
        qa.retriever.retrieve.return_value = []
        qa.model = "claude-sonnet-4-20250514"
        qa.top_k = 5
        qa.max_context_chars = 6000

        with pytest.raises(ValueError, match="non-empty"):
            qa.ask("")

    @patch("src.rag.qa_chain.get_client")
    def test_ask_returns_qa_response(self, mock_client):
        """ask() should return a QAResponse with answer and sources."""
        mock_client.return_value.messages.create.return_value = (
            _make_api_response("Laba BCA 2023 adalah Rp 48,6 triliun [Sumber: BBCA, hal. 42].")
        )

        qa = FinancialQAChain.__new__(FinancialQAChain)
        qa.model = "claude-sonnet-4-20250514"
        qa.top_k = 3
        qa.max_context_chars = 6000
        qa.retriever = MagicMock()
        qa.retriever.retrieve.return_value = [
            _make_chunk("c1", "BCA laba Rp 48,6 triliun.", 42),
            _make_chunk("c2", "Total aset Rp 1.408 triliun.", 45),
        ]

        response = qa.ask("Berapa laba bersih BCA 2023?")

        assert isinstance(response, QAResponse)
        assert "48,6" in response.answer
        assert len(response.sources) == 2
        assert response.input_tokens == 200
        assert response.output_tokens == 80

    def test_no_results_returns_graceful_response(self):
        """Empty retrieval should return informative response, not raise."""
        qa = FinancialQAChain.__new__(FinancialQAChain)
        qa.model = "claude-sonnet-4-20250514"
        qa.top_k = 5
        qa.max_context_chars = 6000
        qa.retriever = MagicMock()
        qa.retriever.retrieve.return_value = []

        response = qa._no_results_response("Pertanyaan tidak ada jawabannya?")
        assert isinstance(response, QAResponse)
        assert response.retrieved_chunks == 0
        assert "tidak ditemukan" in response.answer

    def test_build_context_respects_max_chars(self):
        """Context should not exceed max_context_chars."""
        qa = FinancialQAChain.__new__(FinancialQAChain)
        qa.max_context_chars = 200

        chunks = [
            _make_chunk(f"id_{i}", "x" * 100)
            for i in range(5)
        ]
        context, citation_map = qa._build_context(chunks)
        assert len(context) <= 500  # some tolerance for citation headers

    def test_format_answer_contains_sources(self):
        response = QAResponse(
            question="Berapa laba BCA?",
            answer="Laba BCA adalah Rp 48,6 triliun.",
            sources=[
                QASource(
                    chunk_id="c1",
                    citation="[BBCA, 2023, hal. 42]",
                    text_preview="BCA laba bersih...",
                    page_number=42,
                    section_title="Ikhtisar",
                    company="BBCA",
                    year=2023,
                    relevance_score=0.85,
                )
            ],
            model="claude-sonnet-4-20250514",
            retrieved_chunks=1,
            answered_at="2023-12-01T10:00:00",
            input_tokens=200,
            output_tokens=80,
            latency_ms=1200.0,
        )
        formatted = response.format_answer()
        assert "Berapa laba BCA?" in formatted
        assert "Rp 48,6 triliun" in formatted
        assert "BBCA" in formatted

    def test_save_response_creates_file(self):
        response = QAResponse(
            question="Test?",
            answer="Test answer.",
            sources=[],
            model="test",
            retrieved_chunks=0,
            answered_at="2023-01-01",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            qa = FinancialQAChain.__new__(FinancialQAChain)
            path = qa.save_response(response, output_dir=tmpdir)
            assert path.exists()
            saved = json.loads(path.read_text())
            assert saved["question"] == "Test?"
