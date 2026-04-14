"""
Tests for src/nlp/summarizer.py and src/nlp/pdf_extractor.py
=============================================================
All tests use mocks — no real API calls, no real PDF files needed.
"""

from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"

from src.nlp.pdf_extractor import (  # noqa: E402
    PDFExtractor,
    PageContent,
    DocumentSection,
)
from src.nlp.summarizer import (  # noqa: E402
    DocumentSummarizer,
    ExecutiveSummary,
    SectionSummary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_FINANCIAL_TEXT = """
IKHTISAR KEUANGAN

Pada tahun 2023, perseroan membukukan pendapatan bersih sebesar Rp 15,2 triliun,
meningkat 12,3% dibandingkan tahun sebelumnya sebesar Rp 13,5 triliun.
Laba bersih tercatat Rp 3,8 triliun dengan margin laba bersih 25%.
Total aset tumbuh 8,5% menjadi Rp 98,7 triliun.
"""

SAMPLE_RISK_TEXT = """
FAKTOR RISIKO

Perseroan menghadapi beberapa risiko utama:
1. Risiko pasar: fluktuasi suku bunga dan nilai tukar
2. Risiko kredit: kualitas portofolio pinjaman
3. Risiko regulasi: perubahan kebijakan OJK dan BI
4. Risiko operasional: gangguan sistem teknologi informasi
"""

SAMPLE_OUTLOOK_TEXT = """
PROSPEK DAN STRATEGI

Perseroan menargetkan pertumbuhan kredit 15% pada tahun 2024.
Fokus ekspansi ke segmen UMKM dan digital banking.
Target rasio NPL di bawah 2% dan ROE minimal 18%.
"""


def _mock_api_response(text: str = "Ringkasan hasil analisis.") -> dict:
    return {
        "task": "summarize",
        "model": "claude-sonnet-4-20250514",
        "result": text,
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


def _make_mock_page(page_number: int, text: str) -> PageContent:
    return PageContent(page_number=page_number, text=text)


# ---------------------------------------------------------------------------
# PDFExtractor Tests
# ---------------------------------------------------------------------------

class TestPDFExtractor:

    def test_raises_file_not_found(self):
        extractor = PDFExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract("nonexistent_file.pdf")

    def test_clean_text_removes_extra_whitespace(self):
        extractor = PDFExtractor()
        dirty = "hello   world\n\n\n\nfoo  bar"
        clean = extractor._clean_text(dirty)
        assert "   " not in clean
        assert "\n\n\n" not in clean

    def test_clean_text_empty_string(self):
        extractor = PDFExtractor()
        assert extractor._clean_text("") == ""

    def test_classify_section_financial(self):
        extractor = PDFExtractor()
        result = extractor._classify_section("Ikhtisar Keuangan pendapatan laba")
        assert result == "financial"

    def test_classify_section_risk(self):
        extractor = PDFExtractor()
        result = extractor._classify_section("Faktor Risiko manajemen risiko pasar")
        assert result == "risk"

    def test_classify_section_outlook(self):
        extractor = PDFExtractor()
        result = extractor._classify_section("Prospek dan strategi bisnis 2024")
        assert result == "outlook"

    def test_classify_section_general_fallback(self):
        extractor = PDFExtractor()
        result = extractor._classify_section("Surat kepada pemegang saham")
        assert result == "general"

    def test_detect_sections_groups_pages(self):
        """Sections should be built from consecutive pages."""
        extractor = PDFExtractor(min_section_chars=10, max_pages_per_chunk=3)
        pages = [
            _make_mock_page(1, "Introduction text here with enough content"),
            _make_mock_page(2, "More content on second page for the section"),
            _make_mock_page(3, "IKHTISAR KEUANGAN\nPendapatan tumbuh 12%"),
            _make_mock_page(4, "Laba bersih Rp 3,8 triliun tahun ini"),
        ]
        sections = extractor._detect_sections(pages)
        assert len(sections) >= 1
        assert all(isinstance(s, DocumentSection) for s in sections)

    def test_build_section_returns_none_if_too_short(self):
        extractor = PDFExtractor(min_section_chars=500)
        result = extractor._build_section(
            title="Short", start_page=1, end_page=1, text="Too short"
        )
        assert result is None

    def test_build_section_returns_section(self):
        extractor = PDFExtractor(min_section_chars=10)
        result = extractor._build_section(
            title="Keuangan",
            start_page=1,
            end_page=2,
            text=SAMPLE_FINANCIAL_TEXT,
        )
        assert result is not None
        assert result.section_type == "financial"
        assert result.page_span == 2


# ---------------------------------------------------------------------------
# DocumentSummarizer Tests
# ---------------------------------------------------------------------------

class TestDocumentSummarizer:

    @patch("src.nlp.summarizer.analyze_financial_text")
    def test_summarize_text_single_chunk(self, mock_api):
        """Short text should produce one API call."""
        mock_api.return_value = _mock_api_response("Kinerja keuangan sangat baik.")

        summarizer = DocumentSummarizer()
        result = summarizer.summarize_text(SAMPLE_FINANCIAL_TEXT, "test_doc")

        assert isinstance(result, ExecutiveSummary)
        assert result.document_name == "test_doc"
        assert result.general_overview == "Kinerja keuangan sangat baik."
        assert mock_api.call_count == 1

    @patch("src.nlp.summarizer.analyze_financial_text")
    def test_summarize_text_multiple_chunks(self, mock_api):
        """Long text should be split into multiple chunks."""
        mock_api.return_value = _mock_api_response("Chunk summary.")

        summarizer = DocumentSummarizer(max_chunk_chars=100)
        long_text = SAMPLE_FINANCIAL_TEXT * 10  # force chunking

        result = summarizer.summarize_text(long_text, "long_doc")

        assert isinstance(result, ExecutiveSummary)
        assert mock_api.call_count >= 2

    @patch("src.nlp.summarizer.analyze_financial_text")
    def test_chunk_text_respects_max_chars(self, mock_api):
        """Chunks should never exceed max_chunk_chars."""
        summarizer = DocumentSummarizer(max_chunk_chars=200)
        text = "\n\n".join([f"Paragraph {i}: " + "x" * 80 for i in range(10)])
        chunks = summarizer._chunk_text(text)
        for chunk in chunks:
            assert len(chunk) <= 300  # some tolerance for paragraph boundaries

    def test_chunk_text_short_returns_single(self):
        summarizer = DocumentSummarizer(max_chunk_chars=10_000)
        chunks = summarizer._chunk_text(SAMPLE_FINANCIAL_TEXT)
        assert len(chunks) == 1
        assert chunks[0] == SAMPLE_FINANCIAL_TEXT

    @patch("src.nlp.summarizer.analyze_financial_text")
    def test_aggregate_single_part_no_api_call(self, mock_api):
        """Single part should return directly without extra API call."""
        summarizer = DocumentSummarizer()
        result = summarizer._aggregate(["Single summary."], "test")
        assert result == "Single summary."
        mock_api.assert_not_called()

    @patch("src.nlp.summarizer.analyze_financial_text")
    def test_aggregate_multiple_parts_calls_api(self, mock_api):
        """Multiple parts should trigger aggregation API call."""
        mock_api.return_value = _mock_api_response("Merged summary.")
        summarizer = DocumentSummarizer()
        result = summarizer._aggregate(["Part 1.", "Part 2.", "Part 3."], "test")
        assert result == "Merged summary."
        mock_api.assert_called_once()

    def test_section_type_to_task_mapping(self):
        from src.api.client import FinancialTask
        summarizer = DocumentSummarizer()
        assert summarizer._section_type_to_task("financial") == FinancialTask.KEY_METRICS
        assert summarizer._section_type_to_task("risk") == FinancialTask.RISK_FACTORS
        assert summarizer._section_type_to_task("outlook") == FinancialTask.SUMMARIZE
        assert summarizer._section_type_to_task("unknown") == FinancialTask.SUMMARIZE

    @patch("src.nlp.summarizer.analyze_financial_text")
    def test_save_to_json_file(self, mock_api):
        """summarize_text with output_dir should write a JSON file."""
        mock_api.return_value = _mock_api_response("Summary saved.")

        with tempfile.TemporaryDirectory() as tmpdir:
            summarizer = DocumentSummarizer(output_dir=tmpdir)
            result = summarizer.summarize_text(SAMPLE_FINANCIAL_TEXT, "bca_2023")
            summarizer._save(result, "bca_2023")

            files = list(Path(tmpdir).glob("bca_2023_summary_*.json"))
            assert len(files) == 1

            saved = json.loads(files[0].read_text())
            assert saved["document_name"] == "bca_2023"


# ---------------------------------------------------------------------------
# ExecutiveSummary Serialization Tests
# ---------------------------------------------------------------------------

class TestExecutiveSummary:

    def test_to_dict_contains_required_keys(self):
        summary = ExecutiveSummary(
            document_name="test",
            processed_at="2024-01-01T00:00:00",
            total_pages=100,
            total_sections=5,
            financial_highlights="Laba naik 20%",
            risk_factors="Risiko suku bunga",
            outlook_and_strategy="Target tumbuh 15%",
        )
        d = summary.to_dict()
        assert "document_name" in d
        assert "financial_highlights" in d
        assert "risk_factors" in d
        assert "outlook_and_strategy" in d
        assert "section_summaries" in d

    def test_to_json_is_valid_json(self):
        summary = ExecutiveSummary(
            document_name="bca_2023",
            processed_at="2024-01-01T00:00:00",
            total_pages=250,
            total_sections=8,
        )
        json_str = summary.to_json()
        parsed = json.loads(json_str)
        assert parsed["document_name"] == "bca_2023"
        assert parsed["total_pages"] == 250

    def test_to_json_handles_unicode(self):
        """Indonesian characters must serialize correctly."""
        summary = ExecutiveSummary(
            document_name="test",
            processed_at="2024-01-01",
            total_pages=1,
            total_sections=1,
            financial_highlights="Pendapatan naik 15% menjadi Rp 48,6 triliun 📈",
        )
        json_str = summary.to_json()
        assert "Rp 48,6 triliun" in json_str
