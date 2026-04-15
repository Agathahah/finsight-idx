"""
Tests for src/mcp/server.py and src/api/agent.py
==================================================
Uses mocks throughout — no real API calls, no real PDFs.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"


# ---------------------------------------------------------------------------
# MCP Server Tests
# ---------------------------------------------------------------------------

class TestMCPServerStructure:
    """Test MCP server tool/resource/prompt registration."""

    def test_server_imports_without_error(self):
        """MCP server module should import cleanly."""
        import src.mcp.server as server
        assert hasattr(server, "mcp")
        assert hasattr(server, "summarize_laporan")
        assert hasattr(server, "topic_modeling")
        assert hasattr(server, "tanya_laporan")
        assert hasattr(server, "hitung_rasio")

    def test_hitung_rasio_tool_callable(self):
        """hitung_rasio should be directly callable."""
        from src.mcp.server import hitung_rasio
        result_str = hitung_rasio(
            emiten="BBCA",
            harga_saham=9500,
            eps=485,
        )
        result = json.loads(result_str)
        assert result["emiten"] == "BBCA"
        assert "PER" in result["rasio"]

    def test_hitung_rasio_missing_data(self):
        """Should handle missing optional params gracefully."""
        from src.mcp.server import hitung_rasio
        result_str = hitung_rasio(emiten="BBCA")
        result = json.loads(result_str)
        assert "catatan" in result
        assert result["rasio"] == {}

    def test_topic_modeling_too_few_texts(self):
        """Should return error for fewer than 10 texts."""
        from src.mcp.server import topic_modeling
        result_str = topic_modeling(teks_list=["text1", "text2"])
        result = json.loads(result_str)
        assert "error" in result

    def test_analisis_fundamental_prompt(self):
        """Prompt template should return a non-empty string."""
        from src.mcp.server import analisis_fundamental
        result = analisis_fundamental(emiten="BBCA", tahun=2023)
        assert isinstance(result, str)
        assert "BBCA" in result
        assert "2023" in result
        assert "tanya_laporan" in result
        assert "hitung_rasio" in result

    def test_deteksi_risiko_prompt(self):
        """Risk detection prompt should include emiten and risk categories."""
        from src.mcp.server import deteksi_risiko
        result = deteksi_risiko(emiten="BBRI", tahun=2023, kategori_risiko="credit")
        assert "BBRI" in result
        assert "Kredit" in result or "kredit" in result

    def test_deteksi_risiko_all_categories(self):
        from src.mcp.server import deteksi_risiko
        result = deteksi_risiko(emiten="BMRI", tahun=2023)
        assert "BMRI" in result
        assert "Semua" in result or "semua" in result.lower()

    def test_tanya_laporan_no_index(self):
        """Should handle unindexed collection gracefully."""
        from src.mcp.server import tanya_laporan

        # Patch FinancialQAChain inside the module where it is imported
        with patch("src.rag.qa_chain.FinancialQAChain") as mock_qa_class:
            with patch("src.mcp.server.FinancialQAChain", mock_qa_class, create=True):
                mock_qa_class.return_value.ask.side_effect = ValueError("Collection empty")
                result_str = tanya_laporan("Berapa laba BCA?", emiten="BBCA")

        result = json.loads(result_str)
        assert "error" in result or "hint" in result or "pesan" in result


# ---------------------------------------------------------------------------
# Agent Orchestrator Tests
# ---------------------------------------------------------------------------

class TestFinSightOrchestrator:

    def _mock_summary_result(self):
        """Create a mock summarizer result."""
        mock = MagicMock()
        mock.general_overview = "BCA membukukan laba bersih Rp 48,6 triliun."
        mock.financial_highlights = "Laba naik 19,5% YoY."
        mock.risk_factors = "Risiko suku bunga dan kredit."
        mock.outlook_and_strategy = "Target kredit tumbuh 10-12%."
        mock.total_input_tokens = 500
        mock.total_output_tokens = 200
        mock.document_name = "bbca_2023"
        return mock

    def _mock_api_response(self, text: str) -> MagicMock:
        content = MagicMock()
        content.text = text
        usage = MagicMock()
        usage.input_tokens = 300
        usage.output_tokens = 150
        response = MagicMock()
        response.content = [content]
        response.usage = usage
        return response

    @patch("src.api.agent.get_client")
    @patch("src.api.agent.FinSightOrchestrator._step_summarize")
    def test_analyze_completes_with_mock(
        self, mock_summarize, mock_client
    ):
        """Full analyze() should complete without error."""
        from src.api.agent import FinSightOrchestrator, AnalysisReport

        # Mock step 1 return
        def fake_summarize(report, pdf_path):
            report.executive_summary = "BCA laba Rp 48,6 T."
            report.financial_highlights = "Laba naik 19,5%."
            report.risk_factors = "Risiko suku bunga."
            report.outlook = "Target tumbuh 10-12%."
            return report

        mock_summarize.side_effect = fake_summarize
        mock_client.return_value.messages.create.return_value = (
            self._mock_api_response("# Laporan BBCA 2023\n\nAnalisis lengkap...")
        )

        orchestrator = FinSightOrchestrator()
        report = orchestrator.analyze(
            emiten="BBCA",
            tahun=2023,
            pdf_path="dummy.pdf",
            news_texts=["berita " + str(i) for i in range(15)],
            skip_topic_modeling=True,  # skip BERTopic (not installed)
            skip_rag=True,
        )

        assert isinstance(report, AnalysisReport)
        assert report.emiten == "BBCA"
        assert report.tahun == 2023
        assert "summarize" in report.steps_completed
        assert report.markdown_report != ""

    @patch("src.api.agent.get_client")
    def test_sentiment_analysis_valid_json(self, mock_client):
        """analyze_sentiment_comparison should return SentimentComparison."""
        from src.api.agent import analyze_sentiment_comparison

        mock_client.return_value.messages.create.return_value = (
            self._mock_api_response(json.dumps({
                "sentimen": "Positif",
                "skor": 0.78,
                "tone_laporan": "Optimistis",
                "alignment": "Selaras",
                "catatan": "Sentimen berita positif sesuai kinerja keuangan.",
            }))
        )

        result = analyze_sentiment_comparison(
            news_texts=["BCA laba naik 20%."] * 5,
            report_summary="BCA membukukan kinerja excellent.",
            emiten="BBCA",
        )

        assert result.news_sentiment == "Positif"
        assert result.news_score == 0.78
        assert result.report_tone == "Optimistis"
        assert result.alignment == "Selaras"

    @patch("src.api.agent.get_client")
    def test_sentiment_handles_invalid_json(self, mock_client):
        """Should return default SentimentComparison on parse error."""
        from src.api.agent import analyze_sentiment_comparison

        mock_client.return_value.messages.create.return_value = (
            self._mock_api_response("Invalid response from Claude")
        )

        result = analyze_sentiment_comparison(
            news_texts=["text"] * 5,
            report_summary="summary",
            emiten="BBCA",
        )
        assert result.news_sentiment == "Netral"

    def test_sentiment_empty_news(self):
        """Empty news list should return neutral without API call."""
        from src.api.agent import analyze_sentiment_comparison

        result = analyze_sentiment_comparison(
            news_texts=[],
            report_summary="summary",
            emiten="BBCA",
        )
        assert result.alignment == "Tidak dapat dinilai — tidak ada berita"

    @patch("src.api.agent.get_client")
    def test_generate_markdown_report(self, mock_client):
        """Should return markdown string and token counts."""
        from src.api.agent import generate_markdown_report

        mock_client.return_value.messages.create.return_value = (
            self._mock_api_response(
                "# Laporan BBCA 2023\n\n## Kinerja Keuangan\nLaba naik 19,5%."
            )
        )

        md, in_tok, out_tok = generate_markdown_report(
            emiten="BBCA",
            tahun=2023,
            executive_summary="BCA laba Rp 48,6 T.",
            financial_highlights="Laba naik 19,5%.",
            risk_factors="Risiko suku bunga.",
            outlook="Target 10-12%.",
            topic_summary=None,
            sentiment=None,
        )

        assert "# Laporan BBCA" in md
        assert in_tok == 300
        assert out_tok == 150

    def test_save_report(self):
        """save() should write markdown to file."""
        from src.api.agent import AnalysisReport

        report = AnalysisReport(
            emiten="BBCA",
            tahun=2023,
            generated_at="2023-01-01T00:00:00",
            markdown_report="# Laporan BBCA\n\nTest content.",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = report.save(output_dir=tmpdir)
            assert path.exists()
            assert "BBCA" in path.name
            assert "# Laporan BBCA" in path.read_text()

    def test_fallback_report_contains_data(self):
        """Fallback report should include available data."""
        from src.api.agent import FinSightOrchestrator, AnalysisReport

        report = AnalysisReport(
            emiten="BBCA",
            tahun=2023,
            generated_at="2023-01-01",
            financial_highlights="Laba naik 19,5%.",
            risk_factors="Risiko suku bunga.",
        )

        orchestrator = FinSightOrchestrator()
        md = orchestrator._fallback_report(report)

        assert "BBCA" in md
        assert "2023" in md
        assert "Laba naik 19,5%" in md
        assert "Disclaimer" in md


# ---------------------------------------------------------------------------
# Integration: Tools + Agent
# ---------------------------------------------------------------------------

class TestToolAgentIntegration:

    def test_analyst_agent_uses_financial_tools(self):
        """FinancialAnalystAgent should use the same tool registry."""
        from src.api.tools import TOOL_REGISTRY, TOOL_DEFINITIONS

        # Verify all 3 tools are registered
        assert len(TOOL_REGISTRY) == 3
        assert len(TOOL_DEFINITIONS) == 3

        # Verify tool names match
        registered = set(TOOL_REGISTRY.keys())
        defined = {t["name"] for t in TOOL_DEFINITIONS}
        assert registered == defined

    def test_mcp_tools_match_implementation(self):
        """MCP server tools should wrap the same implementations."""
        from src.mcp.server import hitung_rasio
        from src.api.tools import hitung_rasio_keuangan

        # Both should compute the same PER
        mcp_result = json.loads(hitung_rasio(
            emiten="TEST", harga_saham=1000, eps=100
        ))
        direct_result = hitung_rasio_keuangan(
            emiten="TEST", harga_saham=1000, eps=100
        )

        assert mcp_result["rasio"]["PER"] == direct_result["rasio"]["PER"]
