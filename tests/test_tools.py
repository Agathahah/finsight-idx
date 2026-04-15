"""
Tests for src/api/tools.py
============================
Tests tool implementations, executor, and agent loop.
All Claude API calls are mocked.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"

from src.api.tools import (  # noqa: E402
    hitung_rasio_keuangan,
    bandingkan_emiten,
    cari_di_laporan,
    execute_tool,
    FinancialAnalystAgent,
    AnalystResponse,
    TOOL_DEFINITIONS,
    TOOL_REGISTRY,
)


# ---------------------------------------------------------------------------
# hitung_rasio_keuangan Tests
# ---------------------------------------------------------------------------

class TestHitungRasioKeuangan:

    def test_per_calculation(self):
        result = hitung_rasio_keuangan(
            emiten="BBCA",
            harga_saham=9500,
            eps=485,
        )
        assert result["rasio"]["PER"] == round(9500 / 485, 2)
        assert any("PER" in s for s in result["interpretasi"])

    def test_per_undervalued(self):
        result = hitung_rasio_keuangan(
            emiten="TEST", harga_saham=1000, eps=200
        )
        assert result["rasio"]["PER"] == 5.0
        assert any("undervalued" in s for s in result["interpretasi"])

    def test_per_overvalued(self):
        result = hitung_rasio_keuangan(
            emiten="TEST", harga_saham=10000, eps=200
        )
        assert result["rasio"]["PER"] == 50.0
        assert any("premium" in s.lower() or "overvalued" in s for s in result["interpretasi"])

    def test_pbv_calculation(self):
        result = hitung_rasio_keuangan(
            emiten="BBCA",
            harga_saham=9500,
            book_value_per_share=2500,
        )
        assert result["rasio"]["PBV"] == round(9500 / 2500, 2)

    def test_roe_calculation(self):
        result = hitung_rasio_keuangan(
            emiten="BBCA",
            laba_bersih=48600,
            ekuitas=210000,
        )
        expected_roe = round((48600 / 210000) * 100, 2)
        assert result["rasio"]["ROE"] == f"{expected_roe}%"

    def test_roe_excellent(self):
        result = hitung_rasio_keuangan(
            emiten="BBCA", laba_bersih=200, ekuitas=800
        )
        assert any("excellent" in s for s in result["interpretasi"])

    def test_der_calculation(self):
        result = hitung_rasio_keuangan(
            emiten="BBCA",
            total_hutang=500000,
            ekuitas=210000,
        )
        expected = round(500000 / 210000, 2)
        assert result["rasio"]["DER"] == f"{expected}x"

    def test_npm_calculation(self):
        result = hitung_rasio_keuangan(
            emiten="BBCA",
            laba_bersih=48600,
            pendapatan=149200,
        )
        expected = round((48600 / 149200) * 100, 2)
        assert result["rasio"]["NPM"] == f"{expected}%"

    def test_roa_calculation(self):
        result = hitung_rasio_keuangan(
            emiten="BBCA",
            laba_bersih=48600,
            total_aset=1408000,
        )
        expected = round((48600 / 1408000) * 100, 2)
        assert result["rasio"]["ROA"] == f"{expected}%"

    def test_no_data_returns_catatan(self):
        result = hitung_rasio_keuangan(emiten="BBCA")
        assert result["rasio"] == {}
        assert len(result["catatan"]) > 0

    def test_emiten_uppercase(self):
        result = hitung_rasio_keuangan(emiten="bbca", harga_saham=9500, eps=485)
        assert result["emiten"] == "BBCA"

    def test_all_ratios_computed(self):
        result = hitung_rasio_keuangan(
            emiten="BBCA",
            harga_saham=9500,
            eps=485,
            book_value_per_share=2500,
            laba_bersih=48600,
            ekuitas=210000,
            total_hutang=500000,
            total_aset=1408000,
            pendapatan=149200,
        )
        assert "PER" in result["rasio"]
        assert "PBV" in result["rasio"]
        assert "ROE" in result["rasio"]
        assert "DER" in result["rasio"]
        assert "NPM" in result["rasio"]
        assert "ROA" in result["rasio"]


# ---------------------------------------------------------------------------
# bandingkan_emiten Tests
# ---------------------------------------------------------------------------

class TestBandingkanEmiten:

    def _bbca(self) -> dict:
        return {
            "kode": "BBCA",
            "nama": "Bank Central Asia",
            "roe": 23.5,
            "der": 4.2,
            "npm": 32.6,
            "eps": 485,
            "pertumbuhan_laba": 19.5,
        }

    def _bbri(self) -> dict:
        return {
            "kode": "BBRI",
            "nama": "Bank Rakyat Indonesia",
            "roe": 18.2,
            "der": 5.1,
            "npm": 28.4,
            "eps": 320,
            "pertumbuhan_laba": 17.5,
        }

    def test_returns_comparison_list(self):
        result = bandingkan_emiten(self._bbca(), self._bbri())
        assert "perbandingan" in result
        assert isinstance(result["perbandingan"], list)
        assert len(result["perbandingan"]) > 0

    def test_bbca_wins_roe(self):
        """BBCA ROE 23.5% > BBRI 18.2%"""
        result = bandingkan_emiten(self._bbca(), self._bbri())
        roe_row = next(
            r for r in result["perbandingan"]
            if "ROE" in r["metrik"]
        )
        assert roe_row["lebih_baik"] == "BBCA"

    def test_bbca_wins_der(self):
        """BBCA DER 4.2x < BBRI 5.1x (lower is better)"""
        result = bandingkan_emiten(self._bbca(), self._bbri())
        der_row = next(
            r for r in result["perbandingan"]
            if "DER" in r["metrik"]
        )
        assert der_row["lebih_baik"] == "BBCA"

    def test_rekomendasi_bbca(self):
        result = bandingkan_emiten(self._bbca(), self._bbri())
        assert "BBCA" in result["rekomendasi"]

    def test_skor_in_result(self):
        result = bandingkan_emiten(self._bbca(), self._bbri())
        assert "BBCA" in result["skor"]
        assert "BBRI" in result["skor"]
        assert result["skor"]["BBCA"] >= result["skor"]["BBRI"]

    def test_equal_metrics_shows_sama(self):
        emiten_a = {"kode": "A", "roe": 20.0}
        emiten_b = {"kode": "B", "roe": 20.0}
        result = bandingkan_emiten(emiten_a, emiten_b)
        roe_row = next(r for r in result["perbandingan"] if "ROE" in r["metrik"])
        assert roe_row["lebih_baik"] == "Sama"

    def test_missing_metrics_handled(self):
        """Metrics not provided should be skipped gracefully."""
        emiten_a = {"kode": "A", "roe": 20.0}
        emiten_b = {"kode": "B", "roe": 18.0}
        result = bandingkan_emiten(emiten_a, emiten_b)
        assert isinstance(result["perbandingan"], list)
        # Only ROE should be in comparison (other metrics not provided)
        assert len(result["perbandingan"]) == 1


# ---------------------------------------------------------------------------
# execute_tool Tests
# ---------------------------------------------------------------------------

class TestExecuteTool:

    def test_unknown_tool_returns_error(self):
        result_str = execute_tool("nonexistent_tool", {})
        result = json.loads(result_str)
        assert "error" in result
        assert "nonexistent_tool" in result["error"]

    def test_valid_tool_returns_json(self):
        result_str = execute_tool(
            "hitung_rasio_keuangan",
            {"emiten": "BBCA", "harga_saham": 9500, "eps": 485},
        )
        result = json.loads(result_str)
        assert "rasio" in result
        assert result["emiten"] == "BBCA"

    def test_invalid_params_returns_error(self):
        """Passing wrong param types should return error, not crash."""
        result_str = execute_tool(
            "hitung_rasio_keuangan",
            {"unknown_param": "invalid"},
        )
        # Should either work (with defaults) or return error gracefully
        result = json.loads(result_str)
        assert isinstance(result, dict)

    def test_all_tools_registered(self):
        assert "hitung_rasio_keuangan" in TOOL_REGISTRY
        assert "bandingkan_emiten" in TOOL_REGISTRY
        assert "cari_di_laporan" in TOOL_REGISTRY

    def test_tool_definitions_have_required_fields(self):
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"


# ---------------------------------------------------------------------------
# FinancialAnalystAgent Tests (mocked)
# ---------------------------------------------------------------------------

class TestFinancialAnalystAgent:

    def _mock_end_turn_response(self, text: str) -> MagicMock:
        """Mock a response with stop_reason = end_turn."""
        content_block = MagicMock()
        content_block.type = "text"
        content_block.text = text
        usage = MagicMock()
        usage.input_tokens = 150
        usage.output_tokens = 100
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content = [content_block]
        response.usage = usage
        return response

    def _mock_tool_use_response(
        self,
        tool_name: str,
        tool_input: dict,
        tool_id: str = "toolu_001",
    ) -> MagicMock:
        """Mock a response with stop_reason = tool_use."""
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = tool_id
        tool_block.name = tool_name
        tool_block.input = tool_input
        usage = MagicMock()
        usage.input_tokens = 200
        usage.output_tokens = 50
        response = MagicMock()
        response.stop_reason = "tool_use"
        response.content = [tool_block]
        response.usage = usage
        return response

    @patch("src.api.tools.get_client")
    def test_simple_chat_no_tools(self, mock_client):
        """Direct answer without tool use."""
        mock_client.return_value.messages.create.return_value = (
            self._mock_end_turn_response("BCA adalah bank terbesar di Indonesia.")
        )
        agent = FinancialAnalystAgent()
        response = agent.chat("Apa itu BCA?")
        assert isinstance(response, AnalystResponse)
        assert "BCA" in response.answer
        assert response.tools_used == []
        assert response.turns == 1

    @patch("src.api.tools.get_client")
    def test_chat_with_tool_use(self, mock_client):
        """Agent should execute tool and return final answer."""
        tool_response = self._mock_tool_use_response(
            "hitung_rasio_keuangan",
            {"emiten": "BBCA", "harga_saham": 9500, "eps": 485},
        )
        final_response = self._mock_end_turn_response(
            "PER BBCA adalah 19.59x, tergolong fairly valued."
        )
        mock_client.return_value.messages.create.side_effect = [
            tool_response,
            final_response,
        ]

        agent = FinancialAnalystAgent()
        response = agent.chat("Hitung PER BBCA harga 9500, EPS 485")

        assert "hitung_rasio_keuangan" in response.tools_used
        assert "PER" in response.answer
        assert response.turns == 2
        assert len(response.tool_results) == 1

    @patch("src.api.tools.get_client")
    def test_multi_turn_conversation(self, mock_client):
        """History should be passed correctly for multi-turn."""
        mock_client.return_value.messages.create.return_value = (
            self._mock_end_turn_response("Tentu, BBRI memiliki ROE 18,2%.")
        )
        agent = FinancialAnalystAgent()

        # First turn
        history = [
            {"role": "user", "content": "Bagaimana ROE BBCA?"},
            {"role": "assistant", "content": "ROE BBCA adalah 23,5%."},
        ]

        response = agent.chat(
            "Bagaimana ROE BBRI?",
            history=history,
        )

        # Verify history was included in API call
        call_args = mock_client.return_value.messages.create.call_args
        messages_sent = call_args.kwargs.get("messages") or call_args[1].get("messages")
        assert len(messages_sent) >= 3  # 2 history + 1 new

    @patch("src.api.tools.get_client")
    def test_token_usage_accumulated(self, mock_client):
        """Token usage should accumulate across all API calls."""
        tool_resp = self._mock_tool_use_response(
            "hitung_rasio_keuangan", {"emiten": "BBCA"}
        )
        final_resp = self._mock_end_turn_response("Hasil analisis.")
        mock_client.return_value.messages.create.side_effect = [
            tool_resp, final_resp
        ]

        agent = FinancialAnalystAgent()
        response = agent.chat("Analisis BBCA")
        assert response.input_tokens == 200 + 150  # accumulated
        assert response.output_tokens == 50 + 100

    @patch("src.api.tools.get_client")
    def test_max_tool_rounds_protection(self, mock_client):
        """Should stop after max_tool_rounds even if Claude keeps calling tools."""
        tool_resp = self._mock_tool_use_response(
            "hitung_rasio_keuangan", {"emiten": "BBCA"}
        )
        # Always return tool_use, never end_turn
        mock_client.return_value.messages.create.return_value = tool_resp

        agent = FinancialAnalystAgent(max_tool_rounds=3)
        response = agent.chat("Test infinite loop")
        assert response.turns == 3
