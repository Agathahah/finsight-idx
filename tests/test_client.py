"""
Tests for src/api/client.py
============================
Uses pytest + unittest.mock to test the Claude API client
without making real API calls (no tokens consumed, no cost).
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# Set fake API key BEFORE importing the module under test
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"

from src.api.client import (  # noqa: E402
    FinancialTask,
    analyze_financial_text,
    get_client,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """
BCA membukukan laba bersih sebesar Rp 48,6 triliun pada 2023,
meningkat 19,5% YoY. Total aset tumbuh 8,7% menjadi Rp 1.408 triliun.
"""


def _make_mock_response(result_text: str = "Ringkasan: Kinerja positif.") -> MagicMock:
    """Build a minimal mock that mimics anthropic.types.Message."""
    content_block = MagicMock()
    content_block.text = result_text

    usage = MagicMock()
    usage.input_tokens = 120
    usage.output_tokens = 80

    response = MagicMock()
    response.content = [content_block]
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# Client instantiation tests
# ---------------------------------------------------------------------------

class TestGetClient:
    def test_returns_anthropic_client(self):
        """get_client() should return an Anthropic instance."""
        import anthropic
        client = get_client()
        assert isinstance(client, anthropic.Anthropic)

    def test_raises_if_no_api_key(self, monkeypatch):
        """Should raise EnvironmentError when ANTHROPIC_API_KEY is missing."""
        import src.api.client as client_module

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        client_module._client = None  # reset singleton

        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            client_module.get_client()

        # Restore for subsequent tests
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        client_module._client = None


# ---------------------------------------------------------------------------
# analyze_financial_text tests
# ---------------------------------------------------------------------------

class TestAnalyzeFinancialText:
    @patch("src.api.client.get_client")
    def test_summarize_returns_expected_keys(self, mock_get_client):
        """Result dict must contain task, model, result, usage."""
        mock_get_client.return_value.messages.create.return_value = (
            _make_mock_response("Ringkasan singkat.")
        )

        output = analyze_financial_text(SAMPLE_TEXT, task=FinancialTask.SUMMARIZE)

        assert output["task"] == "summarize"
        assert "result" in output
        assert "usage" in output
        assert output["usage"]["input_tokens"] == 120
        assert output["usage"]["output_tokens"] == 80

    @patch("src.api.client.get_client")
    def test_sentiment_task(self, mock_get_client):
        """Sentiment task should use the correct system prompt."""
        mock_get_client.return_value.messages.create.return_value = (
            _make_mock_response('{"sentiment": "Positif", "score": 0.85}')
        )

        output = analyze_financial_text(SAMPLE_TEXT, task=FinancialTask.SENTIMENT)
        assert output["task"] == "sentiment"
        assert "Positif" in output["result"]

    @patch("src.api.client.get_client")
    def test_key_metrics_task(self, mock_get_client):
        """Key metrics task should return a dict with result containing metrics."""
        mock_get_client.return_value.messages.create.return_value = (
            _make_mock_response('{"laba_bersih": "48.6 triliun IDR"}')
        )

        output = analyze_financial_text(SAMPLE_TEXT, task=FinancialTask.KEY_METRICS)
        assert output["task"] == "key_metrics"

    @patch("src.api.client.get_client")
    def test_all_tasks_are_supported(self, mock_get_client):
        """Every FinancialTask enum value should be callable without error."""
        mock_get_client.return_value.messages.create.return_value = (
            _make_mock_response("ok")
        )

        for task in FinancialTask:
            output = analyze_financial_text(SAMPLE_TEXT, task=task)
            assert output["task"] == task.value

    def test_raises_on_empty_text(self):
        """Empty or whitespace-only text should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            analyze_financial_text("   ")

        with pytest.raises(ValueError, match="non-empty"):
            analyze_financial_text("")

    @patch("src.api.client.get_client")
    def test_custom_model_passed_to_api(self, mock_get_client):
        """Custom model parameter must be forwarded to the API call."""
        mock_get_client.return_value.messages.create.return_value = (
            _make_mock_response("ok")
        )
        custom_model = "claude-opus-4-20250514"

        output = analyze_financial_text(
            SAMPLE_TEXT, task=FinancialTask.SUMMARIZE, model=custom_model
        )
        assert output["model"] == custom_model

        _, kwargs = mock_get_client.return_value.messages.create.call_args
        assert kwargs["model"] == custom_model

    @patch("src.api.client.get_client")
    def test_rate_limit_retries(self, mock_get_client):
        """Should retry on RateLimitError and eventually raise after max attempts."""
        import anthropic

        mock_get_client.return_value.messages.create.side_effect = (
            anthropic.RateLimitError(
                message="rate limit",
                response=MagicMock(status_code=429, headers={}),
                body={},
            )
        )

        with pytest.raises(anthropic.RateLimitError):
            analyze_financial_text(SAMPLE_TEXT)

        # Should have retried MAX_RETRY_ATTEMPTS times
        assert mock_get_client.return_value.messages.create.call_count == 3
