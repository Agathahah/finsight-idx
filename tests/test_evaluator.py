"""
Tests for src/api/evaluator.py
================================
All tests use mocks — no real API calls, no rouge computation on empty strings.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"

from src.api.evaluator import (  # noqa: E402
    EvalSample,
    EvalResult,
    EvalReport,
    RougeScores,
    JudgeScores,
    SummaryEvaluator,
    get_builtin_eval_dataset,
    SCORE_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SOURCE = (
    "BCA membukukan laba bersih Rp 48,6 triliun pada 2023, meningkat 19,5% YoY. "
    "Total aset tumbuh 8,7% menjadi Rp 1.408 triliun. CAR 25,9%."
)

SAMPLE_REFERENCE = (
    "BCA laba bersih Rp 48,6 T (+19,5%), aset Rp 1.408 T (+8,7%), CAR 25,9%."
)

SAMPLE_CANDIDATE = (
    "BCA membukukan laba bersih Rp 48,6 triliun, naik 19,5%. "
    "Total aset Rp 1.408 triliun. Rasio modal 25,9%."
)

VALID_JUDGE_JSON = json.dumps({
    "factual_accuracy": 5,
    "financial_relevance": 4,
    "completeness": 4,
    "conciseness": 5,
    "reasoning": "Ringkasan akurat dan mencakup metrik utama.",
})


def _make_api_response(text: str) -> MagicMock:
    content = MagicMock()
    content.text = text
    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 80
    response = MagicMock()
    response.content = [content]
    response.usage = usage
    return response


def _make_evaluator(tmpdir: str) -> SummaryEvaluator:
    return SummaryEvaluator(
        judge_model="claude-sonnet-4-20250514",
        output_dir=tmpdir,
    )


# ---------------------------------------------------------------------------
# get_builtin_eval_dataset Tests
# ---------------------------------------------------------------------------

class TestBuiltinDataset:

    def test_returns_10_samples(self):
        dataset = get_builtin_eval_dataset()
        assert len(dataset) == 10

    def test_all_samples_have_required_fields(self):
        for sample in get_builtin_eval_dataset():
            assert sample.sample_id
            assert len(sample.source_text) > 50
            assert len(sample.reference_summary) > 20
            assert sample.company
            assert sample.year == 2023

    def test_covers_multiple_companies(self):
        companies = {s.company for s in get_builtin_eval_dataset()}
        assert len(companies) >= 5

    def test_covers_multiple_section_types(self):
        types = {s.section_type for s in get_builtin_eval_dataset()}
        assert "financial" in types
        assert "risk" in types

    def test_indonesian_text_in_sources(self):
        for sample in get_builtin_eval_dataset():
            assert any(
                word in sample.source_text.lower()
                for word in ["triliun", "laba", "kredit", "aset", "risiko"]
            )


# ---------------------------------------------------------------------------
# RougeScores Tests
# ---------------------------------------------------------------------------

class TestRougeScores:

    def test_default_values_are_zero(self):
        scores = RougeScores()
        assert scores.rouge1_f1 == 0.0
        assert scores.rougeL_f1 == 0.0
        assert scores.length_ratio == 0.0

    def test_rouge_computation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = _make_evaluator(tmpdir)
            scores = evaluator._compute_rouge(SAMPLE_CANDIDATE, SAMPLE_REFERENCE)
            assert 0.0 < scores.rouge1_f1 <= 1.0
            assert 0.0 < scores.rougeL_f1 <= 1.0
            assert scores.length_ratio > 0

    def test_identical_text_gives_high_rouge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = _make_evaluator(tmpdir)
            scores = evaluator._compute_rouge(SAMPLE_REFERENCE, SAMPLE_REFERENCE)
            assert scores.rouge1_f1 == 1.0
            assert scores.rougeL_f1 == 1.0

    def test_empty_candidate_gives_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = _make_evaluator(tmpdir)
            scores = evaluator._compute_rouge("", SAMPLE_REFERENCE)
            assert scores.rouge1_f1 == 0.0


# ---------------------------------------------------------------------------
# JudgeScores Tests
# ---------------------------------------------------------------------------

class TestJudgeScores:

    def test_default_scores_zero(self):
        scores = JudgeScores()
        assert scores.factual_accuracy == 0
        assert scores.overall == 0.0

    def test_parse_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = _make_evaluator(tmpdir)
            scores = evaluator._parse_judge_response(VALID_JUDGE_JSON)
            assert scores.factual_accuracy == 5
            assert scores.financial_relevance == 4
            assert scores.completeness == 4
            assert scores.conciseness == 5
            assert scores.reasoning == "Ringkasan akurat dan mencakup metrik utama."

    def test_overall_weighted_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = _make_evaluator(tmpdir)
            scores = evaluator._parse_judge_response(VALID_JUDGE_JSON)
            expected = (
                5 * SCORE_WEIGHTS["factual_accuracy"]
                + 4 * SCORE_WEIGHTS["financial_relevance"]
                + 4 * SCORE_WEIGHTS["completeness"]
                + 5 * SCORE_WEIGHTS["conciseness"]
            )
            assert abs(scores.overall - expected) < 0.01

    def test_parse_json_with_markdown_fences(self):
        """Should handle ```json ... ``` wrapper from Claude."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = _make_evaluator(tmpdir)
            wrapped = f"```json\n{VALID_JUDGE_JSON}\n```"
            scores = evaluator._parse_judge_response(wrapped)
            assert scores.factual_accuracy == 5

    def test_parse_invalid_json_returns_zeros(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = _make_evaluator(tmpdir)
            scores = evaluator._parse_judge_response("invalid response here")
            assert scores.factual_accuracy == 0
            assert scores.overall == 0.0
            assert "invalid response" in scores.raw_response


# ---------------------------------------------------------------------------
# SummaryEvaluator Tests
# ---------------------------------------------------------------------------

class TestSummaryEvaluator:

    @patch("src.api.evaluator.analyze_financial_text")
    @patch("src.api.client.get_client")
    def test_evaluate_single_returns_result(self, mock_get_client, mock_analyze):
        """evaluate_single should produce an EvalResult with all fields."""
        mock_analyze.return_value = {
            "result": SAMPLE_CANDIDATE,
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        mock_get_client.return_value.messages.create.return_value = (
            _make_api_response(VALID_JUDGE_JSON)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = _make_evaluator(tmpdir)
            result = evaluator.evaluate_single(
                source_text=SAMPLE_SOURCE,
                candidate_summary=SAMPLE_CANDIDATE,
                reference_summary=SAMPLE_REFERENCE,
                sample_id="test_001",
            )

        assert isinstance(result, EvalResult)
        assert result.sample_id == "test_001"
        assert result.rouge.rouge1_f1 > 0
        assert result.judge.factual_accuracy == 5

    @patch("src.api.evaluator.analyze_financial_text")
    @patch("src.api.client.get_client")
    def test_evaluate_dataset_aggregates_correctly(
        self, mock_get_client, mock_analyze
    ):
        """evaluate_dataset should return aggregated EvalReport."""
        mock_analyze.return_value = {
            "result": SAMPLE_CANDIDATE,
            "usage": {"input_tokens": 80, "output_tokens": 40},
        }
        mock_get_client.return_value.messages.create.return_value = (
            _make_api_response(VALID_JUDGE_JSON)
        )

        dataset = [
            EvalSample(
                sample_id=f"s{i}",
                company="BBCA",
                year=2023,
                source_text=SAMPLE_SOURCE,
                reference_summary=SAMPLE_REFERENCE,
            )
            for i in range(3)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = _make_evaluator(tmpdir)
            report = evaluator.evaluate_dataset(dataset)

        assert report.n_samples == 3
        assert report.avg_rouge1_f1 > 0
        assert report.avg_factual_accuracy == 5.0
        assert len(report.results) == 3

    def test_skip_judge_mode(self):
        """skip_judge=True should skip API calls for judge."""
        with patch("src.api.evaluator.analyze_financial_text") as mock_analyze:
            mock_analyze.return_value = {
                "result": SAMPLE_CANDIDATE,
                "usage": {"input_tokens": 80, "output_tokens": 40},
            }
            dataset = [
                EvalSample(
                    sample_id="s0",
                    company="BBCA",
                    year=2023,
                    source_text=SAMPLE_SOURCE,
                    reference_summary=SAMPLE_REFERENCE,
                )
            ]
            with tempfile.TemporaryDirectory() as tmpdir:
                evaluator = _make_evaluator(tmpdir)
                report = evaluator.evaluate_dataset(dataset, skip_judge=True)

            assert report.avg_overall_judge == 0.0
            assert report.avg_rouge1_f1 > 0

    def test_save_report_creates_file(self):
        """save_report should write JSON to output_dir."""
        report = EvalReport(
            report_id="test_report",
            created_at="2023-01-01T00:00:00",
            n_samples=2,
            eval_model="claude-sonnet-4-20250514",
            avg_rouge1_f1=0.65,
            avg_rougeL_f1=0.60,
            avg_overall_judge=4.2,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = _make_evaluator(tmpdir)
            path = evaluator.save_report(report, "test_report.json")
            assert path.exists()
            saved = json.loads(path.read_text())
            assert saved["report_id"] == "test_report"
            assert saved["avg_rouge1_f1"] == 0.65


# ---------------------------------------------------------------------------
# EvalReport Tests
# ---------------------------------------------------------------------------

class TestEvalReport:

    def _make_report(self) -> EvalReport:
        return EvalReport(
            report_id="rpt_001",
            created_at="2023-12-01T10:00:00",
            n_samples=10,
            eval_model="claude-sonnet-4-20250514",
            avg_rouge1_f1=0.623,
            avg_rouge2_f1=0.412,
            avg_rougeL_f1=0.598,
            avg_factual_accuracy=4.3,
            avg_financial_relevance=4.5,
            avg_completeness=4.1,
            avg_conciseness=4.4,
            avg_overall_judge=4.33,
        )

    def test_to_json_valid(self):
        report = self._make_report()
        parsed = json.loads(report.to_json())
        assert parsed["report_id"] == "rpt_001"
        assert parsed["avg_rouge1_f1"] == 0.623

    def test_markdown_table_contains_metrics(self):
        report = self._make_report()
        md = report.markdown_table()
        assert "ROUGE-1" in md
        assert "ROUGE-L" in md
        assert "Factual Accuracy" in md
        assert "0.623" in md
        assert "4.33" in md

    def test_to_json_serializable(self):
        report = self._make_report()
        json.dumps(report.to_dict())  # Should not raise
