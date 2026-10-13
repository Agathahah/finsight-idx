"""Answer-level evaluation harness."""

from types import SimpleNamespace

import pytest

from src.rag import eval_answers as ea


def _resp(answer, tin=1000, tout=200):
    return SimpleNamespace(answer=answer, input_tokens=tin, output_tokens=tout)


def test_numbers_match_regardless_of_separators():
    assert ea.contains_expected("Laba bersih Rp48.639 miliar [Sumber: BBCA, hal. 16]", ["48,639"])
    assert not ea.contains_expected("Laba bersih Rp40.756 miliar", ["48.639"])


def test_text_expectations_are_case_insensitive():
    assert ea.contains_expected("Diaudit oleh KAP TANUDIREDJA, Wibisana", ["Tanudiredja"])


def test_cited_pages_are_parsed():
    assert ea.cited_pages("[Sumber: BBCA 2023, hal. 16] dan [BBCA, halaman 117]") == {16, 117}


def test_scoring_answerable_and_unanswerable():
    item = {"question": "q", "relevant_pages": [16], "expected": ["48.639"]}
    s = ea.score_one(item, "Rp48.639 miliar [hal. 16]")
    assert s["correct"] and s["citation_ok"]
    s = ea.score_one(item, "Rp48.639 miliar [hal. 99]")
    assert s["correct"] and not s["citation_ok"]
    refuse = {"question": "q", "answerable": False}
    assert ea.score_one(refuse, "Informasi tidak tersedia dalam dokumen")["correct"]
    assert not ea.score_one(refuse, "Laba BBRI Rp60 triliun")["correct"]


def test_a_refusal_never_counts_as_a_correct_answer():
    item = {"question": "q", "relevant_pages": [16], "expected": ["2023"]}
    assert not ea.score_one(item, "Informasi tahun 2023 tidak tersedia dalam dokumen")["correct"]


def test_report_aggregates(tmp_path):
    golden = [
        {"question": "a", "relevant_pages": [16], "expected": ["48.639"]},
        {"question": "b", "relevant_pages": [20], "expected": ["7,9"]},
        {"question": "c", "answerable": False},
    ]
    answers = {"a": "Rp48.639 miliar [hal. 16]", "b": "Informasi tidak tersedia", "c": "tidak tersedia"}
    r = ea.evaluate(lambda it: _resp(answers[it["question"]]), golden)
    assert r["answer_accuracy"] == 0.5
    assert r["false_refusal_rate"] == 0.5
    assert r["correct_refusal_rate"] == 1.0
    assert r["citation_accuracy"] == 0.5
    assert r["cost_usd"] == pytest.approx(3 * (1000 * 3 + 200 * 15) / 1e6)
    assert [f["question"] for f in r["failures"]] == ["b"]
    assert "Answer accuracy" in ea.markdown_summary(r)


def test_loader_requires_expected_for_answerable(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text('{"question": "x", "relevant_pages": [1]}\n')
    with pytest.raises(ValueError, match="expected"):
        ea.load_golden(p)


def test_example_files_parse():
    from src.rag.eval_retrieval import load_golden as load_retrieval
    assert len(ea.load_golden("eval/golden_answers.example.jsonl")) == 3
    assert len(load_retrieval("eval/golden_answers.example.jsonl")) == 2
