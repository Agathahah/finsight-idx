"""Answer-level evaluation harness."""

from types import SimpleNamespace

import pytest

from src.rag import eval_answers as ea


def _resp(answer, tin=1000, tout=200):
    return SimpleNamespace(answer=answer, input_tokens=tin, output_tokens=tout)


def test_numbers_match_by_value_across_scales():
    assert ea.contains_expected("Laba bersih Rp48.639 miliar [Sumber: BBCA, hal. 16]", ["48.639"])
    assert ea.contains_expected("Total aset Rp1.408,1 triliun", ["1.408.107"])      # triliun vs miliar
    assert ea.contains_expected("Kas Rp21,7 triliun", ["21.701.514"])               # triliun vs juta
    assert ea.contains_expected("dividen Rp205,00 per saham", ["205"])
    assert ea.contains_expected("CAR 29,4%", ["29,4"])
    assert not ea.contains_expected("Laba bersih Rp40.756 miliar", ["48.639"])
    assert not ea.contains_expected("Total aset Rp1.314,7 triliun", ["1.408.107"])
    assert not ea.contains_expected("NPL 1,7%", ["1,9"])
    assert not ea.contains_expected("ada 2 pihak", ["2.000.000"])                   # no rescaling of small numbers


def test_indonesian_number_parsing():
    assert ea.parse_number("1.408.107") == (1408107.0, 0)
    assert ea.parse_number("1.408,1") == (1408.1, 1)
    assert ea.parse_number("29,4") == (29.4, 1)
    assert ea.parse_number("2b,2g") is None


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


def test_errors_are_recorded_and_the_run_continues():
    golden = [{"question": q, "relevant_pages": [1], "expected": ["7"]} for q in "abcd"]
    def ask(it):
        if it["question"] == "b":
            raise TimeoutError("slow")
        return _resp("7 [hal. 1]")
    r = ea.evaluate(ask, golden)
    assert r["questions"] == 3 and r["answer_accuracy"] == 1.0
    assert r["errors"][0]["error"].startswith("TimeoutError")


def test_setup_errors_stop_the_run_early():
    golden = [{"question": str(i), "relevant_pages": [1], "expected": ["7"]} for i in range(10)]
    calls = []
    def ask(it):
        calls.append(1)
        raise RuntimeError("401 invalid x-api-key")
    with pytest.raises(ea.EvalAborted, match="401"):
        ea.evaluate(ask, golden)
    assert len(calls) == 3


def test_context_hit_counts_answer_anywhere_in_top_k():
    from types import SimpleNamespace as NS
    from src.rag.eval_retrieval import evaluate as ev_retrieval
    golden = [{"question": "q", "relevant_pages": [18], "expected": ["1.408.107"]}]
    chunks = [NS(text="lain", page_number=5, company="", year=0),
              NS(text="Total Aset 1.408.107", page_number=158, company="", year=0)]
    r = ev_retrieval(lambda q, it: chunks, golden, k=5)
    assert r["recall_at_k"] == 0.0 and r["context_hit_at_k"] == 1.0


def test_correct_answer_with_a_hedge_still_counts():
    item = {"question": "q", "relevant_pages": [18], "expected": ["1.408.107"]}
    s = ea.score_one(item, "Total aset Rp1.408,1 triliun [hal. 260]. Data 2024 tidak tersedia dalam dokumen.")
    assert s["correct"] and not s["abstained"]


def test_citation_accepted_when_cited_page_holds_the_value():
    item = {"question": "q", "relevant_pages": [18], "expected": ["1.408.107"]}
    pages = {260: "total aset konsolidasi mencapai Rp1.408,1 triliun", 279: "kredit Rp288,6 triliun"}
    assert ea.score_one(item, "Rp1.408,1 triliun [hal. 260]", pages)["citation_basis"] == "page_text"
    assert ea.score_one(item, "Rp1.408,1 triliun [hal. 279]", pages)["citation_ok"] is False


def test_rescore_uses_saved_answers(tmp_path):
    import json
    saved = {"per_question": [{"question": "a", "answer": "Rp1.408,1 triliun [hal. 18]", "latency_ms": 10,
                               "input_tokens": 100, "output_tokens": 10}], "errors": []}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(saved))
    r = ea.rescore(p, [{"question": "a", "relevant_pages": [18], "expected": ["1.408.107"]}], None)
    assert r["answer_accuracy"] == 1.0 and r["config"]["rescored"]
