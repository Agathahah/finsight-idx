"""Answer-level evaluation: is the generated answer right, and does it cite the right page?

Retrieval metrics (eval_retrieval.py) say whether the right page reached the
model. They do not say whether Claude then read the number correctly, cited
the page, or refused when the document has no answer. This harness checks
those three things against a hand-labelled set.

Golden answers (JSON Lines), one question per line. The values below only
show the format; take the real number and page from the PDF yourself:
    {"question": "Berapa laba bersih BCA tahun 2023?", "company": "BBCA", "year": 2023,
     "relevant_pages": [16], "expected": ["48.639"]}
    {"question": "Berapa laba bersih BBRI tahun 2023?", "company": "BBCA", "year": 2023,
     "answerable": false}

``expected`` lists values that must all appear in the answer. Text values are
matched case-insensitively. Numbers are matched by value, not by spelling:
Indonesian formatting is parsed ("1.408.107", "29,4"), and a figure reported in
another scale is accepted when it rounds to the expected one, so "Rp1.408,1
triliun" matches an expected "1.408.107" from a table in miliar Rupiah. Run:
    python -m src.rag.eval_answers eval/golden_questions.jsonl --limit 20
    python -m src.rag.eval_answers eval/golden_questions.jsonl --rescore reports/answer_eval.json
The second form re-scores saved answers without calling the API.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Callable

from src.serving.telemetry import estimate_cost_usd

ABSTAIN_PATTERNS = re.compile(r"(?i)tidak tersedia|tidak ditemukan|tidak terdapat informasi|not available")
NUMBER = re.compile(r"\d[\d.,]*\d|\d")
CITED_PAGE = re.compile(r"(?i)hal(?:aman)?\.?\s*(\d{1,4})")


def digits(s: str) -> str:
    return re.sub(r"[^\d]", "", s)


def numbers_in(text: str) -> set[str]:
    return {digits(m) for m in NUMBER.findall(text)}


def parse_number(token: str) -> tuple[float, int] | None:
    """Parse an Indonesian-formatted number; return (value, decimals).

    "." groups thousands and "," marks decimals: "1.408.107" -> 1408107,
    "29,4" -> 29.4, "1.408,1" -> 1408.1, "205,00" -> 205.0.
    """
    t = token.strip(".,")
    if not t or not t[0].isdigit():
        return None
    if "," in t:
        whole, _, frac = t.rpartition(",")
        if not frac.isdigit() or (whole and not re.fullmatch(r"\d{1,3}(\.\d{3})*|\d+", whole)):
            return None
        whole = whole.replace(".", "") or "0"
        return float(f"{whole}.{frac}"), len(frac)
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", t) or t.isdigit():
        return float(t.replace(".", "")), 0
    return None


SCALES = (1.0, 1e3, 1e6, 1e-3, 1e-6)   # juta <-> miliar <-> triliun


def number_matches(answer: str, expected: str) -> bool:
    """True when some number in ``answer`` equals ``expected`` up to rounding and scale."""
    e = parse_number(expected)
    if e is None:
        return digits(expected) in numbers_in(answer)
    e_val, e_dec = e
    for tok in NUMBER.findall(answer):
        a = parse_number(tok)
        if a is None:
            continue
        a_val, a_dec = a
        for k in SCALES:
            if k != 1.0 and (e_val < 1000 or len(digits(tok)) < 3):
                continue          # rescaling small numbers would match almost anything
            tol = 0.5 * 10 ** -a_dec * k + 0.5 * 10 ** -e_dec
            if abs(a_val * k - e_val) <= tol:
                return True
    return False


def contains_expected(answer: str, expected: list[str]) -> bool:
    low = answer.lower()
    for e in expected:
        if any(ch.isdigit() for ch in e):
            if not number_matches(answer, e):
                return False
        elif e.lower() not in low:
            return False
    return True


def cited_pages(answer: str) -> set[int]:
    return {int(p) for p in CITED_PAGE.findall(answer)}


def is_abstention(answer: str) -> bool:
    return bool(ABSTAIN_PATTERNS.search(answer))


def load_golden(path: str | Path) -> list[dict]:
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("question"):
            raise ValueError(f"line {i}: missing 'question'")
        if r.get("answerable", True) and not r.get("expected"):
            raise ValueError(f"line {i}: answerable question needs 'expected' (or set answerable: false)")
        rows.append(r)
    return rows


def score_one(item: dict, answer: str, page_texts: dict[int, str] | None = None) -> dict:
    """Score one answer.

    correct:     every expected value is in the answer (a hedge elsewhere in the
                 answer does not cancel a correct figure).
    citation_ok: a cited page is a labelled page, or the indexed text of a cited
                 page contains the expected values. Labels list a few pages; a
                 figure such as total assets is printed on ten.
    """
    answerable = item.get("answerable", True)
    abstained = is_abstention(answer)
    if not answerable:
        return {"answerable": False, "correct": abstained, "abstained": abstained, "citation_ok": None}
    # Sentences that refuse ("informasi tahun 2023 tidak tersedia") may echo the
    # expected value; only the rest of the answer can make it correct.
    kept = " ".join(x for x in re.split(r"(?<=[.!?])\s+|\n+", answer) if not is_abstention(x))
    correct = contains_expected(kept, item["expected"])
    pages = cited_pages(answer)
    relevant = set(item.get("relevant_pages") or [])
    citation_ok, basis = None, None
    if pages & relevant:
        citation_ok, basis = True, "label"
    elif page_texts is not None and any(
            contains_expected(page_texts.get(p, ""), item["expected"]) for p in pages):
        citation_ok, basis = True, "page_text"
    elif relevant:
        citation_ok = False
    return {
        "answerable": True,
        "correct": correct,
        "abstained": abstained and not correct,
        "citation_ok": citation_ok,
        "citation_basis": basis,
        "cited_pages": sorted(pages),
    }


def load_page_texts(persist_dir: str | Path) -> dict[int, str] | None:
    """Indexed text per PDF page, read straight from the Chroma SQLite file."""
    import sqlite3
    db = Path(persist_dir) / "chroma.sqlite3"
    if not db.exists():
        return None
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "select p.int_value, d.string_value from embedding_metadata d "
        "join embedding_metadata p on d.id = p.id and p.key = 'page_number' "
        "where d.key = 'chroma:document'").fetchall()
    pages: dict[int, list[str]] = {}
    for page, text in rows:
        pages.setdefault(page, []).append(text)
    return {p: "\n".join(t) for p, t in pages.items()}


class EvalAborted(RuntimeError):
    """The first questions all failed: almost certainly configuration, not quality."""


def evaluate(ask_fn: Callable[[dict], object], golden: list[dict], abort_after: int = 3,
             page_texts: dict[int, str] | None = None) -> dict:
    """``ask_fn(item)`` returns an object with answer, input_tokens, output_tokens.

    A failing question (API error, timeout) is recorded and the run continues,
    so one bad call does not throw away the money already spent. If the first
    ``abort_after`` questions all fail, the run stops: that is a setup problem
    (API key, model id, credit), and continuing would only repeat it.
    """
    per_q, errors = [], []
    for n, item in enumerate(golden, 1):
        t0 = time.perf_counter()
        try:
            resp = ask_fn(item)
        except Exception as exc:  # noqa: BLE001 - recorded and reported
            errors.append({"question": item["question"], "error": f"{type(exc).__name__}: {exc}"[:300]})
            print(f"  [{n}/{len(golden)}] ERROR {errors[-1]['error']}", flush=True)
            if len(errors) == n == abort_after:
                raise EvalAborted(f"first {abort_after} questions failed; last error: {errors[-1]['error']}")
            continue
        latency = (time.perf_counter() - t0) * 1000
        s = score_one(item, resp.answer, page_texts)
        s.update(question=item["question"], latency_ms=round(latency, 1),
                 input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                 answer=resp.answer[:800])
        per_q.append(s)
        print(f"  [{n}/{len(golden)}] {'OK  ' if s['correct'] else 'FAIL'} {item['question'][:70]}", flush=True)

    return summarize(per_q, errors)


def summarize(per_q: list[dict], errors: list[dict] | None = None) -> dict:
    errors = errors or []
    ans = [q for q in per_q if q["answerable"]]
    unans = [q for q in per_q if not q["answerable"]]
    cites = [q for q in ans if q["citation_ok"] is not None]
    tin = sum(q["input_tokens"] for q in per_q)
    tout = sum(q["output_tokens"] for q in per_q)

    def rate(xs, key):
        return round(sum(bool(x[key]) for x in xs) / len(xs), 4) if xs else None

    return {
        "questions": len(per_q),
        "answer_accuracy": rate(ans, "correct"),
        "citation_accuracy": rate(cites, "citation_ok"),
        "citations_found_by_page_text": sum(q.get("citation_basis") == "page_text" for q in ans),
        "false_refusal_rate": rate(ans, "abstained"),
        "correct_refusal_rate": rate(unans, "correct"),
        "mean_latency_ms": round(sum(q["latency_ms"] for q in per_q) / max(len(per_q), 1), 1),
        "total_tokens": {"input": tin, "output": tout},
        "cost_usd": estimate_cost_usd(tin, tout),
        "errors": errors,
        "failures": [
            {"question": q["question"], "abstained": q["abstained"], "cited_pages": q.get("cited_pages"),
             "answer": q.get("answer", "")[:300]}
            for q in per_q if not q["correct"]
        ],
        "per_question": per_q,
    }


def markdown_summary(report: dict) -> str:
    def pct(v):
        return "–" if v is None else f"{v:.0%}"
    return "\n".join([
        "| Metric | Value |", "|---|---|",
        f"| Questions scored | {report['questions']} (errors: {len(report.get('errors', []))}) |",
        f"| Answer accuracy (answerable) | {pct(report['answer_accuracy'])} |",
        f"| Citation accuracy | {pct(report['citation_accuracy'])} |",
        f"| False refusals | {pct(report['false_refusal_rate'])} |",
        f"| Correct refusals (unanswerable) | {pct(report['correct_refusal_rate'])} |",
        f"| Mean latency | {report['mean_latency_ms']:.0f} ms |",
        f"| Cost of this run | ${report['cost_usd']:.4f} |",
    ])


def rescore(report_path: str | Path, golden: list[dict], page_texts: dict[int, str] | None) -> dict:
    """Score saved answers again (new scoring rules, fixed labels) without any API call."""
    old = json.loads(Path(report_path).read_text(encoding="utf-8"))
    by_q = {g["question"]: g for g in golden}
    per_q = []
    for q in old["per_question"]:
        item = by_q.get(q["question"])
        if item is None:
            continue
        s = score_one(item, q.get("answer", ""), page_texts)
        s.update({k: q[k] for k in ("question", "latency_ms", "input_tokens", "output_tokens", "answer") if k in q})
        per_q.append(s)
    report = summarize(per_q, old.get("errors"))
    report["config"] = {**old.get("config", {}), "rescored": True}
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("golden")
    ap.add_argument("--limit", type=int, default=20, help="max questions (each one is a paid API call)")
    ap.add_argument("--persist-dir", default="data/vectorstore")
    ap.add_argument("--out", default="reports/answer_eval.json")
    ap.add_argument("--top-k", type=int, default=None, help="chunks sent to the model (default: RAG_TOP_K or 5)")
    ap.add_argument("--retrieval-mode", choices=["hybrid", "bm25"], default=None)
    ap.add_argument("--rescore", metavar="REPORT", help="re-score a saved report; no API calls")
    args = ap.parse_args()

    golden = load_golden(args.golden)
    page_texts = load_page_texts(args.persist_dir)

    if args.rescore:
        report = rescore(args.rescore, golden, page_texts)
    else:
        from src.rag.qa_chain import FinancialQAChain
        golden = golden[: args.limit]
        est = estimate_cost_usd(2500 * len(golden), 400 * len(golden))
        print(f"Running {len(golden)} questions (rough cost estimate ${est:.3f})")
        qa = FinancialQAChain(persist_dir=args.persist_dir, top_k=args.top_k,
                              retrieval_mode=args.retrieval_mode)
        try:
            report = evaluate(lambda it: qa.ask(it["question"], company_filter=it.get("company"),
                                                year_filter=it.get("year")), golden, page_texts=page_texts)
        except EvalAborted as exc:
            raise SystemExit(f"\nStopped: {exc}\nCheck ANTHROPIC_API_KEY, CLAUDE_MODEL "
                             "(python scripts/check_model.py) and your API credit.")
        report["config"] = {"model": qa.model, "top_k": qa.top_k, "retrieval_mode": qa.retrieval_mode}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(markdown_summary(report))
    if report["errors"]:
        print(f"\n{len(report['errors'])} question(s) errored (not scored):")
        for e in report["errors"][:5]:
            print(" -", e["question"][:60], "|", e["error"][:120])
    if report["failures"]:
        print("\nFailures:")
        for f in report["failures"]:
            print(" -", f["question"], "| refused" if f["abstained"] else f"| cited {f['cited_pages']}")
    print(f"\nFull report: {args.out}")


if __name__ == "__main__":
    main()
