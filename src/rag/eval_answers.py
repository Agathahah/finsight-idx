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

``expected`` lists strings that must all appear in the answer. Numbers are
compared by their digits only, so "48.639", "48,639" and "Rp48.639 miliar"
all match "48.639". Run:
    python -m src.rag.eval_answers eval/golden_answers.jsonl --limit 20
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


def contains_expected(answer: str, expected: list[str]) -> bool:
    nums = numbers_in(answer)
    low = answer.lower()
    for e in expected:
        if any(ch.isdigit() for ch in e):
            if digits(e) not in nums:
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


def score_one(item: dict, answer: str) -> dict:
    answerable = item.get("answerable", True)
    abstained = is_abstention(answer)
    if not answerable:
        return {"answerable": False, "correct": abstained, "abstained": abstained, "citation_ok": None}
    pages = cited_pages(answer)
    relevant = set(item.get("relevant_pages") or [])
    return {
        "answerable": True,
        "correct": (not abstained) and contains_expected(answer, item["expected"]),
        "abstained": abstained,
        "citation_ok": bool(pages & relevant) if relevant else None,
        "cited_pages": sorted(pages),
    }


def evaluate(ask_fn: Callable[[dict], object], golden: list[dict]) -> dict:
    """``ask_fn(item)`` returns an object with answer, input_tokens, output_tokens."""
    per_q = []
    for item in golden:
        t0 = time.perf_counter()
        resp = ask_fn(item)
        latency = (time.perf_counter() - t0) * 1000
        s = score_one(item, resp.answer)
        s.update(question=item["question"], latency_ms=round(latency, 1),
                 input_tokens=resp.input_tokens, output_tokens=resp.output_tokens)
        per_q.append(s)

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
        "false_refusal_rate": rate(ans, "abstained"),
        "correct_refusal_rate": rate(unans, "correct"),
        "mean_latency_ms": round(sum(q["latency_ms"] for q in per_q) / max(len(per_q), 1), 1),
        "total_tokens": {"input": tin, "output": tout},
        "cost_usd": estimate_cost_usd(tin, tout),
        "failures": [
            {"question": q["question"], "abstained": q["abstained"], "cited_pages": q.get("cited_pages")}
            for q in per_q if not q["correct"]
        ],
        "per_question": per_q,
    }


def markdown_summary(report: dict) -> str:
    def pct(v):
        return "–" if v is None else f"{v:.0%}"
    return "\n".join([
        "| Metric | Value |", "|---|---|",
        f"| Questions | {report['questions']} |",
        f"| Answer accuracy (answerable) | {pct(report['answer_accuracy'])} |",
        f"| Citation accuracy | {pct(report['citation_accuracy'])} |",
        f"| False refusals | {pct(report['false_refusal_rate'])} |",
        f"| Correct refusals (unanswerable) | {pct(report['correct_refusal_rate'])} |",
        f"| Mean latency | {report['mean_latency_ms']:.0f} ms |",
        f"| Cost of this run | ${report['cost_usd']:.4f} |",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("golden")
    ap.add_argument("--limit", type=int, default=20, help="max questions (each one is a paid API call)")
    ap.add_argument("--persist-dir", default="data/vectorstore")
    ap.add_argument("--out", default="reports/answer_eval.json")
    args = ap.parse_args()

    from src.rag.qa_chain import FinancialQAChain

    golden = load_golden(args.golden)[: args.limit]
    est = estimate_cost_usd(2500 * len(golden), 400 * len(golden))
    print(f"Running {len(golden)} questions (rough cost estimate ${est:.3f})")
    qa = FinancialQAChain(persist_dir=args.persist_dir)
    report = evaluate(lambda it: qa.ask(it["question"], company_filter=it.get("company"),
                                        year_filter=it.get("year")), golden)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(markdown_summary(report))
    if report["failures"]:
        print("\nFailures:")
        for f in report["failures"]:
            print(" -", f["question"], "| refused" if f["abstained"] else f"| cited {f['cited_pages']}")
    print(f"\nFull report: {args.out}")


if __name__ == "__main__":
    main()
