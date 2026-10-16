"""Rebuild the README answer-metrics table from the saved reports.

    python scripts/update_readme_eval.py

Reads every report listed in RUNS that exists and rewrites the block between
<!-- answer-table:start --> and <!-- answer-table:end --> in README.md, so the
README always shows the numbers in reports/ rather than hand-copied ones.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = [  # (column title, report path)
    ("v4 · hybrid · k = 5", "reports/answer_eval_v4_k5.json"),
    ("v4 · BM25 · k = 10 (deployed)", "reports/answer_eval_v4_k10.json"),
]
START, END = "<!-- answer-table:start -->", "<!-- answer-table:end -->"


def pct(v) -> str:
    return "–" if v is None else f"{v:.0%}"


def table(reports: list[tuple[str, dict | None]]) -> str:
    def row(name, fn):
        return f"| {name} | " + " | ".join(fn(r) if r else "not run yet" for _, r in reports) + " |"

    def per_question(r):
        return f"${r['cost_usd'] / max(r['questions'], 1):.4f}"

    lines = [
        "| Answer metric | " + " | ".join(t for t, _ in reports) + " |",
        "|---|" + "---|" * len(reports),
        row("Questions scored", lambda r: f"{r['questions']}"),
        row("Answer accuracy (answerable)", lambda r: pct(r["answer_accuracy"])),
        row("Citation accuracy", lambda r: pct(r["citation_accuracy"])),
        row("False refusals (answerable, but model said not found)", lambda r: pct(r["false_refusal_rate"])),
        row("Correct refusals (unanswerable)", lambda r: pct(r["correct_refusal_rate"])),
        row("Mean latency", lambda r: f"{r['mean_latency_ms'] / 1000:.1f} s"),
        row("Cost per question", per_question),
    ]
    return "\n".join(lines)


def main() -> int:
    reports = []
    for title, rel in RUNS:
        path = ROOT / rel
        reports.append((title, json.loads(path.read_text(encoding="utf-8")) if path.exists() else None))
    if not any(r for _, r in reports):
        sys.exit("No answer reports found in reports/")
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        sys.exit(f"README.md has no {START} … {END} block")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    block = table(reports)
    readme.write_text(f"{head}{START}\n{block}\n{END}{tail}", encoding="utf-8")
    print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
