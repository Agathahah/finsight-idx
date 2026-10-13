"""Retrieval evaluation against a hand-labelled golden set.

Golden set format (JSON Lines), one question per line:
    {"question": "Berapa laba bersih BBCA 2023?", "company": "BBCA", "year": 2023,
     "relevant_pages": [112, 113]}

A retrieved chunk counts as relevant when its page is in ``relevant_pages``
(and company/year match when given). Run:
    python -m src.rag.eval_retrieval eval/golden_questions.jsonl --k 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_golden(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    # Questions the document cannot answer have no relevant page: they belong
    # to the answer-level eval (eval_answers.py), not to retrieval recall.
    rows = [r for r in rows if r.get("answerable", True)]
    for r in rows:
        if not r.get("question") or not r.get("relevant_pages"):
            raise ValueError(f"Each row needs 'question' and 'relevant_pages': {r}")
    return rows


def is_relevant(chunk, item: dict) -> bool:
    if item.get("company") and chunk.company != item["company"]:
        return False
    if item.get("year") and chunk.year != item["year"]:
        return False
    return chunk.page_number in set(item["relevant_pages"])


def recall_at_k(flags: list[bool], k: int) -> float:
    """1.0 if any of the first k results is relevant (hit rate)."""
    return float(any(flags[:k]))


def reciprocal_rank(flags: list[bool]) -> float:
    for i, ok in enumerate(flags, start=1):
        if ok:
            return 1.0 / i
    return 0.0


def evaluate(retrieve_fn, golden: list[dict], k: int = 5) -> dict:
    """``retrieve_fn(question, item) -> list[RetrievedChunk]``"""
    per_q = []
    for item in golden:
        chunks = retrieve_fn(item["question"], item)
        flags = [is_relevant(c, item) for c in chunks]
        per_q.append({"question": item["question"], "recall_at_k": recall_at_k(flags, k),
                      "rr": reciprocal_rank(flags)})
    n = max(len(per_q), 1)
    return {
        "questions": len(per_q),
        "k": k,
        "recall_at_k": round(sum(q["recall_at_k"] for q in per_q) / n, 4),
        "mrr": round(sum(q["rr"] for q in per_q) / n, 4),
        "misses": [q["question"] for q in per_q if q["recall_at_k"] == 0],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--persist-dir", default="data/vectorstore")
    args = ap.parse_args()

    from src.rag.retriever import RAGRetriever

    retriever = RAGRetriever(persist_dir=args.persist_dir)
    golden = load_golden(args.golden)
    hybrid = evaluate(lambda q, it: retriever.retrieve(
        q, top_k=args.k, company_filter=it.get("company"), year_filter=it.get("year")), golden, args.k)
    semantic = evaluate(lambda q, it: retriever.retrieve_semantic_only(
        q, top_k=args.k, where_filter=retriever._build_filter(it.get("company"), it.get("year"), None)),
        golden, args.k)
    print(json.dumps({"hybrid": hybrid, "semantic_only": semantic}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
