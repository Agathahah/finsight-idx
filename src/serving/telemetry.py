"""
Per-query telemetry: latency, tokens, estimated cost, pages cited.

One JSON line per request in ``logs/queries.jsonl`` (path: QUERY_LOG_PATH).
The question text is NOT stored by default: only its length and a short
hash, so the log can be shared without exposing what users asked. Set
LOG_QUESTION_TEXT=1 to store it (for your own debugging).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# USD per million tokens. Defaults are Claude Sonnet 4.5 list prices;
# override when you change CLAUDE_MODEL.
DEFAULT_INPUT_USD_PER_MTOK = 3.0
DEFAULT_OUTPUT_USD_PER_MTOK = 15.0

_lock = threading.Lock()


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    pin = float(os.getenv("CLAUDE_INPUT_USD_PER_MTOK", DEFAULT_INPUT_USD_PER_MTOK))
    pout = float(os.getenv("CLAUDE_OUTPUT_USD_PER_MTOK", DEFAULT_OUTPUT_USD_PER_MTOK))
    return round(input_tokens / 1e6 * pin + output_tokens / 1e6 * pout, 6)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def question_fingerprint(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode()).hexdigest()[:12]


def log_query(
    *,
    request_id: str,
    question: str,
    status: str,
    latency_ms: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    pages: Optional[list[int]] = None,
    retrieved_chunks: int = 0,
    error_type: Optional[str] = None,
    path: Optional[str | Path] = None,
) -> dict:
    """Append one record to the query log and return it."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request_id": request_id,
        "status": status,                      # ok | rejected | rate_limited | budget_exhausted | error | no_results
        "question_len": len(question),
        "question_hash": question_fingerprint(question),
        "latency_ms": round(latency_ms, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": estimate_cost_usd(input_tokens, output_tokens),
        "pages": pages or [],
        "retrieved_chunks": retrieved_chunks,
        "error_type": error_type,
    }
    if os.getenv("LOG_QUESTION_TEXT", "0") == "1":
        record["question"] = question
    target = Path(path or os.getenv("QUERY_LOG_PATH", "logs/queries.jsonl"))
    target.parent.mkdir(parents=True, exist_ok=True)
    with _lock, target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def summarize_log(path: str | Path = "logs/queries.jsonl") -> dict:
    """Aggregate the query log: volume, error rate, latency percentiles, cost."""
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return {"requests": 0}
    ok = sorted(r["latency_ms"] for r in rows if r["status"] == "ok")

    def pct(p: float) -> Optional[float]:
        if not ok:
            return None
        return ok[min(len(ok) - 1, int(round(p * (len(ok) - 1))))]

    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    return {
        "requests": len(rows),
        "by_status": by_status,
        "error_rate": round(by_status.get("error", 0) / len(rows), 4),
        "latency_ms_p50": pct(0.5),
        "latency_ms_p95": pct(0.95),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
        "avg_cost_usd_ok": round(sum(r["cost_usd"] for r in rows if r["status"] == "ok") / max(len(ok), 1), 5),
    }


if __name__ == "__main__":
    import sys
    print(json.dumps(summarize_log(sys.argv[1] if len(sys.argv) > 1 else "logs/queries.jsonl"), indent=2))
