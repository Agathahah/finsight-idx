"""Thumbs up/down on RAG answers, joined to the query log by request id."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

# format_answer() ends every answer with: id `<12 hex chars>`
ANSWER_ID = re.compile(r"id `([0-9a-f]{12})`")
VOTES = {"up", "down"}
_lock = threading.Lock()


def record_feedback(answer_markdown: str, vote: str, path: str | Path | None = None) -> str:
    """Append {ts, request_id, vote} and return a short message for the UI."""
    if vote not in VOTES:
        raise ValueError(f"vote must be one of {sorted(VOTES)}")
    match = ANSWER_ID.search(answer_markdown or "")
    if not match:
        return "Belum ada jawaban untuk dinilai."
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request_id": match.group(1),
        "vote": vote,
    }
    target = Path(path or os.getenv("FEEDBACK_LOG_PATH", "logs/feedback.jsonl"))
    target.parent.mkdir(parents=True, exist_ok=True)
    with _lock, target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return "Terima kasih, penilaianmu tercatat."
