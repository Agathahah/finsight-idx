"""Weekly usage report from the telemetry dataset on the Hugging Face Hub.

python -m src.serving.telemetry_report Agatha1911/finsight-telemetry
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def read_jsonl(files: list[Path]) -> pd.DataFrame:
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def report(folder: Path) -> dict:
    queries = read_jsonl(sorted(folder.glob("logs/queries-*.jsonl")))
    feedback = read_jsonl(sorted(folder.glob("logs/feedback-*.jsonl")))
    if queries.empty:
        return {"questions": 0}
    queries["week"] = pd.to_datetime(queries["ts"]).dt.strftime("%G-W%V")
    ok = queries[queries["status"] == "ok"]
    out = {
        "questions": int(len(queries)),
        "answered": int(len(ok)),
        "per_week": queries.groupby("week").size().to_dict(),
        "latency_p50_ms": float(ok["latency_ms"].quantile(0.5)) if len(ok) else None,
        "latency_p95_ms": float(ok["latency_ms"].quantile(0.95)) if len(ok) else None,
        "error_rate": round(float((queries["status"] == "error").mean()), 4),
        "cost_usd": round(float(queries["cost_usd"].sum()), 4),
    }
    if not feedback.empty:
        last_vote = feedback.drop_duplicates("request_id", keep="last")
        rated = last_vote.merge(ok[["request_id"]], on="request_id")
        out["rated_answers"] = int(len(rated))
        out["feedback_rate"] = round(len(rated) / max(len(ok), 1), 4)
        out["thumbs_up_share"] = round(float((rated["vote"] == "up").mean()), 4) if len(rated) else None
    return out


if __name__ == "__main__":
    # imported here: only needed when reading from the Hub
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(repo_id=sys.argv[1], repo_type="dataset"))
    print(json.dumps(report(local), indent=2))
