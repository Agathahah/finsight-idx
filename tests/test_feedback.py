import json

import pytest

from src.serving.feedback import record_feedback
from src.serving.telemetry_report import report

ANSWER = "Laba bersih naik 12% [hal. 45]\n\n1.2 dtk · ≈ $0.0031 · id `a1b2c3d4e5f6`\n\nBukan saran investasi."


def test_records_vote(tmp_path):
    p = tmp_path / "fb.jsonl"
    assert "tercatat" in record_feedback(ANSWER, "up", path=p)
    assert json.loads(p.read_text())["request_id"] == "a1b2c3d4e5f6"


def test_no_answer_yet(tmp_path):
    p = tmp_path / "fb.jsonl"
    assert record_feedback("*Ketik pertanyaan...*", "down", path=p).startswith("Belum")
    assert not p.exists()


def test_rejects_unknown_vote(tmp_path):
    with pytest.raises(ValueError):
        record_feedback(ANSWER, "meh", path=tmp_path / "x.jsonl")


def test_report_joins_feedback(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    q = [
        {"ts": "2026-12-01T03:00:00+00:00", "request_id": "a1b2c3d4e5f6", "status": "ok", "latency_ms": 1200, "cost_usd": 0.003},
        {"ts": "2026-12-01T04:00:00+00:00", "request_id": "bbbbbbbbbbbb", "status": "ok", "latency_ms": 2400, "cost_usd": 0.004},
        {"ts": "2026-12-02T04:00:00+00:00", "request_id": "cccccccccccc", "status": "error", "latency_ms": 50, "cost_usd": 0.0},
    ]
    (logs / "queries-run1.jsonl").write_text("\n".join(json.dumps(r) for r in q) + "\n")
    record_feedback(ANSWER, "down", path=logs / "feedback-run1.jsonl")
    record_feedback(ANSWER, "up", path=logs / "feedback-run1.jsonl")  # changed mind: last vote counts
    out = report(tmp_path)
    assert out["answered"] == 2 and out["rated_answers"] == 1
    assert out["thumbs_up_share"] == 1.0 and out["error_rate"] == round(1 / 3, 4)
