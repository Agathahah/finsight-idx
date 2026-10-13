"""Input guard, rate limiter and query telemetry."""

import json

import pytest

from src.serving import guard, telemetry


def test_question_is_cleaned():
    assert guard.validate_question("  Berapa\nlaba\x00 bersih?  ") == "Berapa laba bersih?"


@pytest.mark.parametrize("bad", ["", "   ", "hai", None, 123, "x" * 501])
def test_bad_questions_rejected_with_user_message(bad):
    with pytest.raises(guard.InvalidQuestion) as e:
        guard.validate_question(bad)
    assert str(e.value)


def test_year_and_ticker_validation():
    assert guard.validate_year("2023") == 2023 and guard.validate_year("") is None
    assert guard.validate_ticker(" bbca ") == "BBCA" and guard.validate_ticker(None) is None
    for bad in ("20x3", "1990"):
        with pytest.raises(guard.InvalidQuestion):
            guard.validate_year(bad)
    with pytest.raises(guard.InvalidQuestion):
        guard.validate_ticker("BBCA; DROP")


def test_rate_limiter_per_key_and_window():
    now = [0.0]
    rl = guard.RateLimiter(max_calls=2, per_seconds=60, global_max_calls=100, clock=lambda: now[0])
    assert rl.allow("a")[0] and rl.allow("a")[0]
    ok, wait = rl.allow("a")
    assert not ok and wait == pytest.approx(60)
    assert rl.allow("b")[0]            # other visitors unaffected
    now[0] = 61
    assert rl.allow("a")[0]            # window slid


def test_rate_limiter_global_budget():
    now = [0.0]
    rl = guard.RateLimiter(max_calls=10, per_seconds=60, global_max_calls=3, clock=lambda: now[0])
    assert all(rl.allow(k)[0] for k in "abc")
    assert not rl.allow("d")[0]


def test_query_log_hides_question_text_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("LOG_QUESTION_TEXT", raising=False)
    path = tmp_path / "q.jsonl"
    rec = telemetry.log_query(request_id="r1", question="Berapa laba bersih?", status="ok",
                              latency_ms=1234.5, input_tokens=2000, output_tokens=300,
                              pages=[16], retrieved_chunks=5, path=path)
    assert "question" not in rec and rec["question_len"] == 19
    assert rec["cost_usd"] == pytest.approx(2000 * 3 / 1e6 + 300 * 15 / 1e6)
    assert json.loads(path.read_text())["request_id"] == "r1"


def test_log_summary(tmp_path):
    path = tmp_path / "q.jsonl"
    for i, st in enumerate(["ok", "ok", "ok", "error", "rate_limited"]):
        telemetry.log_query(request_id=str(i), question="q?", status=st, latency_ms=100 * (i + 1),
                            input_tokens=1000 if st == "ok" else 0, path=path)
    s = telemetry.summarize_log(path)
    assert s["requests"] == 5 and s["by_status"]["ok"] == 3
    assert s["error_rate"] == 0.2
    assert s["latency_ms_p50"] == 200 and s["latency_ms_p95"] == 300
