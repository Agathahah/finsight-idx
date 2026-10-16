"""The demo's request path, tested without Gradio or the Claude API."""

import json
from types import SimpleNamespace

from src.serving.guard import RateLimiter
from src.serving.qa_service import QAService, client_key_from_request, trim_agent_history


def _resp(chunks=5, ocr=False):
    src = SimpleNamespace(page_number=16, citation="[BBCA, 2023, hal. 16" + (", OCR]" if ocr else "]"),
                          text_preview="Laba tahun berjalan ...")
    return SimpleNamespace(answer="Rp48.639 miliar [Sumber: BBCA 2023, hal. 16]", sources=[src, src],
                           input_tokens=2000, output_tokens=300, latency_ms=1500.0,
                           retrieved_chunks=chunks)


class FakeChain:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp or _resp(), exc, []

    def ask(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return self.resp


def _service(tmp_path, chain=None, limiter=None, key=True):
    chain = chain or FakeChain()
    svc = QAService(persist_dir=str(tmp_path), chain_factory=lambda: chain,
                    limiter=limiter or RateLimiter(max_calls=100, global_max_calls=100),
                    log_path=str(tmp_path / "q.jsonl"), api_key_present=lambda: key)
    return svc, chain


def _log(tmp_path):
    return [json.loads(l) for l in (tmp_path / "q.jsonl").read_text().splitlines()]


def test_happy_path_formats_answer_and_logs(tmp_path):
    svc, chain = _service(tmp_path)
    out = svc.ask("  Berapa laba bersih BCA 2023? ", "bbca", "2023", "1.2.3.4")
    assert chain.calls == [{"question": "Berapa laba bersih BCA 2023?", "company_filter": "BBCA",
                            "year_filter": 2023}]
    assert "Rp48.639" in out and "hal. 16" in out and "id `" in out
    assert out.count("[BBCA, 2023, hal. 16]") == 1          # duplicate pages collapsed
    rec = _log(tmp_path)[0]
    assert rec["status"] == "ok" and rec["pages"] == [16] and rec["cost_usd"] > 0


def test_invalid_input_never_reaches_the_api(tmp_path):
    svc, chain = _service(tmp_path)
    assert svc.ask("hi", None, None).startswith("⚠️")
    assert svc.ask("Berapa laba?", "BBCA", "dua ribu").startswith("⚠️")
    assert chain.calls == []
    assert [r["status"] for r in _log(tmp_path)] == ["rejected", "rejected"]


def test_rate_limit(tmp_path):
    svc, chain = _service(tmp_path, limiter=RateLimiter(max_calls=1, global_max_calls=10))
    svc.ask("Berapa laba bersih?", None, None, "a")
    assert svc.ask("Berapa laba bersih?", None, None, "a").startswith("⏳")
    assert len(chain.calls) == 1


def test_errors_are_hidden_from_users_but_logged(tmp_path):
    svc, _ = _service(tmp_path, chain=FakeChain(exc=RuntimeError("secret stack detail")))
    out = svc.ask("Berapa laba bersih?", None, None)
    assert "secret" not in out and "id `" in out
    assert _log(tmp_path)[0]["error_type"] == "RuntimeError"


def test_broken_index_fails_fast_on_every_request(tmp_path):
    from src.rag.manifest import IndexVersionError
    built = []

    def factory():
        built.append(1)
        raise IndexVersionError("old index")
    svc = QAService(persist_dir=str(tmp_path), chain_factory=factory, log_path=str(tmp_path / "q.jsonl"),
                    limiter=RateLimiter(max_calls=10, global_max_calls=10), api_key_present=lambda: True)
    for _ in range(2):
        assert "Index dokumen belum siap" in svc.ask("Berapa laba bersih?", None, None)
    assert built == [1]


def test_missing_api_key(tmp_path):
    svc, chain = _service(tmp_path, key=False)
    assert "API key" in svc.ask("Berapa laba bersih?", None, None) and chain.calls == []


def test_ocr_sources_get_a_warning(tmp_path):
    svc, _ = _service(tmp_path, chain=FakeChain(resp=_resp(ocr=True)))
    assert "OCR" in svc.ask("Siapa auditor BCA?", None, None)


def test_status_reads_manifest(tmp_path):
    from src.rag.manifest import record_document
    svc, _ = _service(tmp_path)
    assert "belum ada manifest" in svc.status()
    record_document(tmp_path, "d", {"company": "BBCA", "year": 2023, "total_chunks": 10953,
                                    "total_pages": 747, "pages_ocr": [570, 571]},
                    embed_model="m", chunk_size=256, chunk_overlap=64)
    assert "BBCA 2023: 10,953 chunk dari 747 halaman (OCR 2 hlm)" in svc.status()


def test_client_key_prefers_forwarded_header():
    req = SimpleNamespace(headers={"x-forwarded-for": "9.9.9.9, 10.0.0.1"},
                          client=SimpleNamespace(host="10.0.0.1"))
    assert client_key_from_request(req) == "9.9.9.9"
    assert client_key_from_request(SimpleNamespace(headers={}, client=SimpleNamespace(host="1.1.1.1"))) == "1.1.1.1"
    assert client_key_from_request(None) == "anonymous"


def test_agent_history_is_cut_only_at_plain_user_turns():
    msgs = []
    for i in range(6):
        msgs += [{"role": "user", "content": f"q{i}"},
                 {"role": "assistant", "content": [{"type": "tool_use"}]},
                 {"role": "user", "content": [{"type": "tool_result"}]},
                 {"role": "assistant", "content": "a"}]
    out = trim_agent_history(msgs, max_messages=10)
    assert len(out) <= 10 and out[0] == {"role": "user", "content": "q4"}
    assert trim_agent_history(msgs[:4], max_messages=10) == msgs[:4]


def test_daily_budget_blocks_api_calls_once_spent(tmp_path):
    from src.serving.guard import DailyBudget
    chain = FakeChain()
    svc = QAService(persist_dir=str(tmp_path), chain_factory=lambda: chain,
                    limiter=RateLimiter(max_calls=100, global_max_calls=100),
                    log_path=str(tmp_path / "q.jsonl"), api_key_present=lambda: True,
                    budget=DailyBudget(limit_usd=0.01))
    first = svc.ask("Berapa laba bersih BCA 2023?")       # 2000 in / 300 out ≈ $0.0105
    second = svc.ask("Berapa total aset BCA 2023?")
    assert "## Jawaban" in first and second.startswith("⏳ Kuota harian")
    assert len(chain.calls) == 1
    assert [r["status"] for r in _log(tmp_path)] == ["ok", "budget_exhausted"]
