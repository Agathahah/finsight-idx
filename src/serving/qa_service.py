"""
The request path of the demo, separated from Gradio so it can be tested.

Every question goes through the same steps:
    validate input -> rate limit -> (cached) QA chain -> format -> log
and every outcome, including rejections and errors, leaves one line in the
query log. Users see a short message and a request id; details stay in the log.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

from loguru import logger

from src.serving.guard import (
    InvalidQuestion, RateLimiter, validate_question, validate_ticker, validate_year,
)
from src.serving.telemetry import estimate_cost_usd, log_query, new_request_id

DISCLAIMER = "*Bukan rekomendasi investasi. Periksa angka di halaman sumber sebelum dipakai.*"
MAX_AGENT_MESSAGES = int(os.getenv("MAX_AGENT_MESSAGES", "20"))


def client_key_from_request(request) -> str:
    """Best-effort visitor key: proxy header, then socket address, then session."""
    if request is None:
        return "anonymous"
    headers = getattr(request, "headers", None) or {}
    fwd = headers.get("x-forwarded-for", "") if hasattr(headers, "get") else ""
    if fwd:
        return fwd.split(",")[0].strip()
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    return host or getattr(request, "session_hash", None) or "anonymous"


def format_answer(resp, request_id: str) -> str:
    lines = [f"## Jawaban\n{resp.answer}\n"]
    if resp.sources:
        lines.append("## Sumber")
        seen = set()
        for src in resp.sources:
            if src.page_number in seen:
                continue
            seen.add(src.page_number)
            lines.append(f"- **{src.citation}**: _{src.text_preview[:120].strip()}…_")
            if len(seen) == 3:
                break
        if any("OCR" in s.citation for s in resp.sources):
            lines.append("\n> Sebagian sumber berasal dari halaman pindaian (OCR); periksa angkanya di PDF.")
    cost = estimate_cost_usd(resp.input_tokens, resp.output_tokens)
    lines.append(
        f"\n---\n{resp.input_tokens} token masuk / {resp.output_tokens} token keluar · "
        f"{resp.latency_ms / 1000:.1f} dtk · ≈ ${cost:.4f} · id `{request_id}`\n\n{DISCLAIMER}"
    )
    return "\n".join(lines)


class QAService:
    def __init__(
        self,
        persist_dir: str = "data/vectorstore",
        chain_factory: Optional[Callable[[], object]] = None,
        limiter: Optional[RateLimiter] = None,
        log_path: Optional[str] = None,
        api_key_present: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.persist_dir = persist_dir
        self._chain_factory = chain_factory or self._default_chain
        self.limiter = limiter or RateLimiter()
        self.log_path = log_path
        self._api_key_present = api_key_present or (
            lambda: os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-ant-"))
        self._chain = None
        self._chain_error: Optional[Exception] = None
        self._lock = threading.Lock()

    def _default_chain(self):
        from src.rag.qa_chain import FinancialQAChain
        return FinancialQAChain(persist_dir=self.persist_dir)

    def chain(self):
        """Build the QA chain once per process (loads the embedding model and BM25 corpus)."""
        with self._lock:
            if self._chain is None and self._chain_error is None:
                try:
                    self._chain = self._chain_factory()
                except Exception as exc:          # remembered so every request fails fast
                    self._chain_error = exc
                    logger.exception("QA chain failed to start")
            if self._chain_error is not None:
                raise self._chain_error
            return self._chain

    def warmup(self) -> None:
        t0 = time.perf_counter()
        try:
            self.chain()
            logger.info("QA chain ready in {:.1f}s", time.perf_counter() - t0)
        except Exception:
            pass  # already logged; requests will report it

    def status(self) -> str:
        from src.rag.manifest import load_manifest
        m = load_manifest(self.persist_dir)
        if not m:
            return "Index: belum ada manifest. Jalankan re-index sebelum deploy."
        parts = []
        for d in m.get("documents", {}).values():
            ocr = len(d.get("pages_ocr", []))
            parts.append(
                f"{d.get('company')} {d.get('year')}: {d.get('total_chunks'):,} chunk dari "
                f"{d.get('total_pages')} halaman (OCR {ocr} hlm), dibangun {d.get('indexed_at', '')[:10]}"
            )
        return "Index v{} · {} · {}".format(m.get("schema_version"), m.get("embed_model"), "; ".join(parts))

    def ask(self, question, ticker=None, year=None, client_key: str = "anonymous") -> str:
        rid = new_request_id()
        t0 = time.perf_counter()
        q_text = question if isinstance(question, str) else ""

        def log(status, **kw):
            log_query(request_id=rid, question=q_text, status=status,
                      latency_ms=(time.perf_counter() - t0) * 1000, path=self.log_path, **kw)

        try:
            q = validate_question(question)
            t = validate_ticker(ticker)
            y = validate_year(year)
        except InvalidQuestion as exc:
            log("rejected", error_type="invalid_input")
            return f"⚠️ {exc}"

        allowed, wait = self.limiter.allow(client_key)
        if not allowed:
            log("rate_limited")
            return f"⏳ Terlalu banyak pertanyaan. Coba lagi dalam {int(wait) + 1} detik."

        if not self._api_key_present():
            log("error", error_type="missing_api_key")
            return "⚠️ Layanan belum dikonfigurasi (API key tidak ditemukan)."

        try:
            resp = self.chain().ask(question=q, company_filter=t, year_filter=y)
        except Exception as exc:
            log("error", error_type=type(exc).__name__)
            logger.exception("request {} failed", rid)
            if type(exc).__name__ == "IndexVersionError":
                return f"⚠️ Index dokumen belum siap untuk versi aplikasi ini. (id `{rid}`)"
            return f"❌ Terjadi kesalahan saat memproses pertanyaan. Coba lagi nanti. (id `{rid}`)"

        status = "ok" if resp.retrieved_chunks else "no_results"
        log(status, input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            pages=sorted({s.page_number for s in resp.sources}), retrieved_chunks=resp.retrieved_chunks)
        return format_answer(resp, rid)


def trim_agent_history(messages: list[dict], max_messages: int = MAX_AGENT_MESSAGES) -> list[dict]:
    """Keep the most recent turns, cutting only at a plain user message.

    Cutting between a tool_use and its tool_result would make the next API
    call invalid, so the cut point must be a user turn whose content is text.
    """
    if len(messages) <= max_messages:
        return messages
    for i in range(len(messages) - max_messages, len(messages)):
        m = messages[i]
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return messages[i:]
    return []
