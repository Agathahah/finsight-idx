"""
FinSight IDX — Claude API Client
=================================
Production-ready wrapper around the Anthropic Python SDK.

Provides:
- Singleton client instantiation from environment variables
- analyze_financial_text() for financial NLP tasks
- Retry logic with exponential backoff (via tenacity)
- Structured logging (via loguru)
- Full type hints and docstrings
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

import anthropic
from dotenv import load_dotenv
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging

# Load .env at module import time
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
DEFAULT_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))
DEFAULT_TEMPERATURE = float(os.getenv("CLAUDE_TEMPERATURE", "0.2"))

# Retry configuration
MAX_RETRY_ATTEMPTS = 3
RETRY_MIN_WAIT_SECONDS = 2
RETRY_MAX_WAIT_SECONDS = 30


# ---------------------------------------------------------------------------
# Task Definitions
# ---------------------------------------------------------------------------

class FinancialTask(str, Enum):
    """Supported NLP analysis tasks for IDX financial documents."""

    SUMMARIZE = "summarize"
    SENTIMENT = "sentiment"
    KEY_METRICS = "key_metrics"
    RISK_FACTORS = "risk_factors"
    TOPIC_EXTRACT = "topic_extract"


# System prompts keyed by task
TASK_SYSTEM_PROMPTS: dict[FinancialTask, str] = {
    FinancialTask.SUMMARIZE: (
        "Anda adalah analis keuangan senior yang ahli menganalisis laporan tahunan "
        "perusahaan-perusahaan yang terdaftar di Bursa Efek Indonesia (IDX). "
        "Berikan ringkasan yang terstruktur, akurat, dan mudah dipahami."
    ),
    FinancialTask.SENTIMENT: (
        "Anda adalah analis sentimen keuangan. Analisis teks berikut dan tentukan "
        "sentimen keseluruhan (Positif/Netral/Negatif) beserta skor kepercayaan (0–1) "
        "dan alasan singkat. Respons dalam format JSON."
    ),
    FinancialTask.KEY_METRICS: (
        "Anda adalah akuntan publik bersertifikat yang mengekstraksi metrik keuangan utama "
        "dari laporan perusahaan IDX. Ekstraksi nilai numerik, satuan, dan periode pelaporan. "
        "Respons dalam format JSON terstruktur."
    ),
    FinancialTask.RISK_FACTORS: (
        "Anda adalah analis risiko keuangan. Identifikasi dan kategorikan faktor-faktor risiko "
        "utama yang disebutkan dalam teks laporan keuangan IDX. "
        "Kelompokkan berdasarkan: risiko pasar, risiko operasional, risiko regulasi, risiko lainnya."
    ),
    FinancialTask.TOPIC_EXTRACT: (
        "Anda adalah peneliti NLP yang mengekstraksi topik-topik utama dari teks keuangan Indonesia. "
        "Identifikasi 3–5 topik dominan beserta kata kunci pendukungnya. "
        "Respons dalam format JSON."
    ),
}

TASK_USER_INSTRUCTIONS: dict[FinancialTask, str] = {
    FinancialTask.SUMMARIZE: (
        "Buat ringkasan terstruktur dari teks laporan keuangan berikut. "
        "Sertakan: (1) poin-poin kinerja utama, (2) pencapaian signifikan, "
        "(3) tantangan yang dihadapi.\n\nTeks:\n{text}"
    ),
    FinancialTask.SENTIMENT: (
        "Analisis sentimen dari teks berikut:\n\n{text}"
    ),
    FinancialTask.KEY_METRICS: (
        "Ekstraksi semua metrik keuangan kuantitatif dari teks berikut:\n\n{text}"
    ),
    FinancialTask.RISK_FACTORS: (
        "Identifikasi faktor risiko dalam teks laporan berikut:\n\n{text}"
    ),
    FinancialTask.TOPIC_EXTRACT: (
        "Ekstraksi topik-topik utama dari teks keuangan berikut:\n\n{text}"
    ),
}


# ---------------------------------------------------------------------------
# Client Factory
# ---------------------------------------------------------------------------

def _build_client() -> anthropic.Anthropic:
    """
    Instantiate the Anthropic client from environment variables.

    Returns:
        anthropic.Anthropic: Configured SDK client.

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
    logger.debug("Anthropic client initialized | model={}", DEFAULT_MODEL)
    return anthropic.Anthropic(api_key=api_key)


# Module-level singleton — created once on first import
_client: Optional[anthropic.Anthropic] = None


def get_client() -> anthropic.Anthropic:
    """
    Return the module-level Anthropic client singleton.

    Lazy-initializes on first call so that unit tests can patch
    the environment before the client is created.

    Returns:
        anthropic.Anthropic: Shared client instance.
    """
    global _client
    if _client is None:
        _client = _build_client()
    return _client


# ---------------------------------------------------------------------------
# Core Analysis Function
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(
        (anthropic.RateLimitError, anthropic.InternalServerError)
    ),
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    wait=wait_exponential(
        multiplier=1,
        min=RETRY_MIN_WAIT_SECONDS,
        max=RETRY_MAX_WAIT_SECONDS,
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),  # type: ignore[arg-type]
    reraise=True,
)
def analyze_financial_text(
    text: str,
    task: FinancialTask = FinancialTask.SUMMARIZE,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """
    Send a financial text excerpt to Claude for NLP analysis.

    Supports multiple task types (summarization, sentiment, metric extraction,
    risk identification, topic extraction). Automatically retries on rate-limit
    and server errors with exponential backoff.

    Args:
        text:        The financial text to analyze (e.g., excerpt from IDX report).
        task:        Analysis task type. Defaults to FinancialTask.SUMMARIZE.
        model:       Claude model ID. Defaults to CLAUDE_MODEL env var.
        max_tokens:  Maximum tokens in the response. Defaults to CLAUDE_MAX_TOKENS.
        temperature: Sampling temperature (0.0–1.0). Lower = more deterministic.

    Returns:
        dict with keys:
            - "task"      (str):   Task name that was executed.
            - "model"     (str):   Model used.
            - "result"    (str):   Claude's response text.
            - "usage"     (dict):  Token usage (input_tokens, output_tokens).

    Raises:
        ValueError:                    If text is empty or task is invalid.
        anthropic.AuthenticationError: If the API key is invalid.
        anthropic.RateLimitError:      After max retries are exhausted.
        anthropic.APIError:            For other unrecoverable API errors.
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string.")

    system_prompt = TASK_SYSTEM_PROMPTS[task]
    user_message = TASK_USER_INSTRUCTIONS[task].format(text=text.strip())

    logger.info(
        "Sending request | task={} | model={} | input_chars={}",
        task.value,
        model,
        len(text),
    )

    try:
        response = get_client().messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.AuthenticationError as exc:
        logger.error("Authentication failed — check ANTHROPIC_API_KEY: {}", exc)
        raise
    except anthropic.BadRequestError as exc:
        logger.error("Bad request (prompt may be too long or invalid): {}", exc)
        raise
    except anthropic.RateLimitError as exc:
        logger.warning("Rate limit hit — will retry: {}", exc)
        raise  # tenacity handles retry
    except anthropic.InternalServerError as exc:
        logger.warning("Anthropic server error — will retry: {}", exc)
        raise  # tenacity handles retry
    except anthropic.APIError as exc:
        logger.error("Unhandled API error: {}", exc)
        raise

    result_text = response.content[0].text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    logger.success(
        "Response received | task={} | input_tokens={} | output_tokens={}",
        task.value,
        usage["input_tokens"],
        usage["output_tokens"],
    )

    return {
        "task": task.value,
        "model": model,
        "result": result_text,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Demo / Manual Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sample excerpt from a real IDX annual report (PT Bank Central Asia Tbk, 2023)
    SAMPLE_IDX_TEXT = """
    Pada tahun 2023, BCA membukukan laba bersih sebesar Rp 48,6 triliun, meningkat 19,5%
    dibandingkan tahun sebelumnya. Total aset BCA tumbuh 8,7% menjadi Rp 1.408 triliun.
    Kredit yang disalurkan meningkat 13,8% menjadi Rp 793,2 triliun, didorong oleh
    pertumbuhan kredit korporasi dan UKM. Rasio kecukupan modal (CAR) BCA tercatat sebesar
    25,9%, jauh di atas ketentuan minimum regulator. Dana Pihak Ketiga (DPK) tumbuh 7,6%
    menjadi Rp 1.063 triliun, dengan CASA ratio yang tetap kuat di level 81,7%.
    Tantangan utama yang dihadapi perseroan meliputi tekanan suku bunga global, potensi
    pelemahan daya beli konsumen, serta meningkatnya persaingan dari bank digital.
    """

    print("=" * 60)
    print("FinSight IDX — API Client Demo")
    print("=" * 60)

    for task in [FinancialTask.SUMMARIZE, FinancialTask.SENTIMENT, FinancialTask.KEY_METRICS]:
        print(f"\n📌 Task: {task.value.upper()}")
        print("-" * 40)
        output = analyze_financial_text(SAMPLE_IDX_TEXT, task=task)
        print(output["result"])
        print(f"\n[Tokens used: {output['usage']['input_tokens']} in / {output['usage']['output_tokens']} out]")
