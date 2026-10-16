"""
Input validation and rate limiting for the public demo.

The Space calls a paid API on every question, so two things must hold:
a request is a real question of reasonable size, and one visitor cannot
spend the whole budget.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections import defaultdict, deque

MIN_QUESTION_CHARS = 5
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "500"))
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class InvalidQuestion(ValueError):
    """Raised with a message that is safe to show to the user."""


def validate_question(question: object) -> str:
    """Return a cleaned question or raise InvalidQuestion."""
    if not isinstance(question, str):
        raise InvalidQuestion("Pertanyaan harus berupa teks.")
    q = CONTROL_CHARS.sub(" ", question)
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) < MIN_QUESTION_CHARS:
        raise InvalidQuestion(f"Pertanyaan terlalu pendek (minimal {MIN_QUESTION_CHARS} karakter).")
    if len(q) > MAX_QUESTION_CHARS:
        raise InvalidQuestion(
            f"Pertanyaan terlalu panjang ({len(q)} karakter, maksimal {MAX_QUESTION_CHARS})."
        )
    return q


def validate_year(year: object) -> int | None:
    if year in (None, ""):
        return None
    try:
        y = int(year)
    except (TypeError, ValueError):
        raise InvalidQuestion("Tahun harus berupa angka, misalnya 2023.")
    if not 2000 <= y <= 2100:
        raise InvalidQuestion("Tahun di luar rentang yang didukung (2000–2100).")
    return y


def validate_ticker(ticker: object) -> str | None:
    if ticker in (None, ""):
        return None
    t = str(ticker).strip().upper()
    if not re.fullmatch(r"[A-Z]{4}", t):
        raise InvalidQuestion("Kode emiten terdiri dari 4 huruf, misalnya BBCA.")
    return t


class RateLimiter:
    """Sliding-window limiter: at most ``max_calls`` per ``per_seconds`` per key,
    plus a global ceiling for all keys together (protects the API budget)."""

    def __init__(self, max_calls: int | None = None, per_seconds: float = 60.0,
                 global_max_calls: int | None = None, clock=time.monotonic) -> None:
        self.max_calls = max_calls or int(os.getenv("RATE_LIMIT_PER_MINUTE", "5"))
        self.global_max = global_max_calls or int(os.getenv("RATE_LIMIT_GLOBAL_PER_MINUTE", "60"))
        self.per_seconds = per_seconds
        self._clock = clock
        self._calls: dict[str, deque] = defaultdict(deque)
        self._all: deque = deque()
        self._lock = threading.Lock()

    def _trim(self, q: deque, now: float) -> None:
        while q and now - q[0] >= self.per_seconds:
            q.popleft()

    def allow(self, key: str) -> tuple[bool, float]:
        """Return (allowed, seconds_until_next_slot)."""
        with self._lock:
            now = self._clock()
            q = self._calls[key]
            self._trim(q, now)
            self._trim(self._all, now)
            if len(q) >= self.max_calls:
                return False, self.per_seconds - (now - q[0])
            if len(self._all) >= self.global_max:
                return False, self.per_seconds - (now - self._all[0])
            q.append(now)
            self._all.append(now)
            return True, 0.0
