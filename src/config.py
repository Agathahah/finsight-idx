"""Single place for runtime settings.

Model IDs change and old snapshots are retired; the first version hard-coded
``claude-sonnet-4-20250514`` in nine places and returned 404 once that ID was
no longer available to the account. Set ``CLAUDE_MODEL`` in ``.env`` and run
``python scripts/check_model.py`` to confirm the ID is available.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5"

# Retrieval settings that won the golden-set evaluation (reports/answer_eval_v4_k10.json:
# answers correct 79% with BM25 top-10 vs 65% with hybrid top-5). Overridable from .env.
DEFAULT_RETRIEVAL_MODE = "bm25"
DEFAULT_TOP_K = 10


def retrieval_settings() -> tuple[str, int]:
    """(mode, top_k) from RETRIEVAL_MODE / RAG_TOP_K, falling back to the evaluated defaults."""
    return (os.getenv("RETRIEVAL_MODE", DEFAULT_RETRIEVAL_MODE).lower(),
            int(os.getenv("RAG_TOP_K", str(DEFAULT_TOP_K))))


def claude_model() -> str:
    """Model ID from the CLAUDE_MODEL environment variable, with a default alias."""
    return os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
