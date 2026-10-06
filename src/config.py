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


def claude_model() -> str:
    """Model ID from the CLAUDE_MODEL environment variable, with a default alias."""
    return os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
