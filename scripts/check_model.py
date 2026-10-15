"""Confirm that CLAUDE_MODEL is available to your API key before running the pipeline.

    python scripts/check_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402

from src.config import claude_model  # noqa: E402


def main() -> int:
    client = anthropic.Anthropic()
    wanted = claude_model()
    available = [m.id for m in client.models.list(limit=100)]
    print("Configured CLAUDE_MODEL:", wanted)
    if wanted in available or any(m.startswith(wanted) for m in available):
        print("OK: model is available.")
        return 0
    print("NOT AVAILABLE. Models visible to this key:")
    for m in available:
        print("  ", m)
    print("Set CLAUDE_MODEL in .env to one of the IDs above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
