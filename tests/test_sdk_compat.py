"""Every Claude API call must match the signature of the installed anthropic SDK.

The other tests mock the client, and a mock accepts any keyword. That is how
`temperature=...` survived until anthropic 1.13 removed it and every real
request failed with TypeError. These tests read the real signature instead.
"""

import ast
import inspect
from pathlib import Path

from anthropic.resources.messages import Messages

from src.api import client

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = set(inspect.signature(Messages.create).parameters) - {"self"}


def _create_calls():
    for path in [*ROOT.joinpath("src").rglob("*.py"), ROOT / "app.py", *ROOT.joinpath("scripts").glob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "create"
                    and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "messages"):
                yield path.relative_to(ROOT), node


def test_api_calls_use_only_parameters_the_sdk_accepts():
    calls = list(_create_calls())
    assert len(calls) >= 5
    bad = [(str(p), n.lineno, kw.arg) for p, n in calls for kw in n.keywords
           if kw.arg is not None and kw.arg not in ALLOWED]
    assert bad == [], f"keywords not accepted by anthropic {client.anthropic.__version__}: {bad}"


def test_sampling_kwargs_follow_the_sdk():
    assert client.SUPPORTS_TEMPERATURE == ("temperature" in ALLOWED)
    assert client.sampling_kwargs(None) == {}
    if client.SUPPORTS_TEMPERATURE:
        assert client.sampling_kwargs(0.1) == {"temperature": 0.1}
    else:
        assert client.sampling_kwargs(0.1) == {}
