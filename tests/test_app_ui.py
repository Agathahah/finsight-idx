"""Guards for the Gradio UI that unit tests of the request path cannot see."""

import ast
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_app_does_not_use_gr_examples():
    """gr.Examples froze tab switching on the deployed Space (Gradio 6.12, 4 tabs):
    the browser loops on svelte "effect_update_depth_exceeded". Use example_buttons()."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "Examples"]
    assert calls == [], f"gr.Examples used at lines {[c.lineno for c in calls]}"


def test_app_has_no_load_event_updating_components():
    """demo.load(outputs=...) on a component inside a tab triggered the same
    endless re-render on Gradio 6.12, so the index status is rendered once."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "load" and isinstance(n.func.value, ast.Name) and n.func.value.id == "demo"]
    assert loads == [], f"demo.load used at lines {[c.lineno for c in loads]}"
