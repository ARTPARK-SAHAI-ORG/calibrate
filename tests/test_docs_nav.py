"""Tests for scripts/docs_nav.py (shared docs.json tab insertion)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from docs_nav import insert_tab  # noqa: E402


def _names(tabs):
    return [t["tab"] for t in tabs]


def test_insert_after_single_anchor() -> None:
    tabs = [{"tab": "Home"}, {"tab": "API reference"}]
    out = insert_tab(tabs, {"tab": "CLI"}, after="Home")
    assert _names(out) == ["Home", "CLI", "API reference"]


def test_insert_is_idempotent() -> None:
    tabs = [{"tab": "Home"}, {"tab": "CLI"}, {"tab": "API reference"}]
    out = insert_tab(tabs, {"tab": "CLI"}, after="Home")
    assert _names(out) == ["Home", "CLI", "API reference"]
    assert _names(out).count("CLI") == 1


def test_priority_list_uses_first_present_anchor() -> None:
    # "CLI" absent -> falls back to "Home"
    tabs = [{"tab": "Home"}, {"tab": "API reference"}]
    out = insert_tab(tabs, {"tab": "Python SDK"}, after=("CLI", "Home"))
    assert _names(out) == ["Home", "Python SDK", "API reference"]
    # "CLI" present -> wins over "Home"
    tabs = [{"tab": "Home"}, {"tab": "CLI"}, {"tab": "API reference"}]
    out = insert_tab(tabs, {"tab": "Python SDK"}, after=("CLI", "Home"))
    assert _names(out) == ["Home", "CLI", "Python SDK", "API reference"]


def test_appends_when_no_anchor_present() -> None:
    tabs = [{"tab": "Home"}, {"tab": "API reference"}]
    out = insert_tab(tabs, {"tab": "CLI"}, after="Nonexistent")
    assert _names(out) == ["Home", "API reference", "CLI"]


def test_remove_set_drops_legacy_names() -> None:
    tabs = [{"tab": "Home"}, {"tab": "SDK"}, {"tab": "Python SDK"}]
    out = insert_tab(
        tabs, {"tab": "Python SDK"}, after="Home", remove={"Python SDK", "SDK"}
    )
    assert _names(out) == ["Home", "Python SDK"]
    assert "SDK" not in _names(out)
