"""Shared helpers for editing the Mintlify ``docs.json`` navigation.

Both doc generators (``generate_cli_docs.py``, ``generate_sdk_docs.py``) need to
drop a tab into the sidebar idempotently and position it after a known anchor.
Keeping that in one place stops the two from drifting into inconsistent
ordering.
"""

from __future__ import annotations

from collections.abc import Iterable


def insert_tab(
    tabs: list[dict],
    tab: dict,
    *,
    after: str | Iterable[str],
    remove: Iterable[str] | None = None,
) -> list[dict]:
    """Return ``tabs`` with ``tab`` inserted after the first present anchor.

    - ``after`` is an anchor tab name, or a priority-ordered list of names; the
      new tab is placed right after the first one that exists, else appended.
    - ``remove`` names any existing tabs to drop first (defaults to the new
      tab's own name), so re-runs replace rather than duplicate.
    """
    remove_names = set(remove) if remove is not None else {tab["tab"]}
    result = [t for t in tabs if t.get("tab") not in remove_names]

    anchors = [after] if isinstance(after, str) else list(after)
    insert_at = len(result)
    for name in anchors:
        idx = next((i for i, t in enumerate(result) if t.get("tab") == name), None)
        if idx is not None:
            insert_at = idx + 1
            break

    result.insert(insert_at, tab)
    return result
