"""Small, dependency-free helpers shared by LLM metric aggregation and the
leaderboard. Kept separate from ``run_tests`` (heavy: pipecat) and
``tests_leaderboard`` (pandas) so both can import it without a cross-module
dependency in the wrong direction.
"""

import math
from typing import List, Optional


def _numeric_or_none(value: object) -> Optional[float]:
    """Return ``value`` if it is a real number, else ``None``.

    Booleans are excluded even though ``bool`` is a subclass of ``int`` — a cost
    or latency is never a true/false value, so ``True`` must not be read as
    ``1.0``.
    """
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) else None


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Linear-interpolated percentile over a pre-sorted list (numpy ``linear``).

    ``sorted_values`` must be non-empty and ascending. ``pct`` is 0–100.
    """
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_values[lo]
    frac = rank - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def _latency_percentiles(values: List[float]) -> Optional[dict]:
    """Aggregate raw latency/ttfb samples into ``{p50, p95, p99, count}``.

    Returns ``None`` for an empty input so callers can omit the block. Values
    are returned as-is (not rounded) — callers round to their unit.
    """
    if not values:
        return None
    ordered = sorted(values)
    return {
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "count": len(ordered),
    }
