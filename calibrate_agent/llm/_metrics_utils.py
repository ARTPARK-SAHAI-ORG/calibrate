"""Small, dependency-free helpers shared by LLM metric aggregation and the
leaderboard. Kept separate from ``run_tests`` (heavy: pipecat) and
``tests_leaderboard`` (pandas) so both can import it without a cross-module
dependency in the wrong direction.
"""

from typing import List, Optional

import numpy as np


def _numeric_or_none(value: object) -> Optional[float]:
    """Return ``value`` if it is a real number, else ``None``.

    Booleans are excluded even though ``bool`` is a subclass of ``int`` — a cost
    or latency is never a true/false value, so ``True`` must not be read as
    ``1.0``.
    """
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) else None


def _latency_percentiles(values: List[float]) -> Optional[dict]:
    """Aggregate raw latency/ttfb samples into ``{p50, p95, p99, count}``.

    Returns ``None`` for an empty input so callers can omit the block. Values
    are cast to plain ``float`` (``np.percentile`` returns ``np.float64``, which
    isn't JSON-serializable) and otherwise left unrounded — callers round to
    their unit.
    """
    if not values:
        return None
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return {
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "count": len(values),
    }
