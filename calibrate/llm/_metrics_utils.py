"""Small, dependency-free helpers shared by LLM metric aggregation and the
leaderboard. Kept separate from ``run_tests`` (heavy: pipecat) and
``tests_leaderboard`` (pandas) so both can import it without a cross-module
dependency in the wrong direction.
"""

from typing import Optional


def _numeric_or_none(value: object) -> Optional[float]:
    """Return ``value`` if it is a real number, else ``None``.

    Booleans are excluded even though ``bool`` is a subclass of ``int`` — a cost
    or latency is never a true/false value, so ``True`` must not be read as
    ``1.0``.
    """
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) else None
