"""
Unit tests for the ConversationState class in calibrate_agent.llm.run_simulation.

All tests are pure async — no mocks or external dependencies needed.
"""

import asyncio

from calibrate_agent.llm.run_simulation import ConversationState


# ---------------------------------------------------------------------------
# 1. record_turn increments turn_count
# ---------------------------------------------------------------------------


def test_record_turn_increments_turn_count():
    state = ConversationState(max_turns=5)
    asyncio.run(state.record_turn())
    assert state.turn_count == 1
    asyncio.run(state.record_turn())
    assert state.turn_count == 2


# ---------------------------------------------------------------------------
# 2. record_turn returns True while under max_turns
# ---------------------------------------------------------------------------


def test_record_turn_returns_true_while_under_max_turns():
    state = ConversationState(max_turns=3)
    result = asyncio.run(state.record_turn())
    assert result is True
    result = asyncio.run(state.record_turn())
    assert result is True


# ---------------------------------------------------------------------------
# 3. record_turn returns False when turn_count >= max_turns
# ---------------------------------------------------------------------------


def test_record_turn_returns_false_at_max_turns():
    state = ConversationState(max_turns=2)
    asyncio.run(state.record_turn())  # turn 1 -> True
    result = asyncio.run(state.record_turn())  # turn 2 -> reaches max, returns False
    assert result is False
    assert state.finished is True


# ---------------------------------------------------------------------------
# 4. record_turn returns False when already finished (after mark_finished)
# ---------------------------------------------------------------------------


def test_record_turn_returns_false_when_already_finished():
    state = ConversationState(max_turns=10)
    asyncio.run(state.mark_finished())
    result = asyncio.run(state.record_turn())
    assert result is False


# ---------------------------------------------------------------------------
# 5. mark_finished returns True on first call
# ---------------------------------------------------------------------------


def test_mark_finished_returns_true_first_call():
    state = ConversationState(max_turns=5)
    result = asyncio.run(state.mark_finished())
    assert result is True


# ---------------------------------------------------------------------------
# 6. mark_finished returns False on second call (idempotent)
# ---------------------------------------------------------------------------


def test_mark_finished_returns_false_second_call():
    state = ConversationState(max_turns=5)
    asyncio.run(state.mark_finished())
    result = asyncio.run(state.mark_finished())
    assert result is False


# ---------------------------------------------------------------------------
# 7. finished flag is True after mark_finished
# ---------------------------------------------------------------------------


def test_finished_flag_true_after_mark_finished():
    state = ConversationState(max_turns=5)
    assert state.finished is False
    asyncio.run(state.mark_finished())
    assert state.finished is True


# ---------------------------------------------------------------------------
# 8. Concurrent record_turn calls from multiple tasks don't exceed max_turns
# ---------------------------------------------------------------------------


def test_concurrent_record_turn_does_not_exceed_max_turns():
    """Fire many concurrent record_turn tasks; the turn count must not exceed max_turns."""

    async def _run():
        max_turns = 5
        state = ConversationState(max_turns=max_turns)
        # Launch 20 concurrent calls
        results = await asyncio.gather(*[state.record_turn() for _ in range(20)])
        # turn_count must not exceed max_turns
        assert state.turn_count <= max_turns, (
            f"turn_count ({state.turn_count}) exceeded max_turns ({max_turns})"
        )
        # Number of True results must equal max_turns - 1 (since the max-th turn returns False)
        true_count = sum(1 for r in results if r is True)
        assert true_count == max_turns - 1, (
            f"Expected {max_turns - 1} True results, got {true_count}"
        )

    asyncio.run(_run())
