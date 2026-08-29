"""
Tests for resuming the text-simulation eval-only run
(``calibrate_agent.llm.run_simulation.run_eval_only_simulations``).

The judge is replaced by a counting stand-in, so no API calls are made.
"""

import asyncio
import json

import pandas as pd
import pytest
from unittest.mock import patch

import calibrate_agent.llm.run_simulation as run_simulation
from calibrate_agent.llm.run_simulation import (
    run_eval_only_simulations,
    validate_simulation_eval_only_dataset,
)


def _evaluator(name="helpfulness"):
    return {
        "name": name,
        "system_prompt": "Evaluate whether the agent was helpful.",
        "judge_model": "openai/gpt-4.1",
    }


def _conversation(conversation_id, content):
    """One dataset item whose score is decided by ``content``."""
    return {
        "id": conversation_id,
        "name": f"conv {conversation_id}",
        "conversation_history": [{"role": "user", "content": content}],
    }


class _CountingJudge:
    """Stand-in for ``evaluate_simuation``: a transcript saying "good" passes."""

    def __init__(self):
        self.transcripts = []

    async def __call__(self, transcript, evaluators, fallback_model=None):
        self.transcripts.append(transcript)
        passed = transcript[-1]["content"] == "good"
        return {
            ev["name"]: {"match": passed, "reasoning": "because"} for ev in evaluators
        }

    @property
    def call_count(self):
        return len(self.transcripts)


def _run(dataset, output_dir, evaluators=None, overwrite=False):
    """Run the eval-only pass and return the judge stand-in used."""
    judge = _CountingJudge()
    with patch("calibrate_agent.llm.run_simulation.evaluate_simuation", judge):
        failed = asyncio.run(
            run_eval_only_simulations(
                config={"evaluators": evaluators or [_evaluator()]},
                dataset=dataset,
                output_dir=str(output_dir),
                parallel=2,
                overwrite=overwrite,
            )
        )
    assert failed == 0, f"Expected no failures, got {failed}"
    return judge


def _results(output_dir):
    return pd.read_csv(output_dir / "results.csv", dtype={"id": str})


def _metrics(output_dir):
    with open(output_dir / "metrics.json") as f:
        return json.load(f)


def _row_dirs(output_dir):
    return sorted(d.name for d in output_dir.iterdir() if d.name.startswith("row_"))


def _transcript(output_dir, row_name):
    with open(output_dir / row_name / "transcript.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Resuming
# ---------------------------------------------------------------------------


def test_resumed_run_only_judges_the_missing_conversations(tmp_path):
    out = tmp_path / "out"
    first = _run([_conversation("a", "good"), _conversation("b", "bad")], out)
    assert first.call_count == 2

    second = _run(
        [
            _conversation("a", "good"),
            _conversation("b", "bad"),
            _conversation("c", "good"),
        ],
        out,
    )
    assert second.call_count == 1, (
        f"Expected only the new conversation to be judged, got "
        f"{second.call_count} judge calls"
    )
    assert second.transcripts[0][-1]["content"] == "good"


def test_resumed_run_writes_every_row_to_results_csv(tmp_path):
    out = tmp_path / "out"
    _run([_conversation("a", "good"), _conversation("b", "bad")], out)
    _run(
        [
            _conversation("a", "good"),
            _conversation("b", "bad"),
            _conversation("c", "good"),
        ],
        out,
    )

    df = _results(out)
    assert list(df["id"]) == ["a", "b", "c"], f"Unexpected rows: {df.to_dict()}"
    assert list(df["row_id"]) == ["row_1", "row_2", "row_3"]
    assert list(df["helpfulness"]) == [True, False, True]
    assert _row_dirs(out) == ["row_1", "row_2", "row_3"]


def test_resumed_metrics_match_a_single_run(tmp_path):
    dataset = [
        _conversation("a", "good"),
        _conversation("b", "bad"),
        _conversation("c", "good"),
    ]

    resumed = tmp_path / "resumed"
    _run(dataset[:2], resumed)
    _run(dataset, resumed)

    single = tmp_path / "single"
    _run(dataset, single)

    assert _metrics(resumed) == _metrics(single)
    assert _metrics(single)["helpfulness"]["mean"] == pytest.approx(2 / 3)


def test_numeric_looking_id_resumes(tmp_path):
    out = tmp_path / "out"
    dataset = [_conversation("1", "good"), _conversation("2", "bad")]
    _run(dataset, out)

    second = _run(dataset, out)
    assert second.call_count == 0, (
        f"Expected nothing to be judged again, got {second.call_count} judge calls"
    )
    assert list(_results(out)["id"]) == ["1", "2"]


def test_reordered_dataset_reuses_each_conversations_own_scores(tmp_path):
    out = tmp_path / "out"
    _run([_conversation("a", "good"), _conversation("b", "bad")], out)

    reordered = [
        _conversation("c", "bad"),
        _conversation("a", "good"),
        _conversation("b", "bad"),
    ]
    second = _run(reordered, out)

    assert second.call_count == 1, (
        f"Expected only conversation 'c' to be judged, got "
        f"{second.call_count} judge calls"
    )

    df = _results(out)
    assert list(df["id"]) == ["c", "a", "b"]
    assert list(df["helpfulness"]) == [False, True, False], (
        f"Scores followed positions instead of ids: {df.to_dict()}"
    )
    assert _row_dirs(out) == ["row_1", "row_2", "row_3"]
    assert _transcript(out, "row_1")[-1]["content"] == "bad"
    assert _transcript(out, "row_2")[-1]["content"] == "good"


def test_conversation_dropped_from_the_dataset_is_dropped_from_results(tmp_path):
    out = tmp_path / "out"
    _run([_conversation("a", "good"), _conversation("b", "bad")], out)

    _run([_conversation("a", "good")], out)

    df = _results(out)
    assert list(df["id"]) == ["a"]
    assert _row_dirs(out) == ["row_1"], (
        f"Stale conversation folder left behind: {_row_dirs(out)}"
    )


def test_changed_evaluator_set_is_not_mixed_with_the_old_one(tmp_path):
    out = tmp_path / "out"
    dataset = [_conversation("a", "good"), _conversation("b", "bad")]
    _run(dataset, out, evaluators=[_evaluator("old_check")])

    second = _run(dataset, out, evaluators=[_evaluator("new_check")])
    assert second.call_count == 2, (
        f"Expected both conversations to be judged again under the new "
        f"evaluator, got {second.call_count} judge calls"
    )

    metrics = _metrics(out)
    assert set(metrics) == {"new_check"}
    df = _results(out)
    assert "old_check" not in df.columns


# ---------------------------------------------------------------------------
# Overwrite
# ---------------------------------------------------------------------------


def test_overwrite_starts_clean(tmp_path):
    out = tmp_path / "out"
    dataset = [_conversation("a", "good"), _conversation("b", "bad")]
    _run(dataset, out)

    second = _run(dataset, out, overwrite=True)
    assert second.call_count == 2, (
        f"Expected every conversation to be judged again, got "
        f"{second.call_count} judge calls"
    )
    assert _row_dirs(out) == ["row_1", "row_2"]
    assert list(_results(out)["id"]) == ["a", "b"]


def test_overwrite_removes_folders_of_conversations_no_longer_in_the_dataset(tmp_path):
    out = tmp_path / "out"
    _run([_conversation("a", "good"), _conversation("b", "bad")], out)

    _run([_conversation("a", "good")], out, overwrite=True)
    assert _row_dirs(out) == ["row_1"]


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------


def test_dataset_without_id_is_rejected(tmp_path):
    dataset = [{"conversation_history": [{"role": "user", "content": "hi"}]}]

    is_valid, error = validate_simulation_eval_only_dataset(dataset)
    assert is_valid is False
    assert "'id'" in error, f"Error does not mention the id field: {error}"

    with pytest.raises(ValueError, match="'id'"):
        asyncio.run(
            run_eval_only_simulations(
                config={"evaluators": [_evaluator()]},
                dataset=dataset,
                output_dir=str(tmp_path / "out"),
            )
        )


def test_duplicate_ids_are_rejected(tmp_path):
    dataset = [_conversation("a", "good"), _conversation("a", "bad")]

    is_valid, error = validate_simulation_eval_only_dataset(dataset)
    assert is_valid is False
    assert "duplicate id" in error.lower(), f"Unclear error message: {error}"

    with pytest.raises(ValueError, match="duplicate id"):
        asyncio.run(
            run_eval_only_simulations(
                config={"evaluators": [_evaluator()]},
                dataset=dataset,
                output_dir=str(tmp_path / "out"),
            )
        )


def test_valid_dataset_passes_validation():
    is_valid, error = validate_simulation_eval_only_dataset(
        [_conversation("a", "good"), _conversation(2, "bad")]
    )
    assert is_valid, error


def test_empty_dataset_is_rejected(tmp_path):
    is_valid, error = validate_simulation_eval_only_dataset([])
    assert is_valid is False
    assert "empty" in error.lower(), f"Unclear error message: {error}"

    out = tmp_path / "out"
    _run([_conversation("a", "good")], out)

    with pytest.raises(ValueError, match="empty"):
        asyncio.run(
            run_eval_only_simulations(
                config={"evaluators": [_evaluator()]},
                dataset=[],
                output_dir=str(out),
            )
        )
    assert _row_dirs(out) == ["row_1"], (
        f"An empty dataset wiped the stored conversations: {_row_dirs(out)}"
    )
    assert list(_results(out)["id"]) == ["a"]


# ---------------------------------------------------------------------------
# Reuse only when the rubric and the conversation are unchanged
# ---------------------------------------------------------------------------


def _rating_evaluator(name="helpfulness", scale_max=5):
    return {
        "name": name,
        "system_prompt": "Rate how helpful the agent was.",
        "judge_model": "openai/gpt-4.1",
        "type": "rating",
        "scale_min": 1,
        "scale_max": scale_max,
    }


class _RatingJudge:
    """Stand-in for ``evaluate_simuation`` that always returns the top rating."""

    def __init__(self):
        self.transcripts = []

    async def __call__(self, transcript, evaluators, fallback_model=None):
        self.transcripts.append(transcript)
        return {
            ev["name"]: {"score": int(ev["scale_max"]), "reasoning": "because"}
            for ev in evaluators
        }

    @property
    def call_count(self):
        return len(self.transcripts)


def _run_rating(dataset, output_dir, evaluators):
    judge = _RatingJudge()
    with patch("calibrate_agent.llm.run_simulation.evaluate_simuation", judge):
        failed = asyncio.run(
            run_eval_only_simulations(
                config={"evaluators": evaluators},
                dataset=dataset,
                output_dir=str(output_dir),
                parallel=2,
            )
        )
    assert failed == 0, f"Expected no failures, got {failed}"
    return judge


def test_evaluator_turned_into_a_rating_is_judged_again(tmp_path):
    out = tmp_path / "out"
    dataset = [_conversation("a", "good"), _conversation("b", "bad")]
    _run(dataset, out, evaluators=[_evaluator("helpfulness")])

    second = _run_rating(dataset, out, evaluators=[_rating_evaluator("helpfulness")])
    assert second.call_count == 2, (
        f"Pass/fail scores were reused under a rating rubric of the same name, "
        f"got {second.call_count} judge calls"
    )

    metrics = _metrics(out)["helpfulness"]
    assert metrics["type"] == "rating", metrics
    assert metrics["mean"] == pytest.approx(5.0), metrics


def test_changed_rating_scale_is_judged_again(tmp_path):
    out = tmp_path / "out"
    dataset = [_conversation("a", "good")]
    _run_rating(dataset, out, evaluators=[_rating_evaluator(scale_max=5)])

    second = _run_rating(dataset, out, evaluators=[_rating_evaluator(scale_max=10)])
    assert second.call_count == 1, (
        f"Scores from the 1-5 scale were reused on the 1-10 scale, got "
        f"{second.call_count} judge calls"
    )
    assert _metrics(out)["helpfulness"]["mean"] == pytest.approx(10.0)


def test_edited_conversation_with_the_same_id_is_judged_again(tmp_path):
    out = tmp_path / "out"
    _run([_conversation("a", "good"), _conversation("b", "bad")], out)

    second = _run([_conversation("a", "bad"), _conversation("b", "bad")], out)
    assert second.call_count == 1, (
        f"Expected the edited conversation to be judged again, got "
        f"{second.call_count} judge calls"
    )

    df = _results(out)
    assert list(df["helpfulness"]) == [False, False], (
        f"The edited conversation kept the old conversation's score: {df.to_dict()}"
    )
    assert _transcript(out, "row_1")[-1]["content"] == "bad"


# ---------------------------------------------------------------------------
# Interrupted runs and failed judging
# ---------------------------------------------------------------------------


def test_interrupted_write_keeps_the_reused_scores_on_disk(tmp_path):
    out = tmp_path / "out"
    _run([_conversation("a", "good"), _conversation("b", "bad")], out)

    real_write = run_simulation._write_eval_only_row
    calls = {"n": 0}

    def failing_write(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real_write(*args, **kwargs)

    reordered = [
        _conversation("b", "bad"),
        _conversation("a", "good"),
        _conversation("c", "good"),
    ]
    with patch.object(run_simulation, "_write_eval_only_row", failing_write):
        with pytest.raises(OSError):
            _run(reordered, out)

    third = _run(reordered, out)
    assert third.call_count == 1, (
        f"An interrupted run lost scores that were already paid for: "
        f"{third.call_count} judge calls instead of 1"
    )


def test_conversation_the_judge_fails_on_leaves_a_folder_recording_the_failure(
    tmp_path,
):
    out = tmp_path / "out"

    async def judge(transcript, evaluators, fallback_model=None):
        if transcript[-1]["content"] == "boom":
            raise RuntimeError("judge exploded")
        return {ev["name"]: {"match": True, "reasoning": "ok"} for ev in evaluators}

    with patch("calibrate_agent.llm.run_simulation.evaluate_simuation", judge):
        failed = asyncio.run(
            run_eval_only_simulations(
                config={"evaluators": [_evaluator()]},
                dataset=[_conversation("a", "good"), _conversation("b", "boom")],
                output_dir=str(out),
            )
        )

    assert failed == 1
    assert _row_dirs(out) == ["row_1", "row_2"], (
        f"The failed conversation left no folder: {_row_dirs(out)}"
    )
    with open(out / "row_2" / "row.json") as f:
        stored = json.load(f)
    assert "judge exploded" in stored["error"], stored
    assert "evaluation_results" not in stored, stored
