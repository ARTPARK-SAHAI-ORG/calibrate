"""Tests covering remaining gaps in calibrate_agent/judges.py."""

import unittest
from unittest.mock import patch, MagicMock, AsyncMock


class TestUtilityHelpers(unittest.TestCase):
    def test_build_openrouter_client(self):
        from calibrate_agent import judges as J

        with patch.object(J, "AsyncOpenAI") as MockClient:
            J._build_openrouter_client()
            MockClient.assert_called_once()

    def test_attach_evaluator_id_with_id(self):
        from calibrate_agent.judges import attach_evaluator_id

        out = attach_evaluator_id(
            {"id": "ev_1", "name": "x"}, {"reasoning": "r", "match": True}
        )
        self.assertEqual(out["evaluator_id"], "ev_1")
        self.assertTrue(out["match"])

    def test_attach_evaluator_id_no_id(self):
        from calibrate_agent.judges import attach_evaluator_id

        original = {"reasoning": "r", "match": True}
        out = attach_evaluator_id({"name": "x"}, original)
        self.assertIs(out, original)

    def test_attach_evaluator_id_non_dict_result(self):
        from calibrate_agent.judges import attach_evaluator_id

        out = attach_evaluator_id({"id": "ev_1"}, "not-a-dict")
        self.assertEqual(out, "not-a-dict")

    def test_format_eval_lines_rating(self):
        from calibrate_agent.judges import format_evaluation_result_lines

        lines = format_evaluation_result_lines(
            {"name": "n", "type": "rating", "value": 4, "scale_max": 5, "reasoning": "ok"}
        )
        self.assertIn("4/5", lines[0])
        self.assertIn("Reason:", lines[1])

    def test_format_eval_lines_rating_no_scale_max(self):
        from calibrate_agent.judges import format_evaluation_result_lines

        lines = format_evaluation_result_lines(
            {"name": "n", "type": "rating", "value": 3}
        )
        self.assertIn("3", lines[0])

    def test_format_eval_lines_binary_fail(self):
        from calibrate_agent.judges import format_evaluation_result_lines

        lines = format_evaluation_result_lines({"name": "x", "type": "binary", "value": 0})
        self.assertIn("Fail", lines[0])

    def test_format_eval_lines_no_reasoning(self):
        from calibrate_agent.judges import format_evaluation_result_lines

        lines = format_evaluation_result_lines({"name": "x", "type": "binary", "value": 1})
        self.assertEqual(len(lines), 1)

    def test_rating_range_invalid(self):
        from calibrate_agent.judges import _rating_range

        with self.assertRaises(ValueError):
            _rating_range({"scale_min": 5, "scale_max": 1, "name": "bad"})

    def test_rating_range_valid(self):
        from calibrate_agent.judges import _rating_range

        self.assertEqual(_rating_range({"scale_min": 1, "scale_max": 3}), [1, 2, 3])

    def test_normalize_non_dict(self):
        from calibrate_agent.judges import _normalize_judge_api_result

        self.assertEqual(_normalize_judge_api_result("nope", "X"), "nope")

    def test_normalize_unwraps_nested(self):
        from calibrate_agent.judges import _normalize_judge_api_result

        result = {"X": {"reasoning": "r", "match": True}}
        out = _normalize_judge_api_result(result, "X")
        self.assertEqual(out, {"reasoning": "r", "match": True})

    def test_normalize_keeps_flat(self):
        from calibrate_agent.judges import _normalize_judge_api_result

        result = {"reasoning": "r", "match": True}
        self.assertEqual(_normalize_judge_api_result(result, "X"), result)


class TestRequireUniqueEvaluatorNames(unittest.TestCase):
    def test_non_list_is_noop(self):
        from calibrate_agent.judges import require_unique_evaluator_names

        require_unique_evaluator_names(None)
        require_unique_evaluator_names("not-a-list")
        require_unique_evaluator_names({"name": "x"})

    def test_non_dict_skipped(self):
        from calibrate_agent.judges import require_unique_evaluator_names

        require_unique_evaluator_names(["string-item", {"name": "ok"}])

    def test_non_string_name_skipped(self):
        from calibrate_agent.judges import require_unique_evaluator_names

        require_unique_evaluator_names([{"name": 123}, {"name": "ok"}])

    def test_duplicates_raise(self):
        from calibrate_agent.judges import require_unique_evaluator_names

        with self.assertRaises(ValueError) as ctx:
            require_unique_evaluator_names(
                [{"name": "dup"}, {"name": "dup"}, {"name": "ok"}]
            )
        self.assertIn("Duplicate", str(ctx.exception))

    def test_unique_ok(self):
        from calibrate_agent.judges import require_unique_evaluator_names

        require_unique_evaluator_names([{"name": "a"}, {"name": "b"}])


class TestRequireSimulationEvaluators(unittest.TestCase):
    def test_empty_raises(self):
        from calibrate_agent.judges import require_simulation_evaluators

        with self.assertRaises(ValueError):
            require_simulation_evaluators([])
        with self.assertRaises(ValueError):
            require_simulation_evaluators(None)

    def test_non_empty_ok(self):
        from calibrate_agent.judges import require_simulation_evaluators

        require_simulation_evaluators([{"name": "x"}])


class TestJudgeOneTextWithLangfuse(unittest.IsolatedAsyncioTestCase):
    async def test_judge_one_text_langfuse_branch(self):
        """Force langfuse_enabled True so the metadata update branch executes."""
        from calibrate_agent import judges as J

        # Build a fake instructor-patched client that returns a Pydantic-like result.
        fake_resp = MagicMock()
        fake_resp.model_dump.return_value = {"reasoning": "r", "match": True}

        fake_client = MagicMock()
        fake_client.chat.completions.create = AsyncMock(return_value=fake_resp)

        fake_lf = MagicMock()

        with patch.object(J, "instructor") as mock_instructor, \
             patch.object(J, "_build_openrouter_client", return_value=MagicMock()), \
             patch.object(J, "langfuse_enabled", True), \
             patch.object(J, "langfuse", fake_lf):
            mock_instructor.apatch.return_value = fake_client
            result = await J._judge_one_text(
                {"name": "n", "system_prompt": "sp"},
                "user prompt",
                "openai/gpt-4.1",
            )

        self.assertEqual(result["match"], True)
        fake_lf.update_current_span.assert_called_once()


class TestJudgeOneAudioWithLangfuse(unittest.IsolatedAsyncioTestCase):
    async def test_judge_one_audio_langfuse_branch(self):
        from calibrate_agent import judges as J

        fake_resp = MagicMock()
        fake_resp.model_dump.return_value = {"reasoning": "r", "match": True}

        fake_client = MagicMock()
        fake_client.chat.completions.create = AsyncMock(return_value=fake_resp)

        fake_lf = MagicMock()

        with patch.object(J, "instructor") as mock_instructor, \
             patch.object(J, "_build_openrouter_client", return_value=MagicMock()), \
             patch.object(J, "langfuse_enabled", True), \
             patch.object(J, "langfuse", fake_lf), \
             patch("calibrate_agent.langfuse.create_langfuse_audio_media", return_value=None):
            mock_instructor.apatch.return_value = fake_client
            result = await J._judge_one_audio(
                {"name": "n", "system_prompt": "sp"},
                "ref text",
                "/tmp/dummy.wav",
                "BASE64DATA",
                "openai/gpt-audio",
            )

        self.assertEqual(result["match"], True)
        fake_lf.update_current_trace.assert_called_once()


if __name__ == "__main__":
    unittest.main()
