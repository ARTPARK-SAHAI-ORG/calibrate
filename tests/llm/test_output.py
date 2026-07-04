"""
Tests for calibrate_agent/llm/_output.py — print_benchmark_summary function.

Run with:
    python -m pytest tests/test_output.py -v
"""

import io
import unittest
from unittest.mock import patch


def _call(models, model_results, leaderboard_dir="./leaderboard", model_label=None):
    """Helper: call print_benchmark_summary and return (printed_lines, return_value)."""
    from calibrate_agent.llm._output import print_benchmark_summary

    captured = []
    with patch("builtins.print", side_effect=lambda *args, **kwargs: captured.append(
        " ".join(str(a) for a in args)
    )):
        result = print_benchmark_summary(
            models=models,
            model_results=model_results,
            leaderboard_dir=leaderboard_dir,
            model_label=model_label,
        )

    return captured, result


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences for easier assertion."""
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _plain_lines(captured):
    """Return list of ANSI-stripped lines from captured print calls."""
    return [_strip_ansi(line) for line in captured]


class TestAllModelsPass(unittest.TestCase):
    """All models pass — correct passed/total (pct%) lines, returns False."""

    def test_returns_false_when_all_pass(self):
        _, result = _call(
            models=["model-a"],
            model_results={"model-a": {"metrics": {"passed": 3, "total": 3}}},
        )
        self.assertFalse(result)

    def test_correct_fraction_and_pct(self):
        captured, _ = _call(
            models=["model-a"],
            model_results={"model-a": {"metrics": {"passed": 2, "total": 4}}},
        )
        lines = _plain_lines(captured)
        # Should contain "2/4 (50.0%)"
        combined = "\n".join(lines)
        self.assertIn("2/4", combined)
        self.assertIn("50.0%", combined)

    def test_100_pct_shown(self):
        captured, _ = _call(
            models=["x"],
            model_results={"x": {"metrics": {"passed": 5, "total": 5}}},
        )
        lines = _plain_lines(captured)
        combined = "\n".join(lines)
        self.assertIn("5/5", combined)
        self.assertIn("100.0%", combined)


class TestPartialPass(unittest.TestCase):
    """Partial pass — correct fractions and percentages."""

    def test_partial_pass_fraction(self):
        captured, result = _call(
            models=["m"],
            model_results={"m": {"metrics": {"passed": 1, "total": 3}}},
        )
        lines = _plain_lines(captured)
        combined = "\n".join(lines)
        self.assertIn("1/3", combined)
        self.assertIn("33.3%", combined)
        self.assertFalse(result)

    def test_two_thirds(self):
        captured, _ = _call(
            models=["m"],
            model_results={"m": {"metrics": {"passed": 2, "total": 3}}},
        )
        combined = "\n".join(_plain_lines(captured))
        self.assertIn("2/3", combined)
        self.assertIn("66.7%", combined)


class TestZeroTotal(unittest.TestCase):
    """Zero total — shows 0/0 (0.0%), no ZeroDivisionError."""

    def test_zero_total_no_crash(self):
        captured, result = _call(
            models=["m"],
            model_results={"m": {"metrics": {"passed": 0, "total": 0}}},
        )
        combined = "\n".join(_plain_lines(captured))
        self.assertIn("0/0", combined)
        self.assertIn("0.0%", combined)
        self.assertFalse(result)


class TestErrorModel(unittest.TestCase):
    """Error model — prints red error line, returns True."""

    def test_error_model_returns_true(self):
        _, result = _call(
            models=["bad-model"],
            model_results={"bad-model": {"status": "error", "error": "API key invalid"}},
        )
        self.assertTrue(result)

    def test_error_message_in_output(self):
        captured, _ = _call(
            models=["bad-model"],
            model_results={"bad-model": {"status": "error", "error": "API key invalid"}},
        )
        combined = "\n".join(captured)
        self.assertIn("API key invalid", combined)

    def test_error_line_contains_red_escape(self):
        captured, _ = _call(
            models=["bad-model"],
            model_results={"bad-model": {"status": "error", "error": "timeout"}},
        )
        # ANSI red is \033[31m
        combined = "\n".join(captured)
        self.assertIn("\033[31m", combined)


class TestMixedPassAndError(unittest.TestCase):
    """Mixed: one pass, one error — returns True."""

    def test_returns_true(self):
        _, result = _call(
            models=["good", "bad"],
            model_results={
                "good": {"metrics": {"passed": 1, "total": 1}},
                "bad": {"status": "error", "error": "failed"},
            },
        )
        self.assertTrue(result)

    def test_both_appear_in_output(self):
        captured, _ = _call(
            models=["good", "bad"],
            model_results={
                "good": {"metrics": {"passed": 1, "total": 1}},
                "bad": {"status": "error", "error": "failed"},
            },
        )
        combined = "\n".join(_plain_lines(captured))
        self.assertIn("good", combined)
        self.assertIn("bad", combined)


class TestModelLabelCallable(unittest.TestCase):
    """model_label callable applied to display name."""

    def test_label_lambda_applied(self):
        captured, _ = _call(
            models=["gpt-4.1"],
            model_results={"gpt-4.1": {"metrics": {"passed": 1, "total": 1}}},
            model_label=lambda m: f"openrouter/{m}",
        )
        combined = "\n".join(_plain_lines(captured))
        self.assertIn("openrouter/gpt-4.1", combined)

    def test_original_name_not_shown_when_label_applied(self):
        """The label replaces the model name in the summary line."""
        captured, _ = _call(
            models=["gpt-4.1"],
            model_results={"gpt-4.1": {"metrics": {"passed": 1, "total": 1}}},
            model_label=lambda m: f"provider/{m}",
        )
        # The label "provider/gpt-4.1" should appear, not just "gpt-4.1" alone
        combined = "\n".join(_plain_lines(captured))
        self.assertIn("provider/gpt-4.1", combined)


class TestLeaderboardDirPrinted(unittest.TestCase):
    """Leaderboard dir path printed at end."""

    def test_leaderboard_path_in_output(self):
        lb_dir = "/tmp/my-leaderboard"
        captured, _ = _call(
            models=["m"],
            model_results={"m": {"metrics": {"passed": 1, "total": 1}}},
            leaderboard_dir=lb_dir,
        )
        combined = "\n".join(captured)
        self.assertIn(lb_dir, combined)


class TestNestedMetricsFormat(unittest.TestCase):
    """Nested metrics format: {"metrics": {"passed": 1, "total": 2}} — correct values."""

    def test_nested_metrics_parsed(self):
        captured, result = _call(
            models=["m"],
            model_results={"m": {"metrics": {"passed": 1, "total": 2}}},
        )
        combined = "\n".join(_plain_lines(captured))
        self.assertIn("1/2", combined)
        self.assertIn("50.0%", combined)
        self.assertFalse(result)

    def test_nested_metrics_all_pass(self):
        captured, result = _call(
            models=["m"],
            model_results={"m": {"metrics": {"passed": 3, "total": 3}}},
        )
        combined = "\n".join(_plain_lines(captured))
        self.assertIn("3/3", combined)
        self.assertFalse(result)


class TestMultipleModels(unittest.TestCase):
    """Multiple models — all appear in output."""

    def test_all_models_in_output(self):
        models = ["model-a", "model-b", "model-c"]
        model_results = {
            "model-a": {"metrics": {"passed": 1, "total": 2}},
            "model-b": {"metrics": {"passed": 2, "total": 2}},
            "model-c": {"metrics": {"passed": 0, "total": 2}},
        }
        captured, result = _call(models=models, model_results=model_results)
        combined = "\n".join(_plain_lines(captured))
        for m in models:
            self.assertIn(m, combined)
        self.assertFalse(result)

    def test_model_order_preserved(self):
        """Models appear in the order given by the models list."""
        models = ["z-model", "a-model"]
        model_results = {
            "z-model": {"metrics": {"passed": 0, "total": 1}},
            "a-model": {"metrics": {"passed": 1, "total": 1}},
        }
        captured, _ = _call(models=models, model_results=model_results)
        combined = "\n".join(_plain_lines(captured))
        z_pos = combined.index("z-model")
        a_pos = combined.index("a-model")
        self.assertLess(z_pos, a_pos)

    def test_overall_summary_header_in_output(self):
        captured, _ = _call(
            models=["m"],
            model_results={"m": {"metrics": {"passed": 1, "total": 1}}},
        )
        combined = "\n".join(_plain_lines(captured))
        self.assertIn("Overall Summary", combined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
