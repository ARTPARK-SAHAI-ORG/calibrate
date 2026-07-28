import argparse
import ast
import inspect

import calibrate_agent._cli_args as cli_args
from calibrate_agent._cli_args import (
    DEFAULT_STT_LLM_JUDGES,
    STT_LLM_JUDGES,
    add_assume_yes_arg,
    add_stt_engine_args,
    add_stt_eval_args,
    add_stt_judges_arg,
    add_stt_max_parallel_arg,
    add_stt_skip_llm_judges_arg,
    parse_stt_llm_judges,
    resolve_stt_llm_judges,
)


def _defaults(include_max_parallel):
    parser = argparse.ArgumentParser()
    add_stt_eval_args(parser, include_max_parallel=include_max_parallel)
    return vars(parser.parse_args([]))


class TestAddSTTEvalArgs:
    def test_multi_provider_defaults(self):
        assert _defaults(include_max_parallel=True) == {
            "skip_llm_judges": False,
            "judges": None,
            "max_parallel": None,
            "engine": "pipeline",
            "max_concurrency": None,
        }

    def test_single_provider_omits_max_parallel(self):
        ns = _defaults(include_max_parallel=False)
        assert "max_parallel" not in ns
        assert ns == {
            "skip_llm_judges": False,
            "judges": None,
            "engine": "pipeline",
            "max_concurrency": None,
        }

    def test_values_parse(self):
        parser = argparse.ArgumentParser()
        add_stt_eval_args(parser, include_max_parallel=True)
        ns = parser.parse_args(
            [
                "--skip-llm-judges",
                "--max-parallel",
                "3",
                "--engine",
                "direct",
                "--max-concurrency",
                "8",
            ]
        )
        assert (ns.skip_llm_judges, ns.max_parallel, ns.engine, ns.max_concurrency) == (
            True,
            3,
            "direct",
            8,
        )

    def test_judges_parses_subset(self):
        parser = argparse.ArgumentParser()
        add_stt_eval_args(parser, include_max_parallel=False)
        ns = parser.parse_args(["--judges", "intent,llm_wer"])
        assert ns.judges == frozenset({"intent", "llm_wer"})

    def test_engine_choices_enforced(self):
        parser = argparse.ArgumentParser()
        add_stt_engine_args(parser)
        try:
            parser.parse_args(["--engine", "bogus"])
        except SystemExit:
            pass
        else:  # pragma: no cover - fails the assert below
            raise AssertionError("invalid --engine choice should have been rejected")


class TestParseSttLlmJudges:
    def test_parses_comma_separated(self):
        assert parse_stt_llm_judges("intent, semantic_wer") == frozenset(
            {"intent", "semantic_wer"}
        )

    def test_rejects_unknown(self):
        try:
            parse_stt_llm_judges("intent,bogus")
        except ValueError as e:
            assert "bogus" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError for unknown judge")

    def test_rejects_empty(self):
        try:
            parse_stt_llm_judges(" , ")
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected ValueError for empty list")

    def test_cli_rejects_unknown_name(self):
        parser = argparse.ArgumentParser()
        add_stt_judges_arg(parser)
        try:
            parser.parse_args(["--judges", "nope"])
        except SystemExit:
            pass
        else:  # pragma: no cover
            raise AssertionError("unknown --judges name should have been rejected")


class TestResolveSttLlmJudges:
    def test_default_is_all_three(self):
        assert resolve_stt_llm_judges(skip_llm_judges=False, judges=None) == (
            DEFAULT_STT_LLM_JUDGES
        )
        assert DEFAULT_STT_LLM_JUDGES == frozenset(STT_LLM_JUDGES)

    def test_skip_returns_empty(self):
        assert resolve_stt_llm_judges(skip_llm_judges=True, judges=None) == frozenset()

    def test_subset_returned(self):
        subset = frozenset({"intent"})
        assert (
            resolve_stt_llm_judges(skip_llm_judges=False, judges=subset) == subset
        )

    def test_mutual_exclusion(self):
        try:
            resolve_stt_llm_judges(
                skip_llm_judges=True, judges=frozenset({"intent"})
            )
        except SystemExit as e:
            assert "--judges" in str(e)
            assert "--skip-llm-judges" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected SystemExit for mutual exclusion")


class TestBuildersComposeConsistently:
    """The individual builders must produce the same actions add_stt_eval_args does."""

    def test_pieces_match_combined(self):
        combined = argparse.ArgumentParser()
        add_stt_eval_args(combined, include_max_parallel=True)

        pieces = argparse.ArgumentParser()
        add_stt_skip_llm_judges_arg(pieces)
        add_stt_judges_arg(pieces)
        add_stt_max_parallel_arg(pieces)
        add_stt_engine_args(pieces)

        def signature(parser):
            return {
                a.dest: (a.option_strings, a.default, a.choices, a.help)
                for a in parser._actions
                if a.dest != "help"
            }

        assert signature(combined) == signature(pieces)


class TestAddAssumeYesArg:
    def test_default_false(self):
        parser = argparse.ArgumentParser()
        add_assume_yes_arg(parser)
        ns = parser.parse_args([])
        assert ns.yes is False

    def test_long_flag_parses(self):
        parser = argparse.ArgumentParser()
        add_assume_yes_arg(parser)
        ns = parser.parse_args(["--yes"])
        assert ns.yes is True

    def test_short_flag_parses(self):
        parser = argparse.ArgumentParser()
        add_assume_yes_arg(parser)
        ns = parser.parse_args(["-y"])
        assert ns.yes is True


class TestStdlibOnly:
    """cli.py builds its parser via this module before any heavy import, so it
    must not pull in scipy/numpy/pipecat. Guard that it imports nothing at all."""

    def test_module_has_no_imports(self):
        tree = ast.parse(inspect.getsource(cli_args))
        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert imports == []
