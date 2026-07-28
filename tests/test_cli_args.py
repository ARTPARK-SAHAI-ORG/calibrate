import argparse
import ast
import inspect

import calibrate_agent._cli_args as cli_args
from calibrate_agent._cli_args import (
    add_assume_yes_arg,
    add_stt_engine_args,
    add_stt_eval_args,
    add_stt_max_parallel_arg,
    add_stt_skip_llm_judges_arg,
)


def _defaults(include_max_parallel):
    parser = argparse.ArgumentParser()
    add_stt_eval_args(parser, include_max_parallel=include_max_parallel)
    return vars(parser.parse_args([]))


class TestAddSTTEvalArgs:
    def test_multi_provider_defaults(self):
        assert _defaults(include_max_parallel=True) == {
            "skip_llm_judges": False,
            "max_parallel": None,
            "engine": "pipeline",
            "max_concurrency": None,
        }

    def test_single_provider_omits_max_parallel(self):
        ns = _defaults(include_max_parallel=False)
        assert "max_parallel" not in ns
        assert ns == {
            "skip_llm_judges": False,
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

    def test_engine_choices_enforced(self):
        parser = argparse.ArgumentParser()
        add_stt_engine_args(parser)
        try:
            parser.parse_args(["--engine", "bogus"])
        except SystemExit:
            pass
        else:  # pragma: no cover - fails the assert below
            raise AssertionError("invalid --engine choice should have been rejected")


class TestBuildersComposeConsistently:
    """The individual builders must produce the same actions add_stt_eval_args does."""

    def test_pieces_match_combined(self):
        combined = argparse.ArgumentParser()
        add_stt_eval_args(combined, include_max_parallel=True)

        pieces = argparse.ArgumentParser()
        add_stt_skip_llm_judges_arg(pieces)
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
