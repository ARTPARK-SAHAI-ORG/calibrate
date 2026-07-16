from pathlib import Path

from calibrate_agent._env import (
    env_int,
    resolve_stt_max_concurrency,
    resolve_stt_parallelism,
)


class TestEnvInt:
    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("CALIBRATE_TEST_INT", raising=False)
        assert env_int("CALIBRATE_TEST_INT", 2) == 2

    def test_parses_int_from_env(self, monkeypatch):
        monkeypatch.setenv("CALIBRATE_TEST_INT", "7")
        assert env_int("CALIBRATE_TEST_INT", 2) == 7

    def test_falls_back_on_invalid_value(self, monkeypatch):
        monkeypatch.setenv("CALIBRATE_TEST_INT", "not-a-number")
        assert env_int("CALIBRATE_TEST_INT", 3) == 3

    def test_empty_string_falls_back(self, monkeypatch):
        monkeypatch.setenv("CALIBRATE_TEST_INT", "")
        assert env_int("CALIBRATE_TEST_INT", 4) == 4

    def test_negative_value_parsed(self, monkeypatch):
        monkeypatch.setenv("CALIBRATE_TEST_INT", "-1")
        assert env_int("CALIBRATE_TEST_INT", 2) == -1

class TestResolveSTTParallelism:
    def _clear(self, monkeypatch):
        monkeypatch.delenv("CALIBRATE_STT_MAX_PARALLEL", raising=False)
        monkeypatch.delenv("CALIBRATE_STT_MAX_CONCURRENCY", raising=False)

    def test_pipeline_default_is_one_one(self, monkeypatch):
        self._clear(monkeypatch)
        assert resolve_stt_parallelism("pipeline") == (1, 1)

    def test_direct_default_is_two_four(self, monkeypatch):
        self._clear(monkeypatch)
        assert resolve_stt_parallelism("direct") == (2, 4)

    def test_unknown_engine_falls_back_to_direct(self, monkeypatch):
        self._clear(monkeypatch)
        assert resolve_stt_parallelism("banana") == (2, 4)

    def test_explicit_values_win_over_everything(self, monkeypatch):
        monkeypatch.setenv("CALIBRATE_STT_MAX_PARALLEL", "9")
        monkeypatch.setenv("CALIBRATE_STT_MAX_CONCURRENCY", "9")
        # Explicit args beat both env and the per-engine default.
        assert resolve_stt_parallelism("pipeline", 3, 5) == (3, 5)

    def test_env_overrides_engine_default(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("CALIBRATE_STT_MAX_PARALLEL", "6")
        monkeypatch.setenv("CALIBRATE_STT_MAX_CONCURRENCY", "8")
        # No explicit args → env wins over the pipeline (1, 1) default.
        assert resolve_stt_parallelism("pipeline") == (6, 8)

    def test_partial_explicit_leaves_other_to_default(self, monkeypatch):
        self._clear(monkeypatch)
        # Only concurrency given explicitly; parallel falls to pipeline default.
        assert resolve_stt_parallelism("pipeline", max_concurrency=5) == (1, 5)


class TestResolveSTTMaxConcurrency:
    def _clear(self, monkeypatch):
        monkeypatch.delenv("CALIBRATE_STT_MAX_PARALLEL", raising=False)
        monkeypatch.delenv("CALIBRATE_STT_MAX_CONCURRENCY", raising=False)

    def test_per_engine_default(self, monkeypatch):
        self._clear(monkeypatch)
        assert resolve_stt_max_concurrency("pipeline") == 1
        assert resolve_stt_max_concurrency("direct") == 4

    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("CALIBRATE_STT_MAX_CONCURRENCY", "9")
        assert resolve_stt_max_concurrency("pipeline", 3) == 3

    def test_env_overrides_default(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("CALIBRATE_STT_MAX_CONCURRENCY", "7")
        assert resolve_stt_max_concurrency("pipeline") == 7

    def test_ignores_the_parallel_env_var(self, monkeypatch):
        self._clear(monkeypatch)
        # A single-provider resolve must not be swayed by the across-providers knob.
        monkeypatch.setenv("CALIBRATE_STT_MAX_PARALLEL", "99")
        assert resolve_stt_max_concurrency("pipeline") == 1


class TestEnvStdlibOnly:
    def test_stays_stdlib_only(self):
        """The module must stay stdlib-only so the CLI can import it at
        parser-build time without dragging in scipy/numpy/pipecat (which trips a
        scipy ``_CopyMode`` incompatibility in the voice path)."""
        import ast
        import calibrate_agent._env as env_mod

        source = Path(env_mod.__file__).read_text()
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        assert imported <= {"os"}, f"_env.py must stay stdlib-only, imports: {imported}"
