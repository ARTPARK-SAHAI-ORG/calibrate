from pathlib import Path

from calibrate_agent._env import env_int


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
