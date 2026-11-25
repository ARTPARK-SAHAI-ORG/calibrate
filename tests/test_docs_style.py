import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_docs_style import (  # noqa: E402
    added_lines,
    check_file,
    is_checked,
    line_violations,
    main,
    prose_lines,
)


class TestIsChecked:
    def test_hand_written_page(self):
        assert is_checked("docs/cli/text-to-text.mdx")

    def test_generated_pages_skipped(self):
        for path in (
            "docs/api-reference/openapi.json",
            "docs/sdk/agents/create.mdx",
            "docs/mcp/tools.mdx",
            "docs/cli/calibrate/agents.mdx",
            "docs/reference/api-keys.mdx",
        ):
            assert not is_checked(path), path

    def test_non_docs_and_non_mdx_skipped(self):
        assert not is_checked("README.md")
        assert not is_checked("docs/docs.json")
        assert not is_checked("calibrate_agent/connections.py")


class TestProseLines:
    def test_frontmatter_and_fences_left_out(self):
        text = (
            "---\n"
            'title: "A page"\n'
            "---\n"
            "Prose here.\n"
            "```json\n"
            '{ "a": "b — c" }\n'
            "```\n"
            "More prose.\n"
        )
        assert prose_lines(text) == {4: "Prose here.", 8: "More prose."}


class TestLineViolations:
    def test_em_dash(self):
        (rule, _), = line_violations("The agent replies — and stops.")
        assert rule == "em-dash"

    def test_banned_phrase_any_case(self):
        (rule, detail), = line_violations("It takes One Instruction Per Call.")
        assert rule == "banned phrase"
        assert "one instruction per call" in detail

    def test_phrase_inside_backticks_ignored(self):
        assert line_violations("The `under the hood` key is sent.") == []

    def test_clean_line(self):
        assert line_violations("Set `agent_type` to `general` in the config.") == []

    def test_several_rules_on_one_line(self):
        rules = {rule for rule, _ in line_violations("Out of the box — it works.")}
        assert rules == {"em-dash", "banned phrase"}


class TestCheckFile:
    def test_reports_line_numbers(self, tmp_path):
        page = tmp_path / "page.mdx"
        page.write_text("Fine line.\nA plain instruction goes here.\n")
        messages = check_file(page, None)
        assert len(messages) == 1
        assert ":2:" in messages[0]

    def test_only_reports_the_lines_asked_for(self, tmp_path):
        page = tmp_path / "page.mdx"
        page.write_text("Out of the box.\nUnder the hood.\n")
        assert len(check_file(page, {2})) == 1

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert check_file(tmp_path / "gone.mdx", None) == []


class TestAddedLines:
    def test_parses_a_unified_diff(self):
        diff = (
            "diff --git a/docs/a.mdx b/docs/a.mdx\n"
            "--- a/docs/a.mdx\n"
            "+++ b/docs/a.mdx\n"
            "@@ -3,0 +4,2 @@\n"
            "+first added\n"
            "+second added\n"
            "@@ -9 +11 @@\n"
            "-gone\n"
            "+replaced\n"
        )
        assert added_lines(diff) == {"docs/a.mdx": {4, 5, 11}}


class TestMain:
    def test_clean_page_passes(self, tmp_path, monkeypatch, capsys):
        page = tmp_path / "docs" / "cli" / "page.mdx"
        page.parent.mkdir(parents=True)
        page.write_text("Set `agent_type` to `general` in the config.\n")
        monkeypatch.chdir(tmp_path)
        assert main(["docs/cli/page.mdx"]) == 0
        assert capsys.readouterr().out == ""

    def test_violation_fails_and_names_the_file(self, tmp_path, monkeypatch, capsys):
        page = tmp_path / "docs" / "cli" / "page.mdx"
        page.parent.mkdir(parents=True)
        page.write_text("An agent that takes one instruction per call.\n")
        monkeypatch.chdir(tmp_path)
        assert main(["docs/cli/page.mdx"]) == 1
        out = capsys.readouterr().out
        assert "docs/cli/page.mdx:1" in out
        assert "DOCS_STYLE.md" in out
