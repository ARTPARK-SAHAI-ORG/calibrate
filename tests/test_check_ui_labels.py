import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_ui_labels import main  # noqa: E402


def run(tmp_path, mdx_text, capsys):
    src = tmp_path / "fe" / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "page.tsx").write_text('<Button>New agent</Button> <Tab label="Add header" />')
    mdx = tmp_path / "page.mdx"
    mdx.write_text(mdx_text)
    code = main(["--frontend", str(tmp_path / "fe"), str(mdx)])
    return code, capsys.readouterr().out


def test_label_in_frontend_passes(tmp_path, capsys):
    assert run(tmp_path, "Click **New agent**.\n", capsys) == (0, "")


def test_missing_label_reports_line(tmp_path, capsys):
    code, out = run(tmp_path, "Intro\n\nClick **New agent** then **Add metric:**\n", capsys)
    assert code == 1
    assert out.strip() == f'{tmp_path / "page.mdx"}:3: "Add metric" not found in frontend'


def test_code_fence_and_inline_code_ignored(tmp_path, capsys):
    text = "```python\nx = '**Add metric**'\n```\nUse `**Add metric**` here.\n"
    assert run(tmp_path, text, capsys) == (0, "")


def test_opt_out_comment(tmp_path, capsys):
    text = "{/* not-a-ui-label: Add metric */}\nThis is **Add metric** emphasis.\n"
    assert run(tmp_path, text, capsys) == (0, "")


def test_leading_plus_stripped(tmp_path, capsys):
    assert run(tmp_path, "Click **+ Add header**.\n", capsys) == (0, "")


def test_long_bold_span_is_not_a_label(tmp_path, capsys):
    assert run(tmp_path, "**this sentence is far too long to be a label**\n", capsys) == (0, "")


def test_bold_span_wrapped_across_lines(tmp_path, capsys):
    code, out = run(tmp_path, "Use **text\nsimulations** only. For **New agent** go.\n", capsys)
    assert code == 1
    assert out.strip() == f'{tmp_path / "page.mdx"}:1: "text simulations" not found in frontend'


def test_part_of_a_word_fails(tmp_path, capsys):
    code, _ = run(tmp_path, "Click **New agen**.\n", capsys)
    assert code == 1


def test_label_only_in_test_files_fails(tmp_path, capsys):
    tests = tmp_path / "fe" / "src" / "__tests__"
    tests.mkdir(parents=True)
    (tests / "page.test.tsx").write_text('expect("Agent list")')
    code, _ = run(tmp_path, "Open **Agent list**.\n", capsys)
    assert code == 1


def test_html_entities_in_source_match(tmp_path, capsys):
    src = tmp_path / "fe" / "src"
    src.mkdir(parents=True)
    (src / "a.tsx").write_text("<span>Request &amp; response</span>")
    assert run(tmp_path, "See **Request & response**.\n", capsys) == (0, "")
