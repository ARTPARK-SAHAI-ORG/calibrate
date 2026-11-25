"""Check that bold UI labels in docs pages appear somewhere in the frontend source.

Usage: python scripts/check_ui_labels.py --frontend <frontend repo> docs/a.mdx ...

A bold span of 1 to 6 words (outside code) is treated as a UI label. A page can
mark a bold span as plain emphasis with {/* not-a-ui-label: Exact text */}.
"""

import argparse
import html
import re
import sys
from pathlib import Path

BOLD = re.compile(r"\*\*([^*]+?)\*\*")
CODE = re.compile(r"^[ \t]*(```|~~~).*?^[ \t]*\1.*?$|`[^`\n]*`", re.M | re.S)
OPT_OUT = re.compile(r"\{/\*\s*not-a-ui-label:\s*(.+?)\s*\*/\}")
SOURCE_EXTS = {".ts", ".tsx", ".js", ".jsx", ".json"}


def read_frontend(frontend: Path) -> str:
    src = frontend / "src"
    if not src.is_dir():
        sys.exit(f"{src} not found: pass the frontend repo root to --frontend")
    # Test files mention labels the app may not show, so only app code counts.
    return html.unescape("\n".join(
        p.read_text(errors="ignore")
        for p in src.rglob("*")
        if p.suffix in SOURCE_EXTS and p.is_file()
        and "__tests__" not in p.parts and ".test." not in p.name
    ))


def labels(text: str):
    """Yield (line number, label) for each bold span that looks like a UI label."""
    skip = set(OPT_OUT.findall(text))
    # Blank out code but keep newlines so line numbers stay right.
    text = CODE.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)
    for m in BOLD.finditer(text):
        label = " ".join(m.group(1).split()).removeprefix("+ ").rstrip(":").strip()
        if label and len(label.split()) <= 6 and label not in skip:
            yield text.count("\n", 0, m.start()) + 1, label


def missing(mdx: Path, source: str):
    # Whole words only, so "Settin" does not pass on "Settings".
    return [
        (n, l) for n, l in labels(mdx.read_text())
        if not re.search(rf"(?<!\w){re.escape(l)}(?!\w)", source)
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", required=True, type=Path)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    source = read_frontend(args.frontend)
    failed = False
    for mdx in args.files:
        for lineno, label in missing(mdx, source):
            print(f'{mdx}:{lineno}: "{label}" not found in frontend')
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
