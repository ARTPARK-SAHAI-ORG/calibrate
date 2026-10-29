import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_page_videos import segments, speak  # noqa: E402

PAGE = """---
title: "Example"
---

Intro line.

1. Click **New agent**.

<Frame>
  <img src="/images/app/one.png" alt="One" />
</Frame>

Paste these instructions:

```
You are a tutor.
```

<Frame>
  <img src="/images/app/two.png" alt="Two" />
</Frame>

Last words.

## Next steps

Not read.
"""


def test_each_picture_gets_the_text_before_it():
    assert segments(PAGE) == [
        ("/images/app/one.png", "Intro line. Click New agent."),
        ("/images/app/two.png", "Paste these instructions (shown on the page). Last words."),
    ]


def test_symbols_are_said_as_words():
    text = speak("What is 1/2 + 1/4? Pick openai/gpt-4.1-mini, call save_progress, open https://example.org/x.")
    assert text == (
        "What is one half plus one quarter? Pick openai gpt-4.1-mini, "
        "call save progress, open the address shown on the page"
    )


def test_unknown_fraction_and_table_rows():
    assert speak("| a | b |\nThen 5/7.") == "Then 5 over 7."
