"""Synthetic Cobra command docs for the CLI-docs generator tests.

We intentionally do NOT snapshot the real ``calibrate-cli/docs`` here — that
drifts and would pin tests to upstream wording. Instead these hand-built samples
exercise every parsing/rendering *variation* the generator must handle:

- bool flags (no type token), typed flags, a flag with a default;
- a required flag marked with a trailing ``[required]``;
- a custom type placeholder (``type=full``) that is not a standard type name;
- placeholder-only junk flags (``--x-api-key … string value``);
- the generic ``--body`` escape hatch;
- a ``jq`` description containing a ``|`` (table-cell escaping);
- root/parent ``SEE ALSO`` lists that drive resource + subcommand ordering;
- a resource with subcommands, a single-command (leaf) resource, and a
  non-API command that should be filtered out.

The live CLI is validated separately by the sync workflow, which regenerates
from the real repo on every run.
"""

from __future__ import annotations

from pathlib import Path

ROOT_CMD = """## calibrate

Calibrate CLI

### Synopsis

Command-line access to the Calibrate platform.

```
calibrate [flags]
```

### Options

```
      --agent-mode             Enable structured errors and default TOON output for AI coding agents.
  -d, --debug                  Log request and response diagnostics to stderr
  -h, --help                   help for calibrate
  -q, --jq string              Filter output using a jq expression (e.g., '.items[] | .id')
      --no-interactive         Disable all interactive features (auto-prompting, TUI forms)
  -o, --output-format string   Output format. Options: pretty, json, yaml, table, toon. (default "pretty")
```

### SEE ALSO

* [calibrate widgets](calibrate_widgets.md)\t - Operations for widgets
* [calibrate ping](calibrate_ping.md)\t - Check connectivity
* [calibrate secrets](calibrate_secrets.md)\t - Manage local secrets
"""

WIDGETS = """## calibrate widgets

Operations for widgets

### Synopsis

Operations for widgets

```
calibrate widgets [flags]
```

### Options

```
  -h, --help   help for widgets
```

### SEE ALSO

* [calibrate](calibrate.md)\t - Calibrate CLI
* [calibrate widgets list](calibrate_widgets_list.md)\t - List widgets
* [calibrate widgets create](calibrate_widgets_create.md)\t - Create a widget
"""

WIDGETS_LIST = """## calibrate widgets list

List widgets

### Synopsis

List every widget in your workspace.

```
calibrate widgets list [flags]
```

### Examples

```
  calibrate widgets list
```

### Options

```
  -h, --help                help for list
      --x-api-key string    string value
      --x-org-uuid string   string value
```

### SEE ALSO

* [calibrate widgets](calibrate_widgets.md)\t - Operations for widgets
"""

WIDGETS_CREATE = """## calibrate widgets create

Create a widget

### Synopsis

Create a new widget in your workspace.

```
calibrate widgets create [flags]
```

### Examples

```
  calibrate widgets create --name <value>
```

### Options

```
  -a, --widget-id string        The widget to update. Must exist. [required]
      --body string             Request body as JSON (alternative to individual flags). Can also be provided via stdin.
  -c, --config-param type=full  Behavioral config. The keys depend on type.
      --enabled                 Enable the widget on create
  -h, --help                    help for create
  -n, --name string             Human-readable name
  -t, --tier tier               Tier level (default "basic")
      --x-api-key string        string value
      --x-org-uuid string       string value
```

### Options inherited from parent commands

```
      --agent-mode   Enable structured errors and default TOON output for AI coding agents.
  -d, --debug        Log request and response diagnostics to stderr
```

### SEE ALSO

* [calibrate widgets](calibrate_widgets.md)\t - Operations for widgets
"""

# A single-command (leaf) resource: parent file, no subcommands.
PING = """## calibrate ping

Check connectivity

### Synopsis

Check that the CLI can reach the API.

```
calibrate ping [flags]
```

### Examples

```
  calibrate ping
```

### Options

```
  -h, --help   help for ping
```

### SEE ALSO

* [calibrate](calibrate.md)\t - Calibrate CLI
"""

# A local/utility command that is NOT an API resource — filtered out.
SECRETS = """## calibrate secrets

Manage local secrets

### Synopsis

Store and read local secrets.

```
calibrate secrets [flags]
```

### Options

```
  -h, --help   help for secrets
```

### SEE ALSO

* [calibrate](calibrate.md)\t - Calibrate CLI
"""

# filename -> raw markdown
SAMPLES: dict[str, str] = {
    "calibrate.md": ROOT_CMD,
    "calibrate_widgets.md": WIDGETS,
    "calibrate_widgets_list.md": WIDGETS_LIST,
    "calibrate_widgets_create.md": WIDGETS_CREATE,
    "calibrate_ping.md": PING,
    "calibrate_secrets.md": SECRETS,
}

# The resources backed by an API operation (an OpenAPI tag). ``secrets`` is not.
API_TAGS = {"widgets", "ping"}


def write_samples(dst: Path) -> Path:
    """Write every sample to ``dst`` as a ``.md`` file; return ``dst``."""
    dst.mkdir(parents=True, exist_ok=True)
    for name, text in SAMPLES.items():
        (dst / name).write_text(text, encoding="utf-8")
    return dst
