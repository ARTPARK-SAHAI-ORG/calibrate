# Installation

Install Pense and get ready to start evaluating voice agents.

## Install uv

Pense uses `uv` to manage dependencies. If you don't have it installed:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installation:

```bash
uv --version
```

## Install Pense

1. Clone or navigate to the Pense repository:

```bash
cd /path/to/agentloop
```

2. Create a virtual environment and install dependencies:

```bash
uv sync --frozen
```

This will:
- Create a virtual environment
- Install all required dependencies
- Set up the `pense` CLI command

## Verify Installation

Check that Pense is installed correctly:

```bash
pense --help
```

You should see the Pense CLI help menu with available commands.

## What's Next?

Now that Pense is installed, configure your API keys:

[Configure API Keys :material-arrow-right:](configuration.md){ .md-button }
