"""Tiny environment-variable helpers.

Deliberately stdlib-only (imports just ``os``): both the CLI argument parser
(``calibrate_agent.cli``) and the library benchmark modules import from here.
The CLI must build its parser *without* importing the heavy
``calibrate_agent.utils`` module — that shifts scipy/numpy init order and trips a
scipy ``_CopyMode`` incompatibility in the voice path — so this module must never
grow a non-stdlib import.
"""

import os


def env_int(name: str, default: int) -> int:
    """Read an int from env var ``name``, falling back to ``default``.

    Used for benchmark concurrency knobs so they can be tuned via the
    environment (e.g. in CI / .env) without a code change; the hardcoded
    ``default`` is the fallback when the var is unset or not a valid int.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
