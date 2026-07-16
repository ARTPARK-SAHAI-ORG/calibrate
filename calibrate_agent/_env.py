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


# Per-engine parallelism fallback defaults for the STT benchmark, as
# ``(max_parallel, max_concurrency)``. The pipeline engine reports TTFS — a
# wall-clock latency measured on the shared event loop — so it defaults to no
# contention (one provider, one clip at a time) to keep that number clean. The
# direct engine has no latency metric, so it defaults to parallel for
# throughput. Precedence when resolving: explicit CLI flag > env var > these.
STT_PARALLELISM_DEFAULTS = {
    "pipeline": (1, 1),
    "direct": (2, 4),
}


def resolve_stt_max_concurrency(engine, max_concurrency=None):
    """Resolve per-provider clip concurrency (the ``CALIBRATE_STT_MAX_CONCURRENCY``
    knob) for an STT run.

    An explicit (non-None) value wins; else the env var if set; else the
    per-engine default in ``STT_PARALLELISM_DEFAULTS`` (unknown engines fall back
    to ``direct``). Single-provider callers use this directly so they don't touch
    the across-providers knob they have no use for.
    """
    if max_concurrency is not None:
        return max_concurrency
    _mp_default, mc_default = STT_PARALLELISM_DEFAULTS.get(
        engine, STT_PARALLELISM_DEFAULTS["direct"]
    )
    return env_int("CALIBRATE_STT_MAX_CONCURRENCY", mc_default)


def resolve_stt_parallelism(engine, max_parallel=None, max_concurrency=None):
    """Resolve ``(max_parallel, max_concurrency)`` for an STT benchmark run.

    An explicit (non-None) value always wins. Otherwise the env var
    (``CALIBRATE_STT_MAX_PARALLEL`` / ``CALIBRATE_STT_MAX_CONCURRENCY``) is used
    if set, else the per-engine default in ``STT_PARALLELISM_DEFAULTS`` (unknown
    engines fall back to the ``direct`` defaults).
    """
    mp_default, _mc_default = STT_PARALLELISM_DEFAULTS.get(
        engine, STT_PARALLELISM_DEFAULTS["direct"]
    )
    resolved_parallel = (
        max_parallel
        if max_parallel is not None
        else env_int("CALIBRATE_STT_MAX_PARALLEL", mp_default)
    )
    return resolved_parallel, resolve_stt_max_concurrency(engine, max_concurrency)
