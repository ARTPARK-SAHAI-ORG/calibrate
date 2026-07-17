"""Shared argparse builders for the STT eval/benchmark CLIs.

The ``--skip-llm-judges``, ``--max-parallel``, ``--engine`` and
``--max-concurrency`` flags are accepted by three entry points that must stay in
lock-step: the top-level ``calibrate-agent stt`` parser (``calibrate_agent.cli``),
the multi-provider benchmark (``calibrate_agent.stt.benchmark``) and the
single-provider eval (``calibrate_agent.stt.eval``). Defining each flag once here
keeps their help text, defaults and choices from drifting apart.

Deliberately stdlib-only (imports nothing): ``calibrate_agent.cli`` builds its
parser by calling these, and the CLI must build its parser *without* importing
the heavy ``calibrate_agent.utils`` module — that shifts scipy/numpy init order
and trips a scipy ``_CopyMode`` incompatibility in the voice path. Like
``calibrate_agent._env``, this module must never grow a non-stdlib import.
"""


def add_stt_skip_llm_judges_arg(parser):
    """Add ``--skip-llm-judges`` (report WER/CER only, no extra LLM judges)."""
    parser.add_argument(
        "--skip-llm-judges",
        action="store_true",
        help=(
            "Skip the extra LLM-based judges (Sarvam intent & entity "
            "preservation, Sarvam LLM-WER/CER, and pipecat-style semantic WER). "
            "They all run by default; passing this reports WER/CER only."
        ),
    )


def add_stt_max_parallel_arg(parser):
    """Add ``--max-parallel`` (providers evaluated concurrently).

    Only the multi-provider entry points expose this — the single-provider eval
    has no across-provider parallelism to tune.
    """
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help=(
            "Number of providers to evaluate in parallel (default: pipeline 1, "
            "direct 2; or $CALIBRATE_STT_MAX_PARALLEL)."
        ),
    )


def add_stt_engine_args(parser):
    """Add ``--engine`` and ``--max-concurrency`` (per-clip transcription knobs)."""
    parser.add_argument(
        "--engine",
        type=str,
        default="pipeline",
        choices=["direct", "pipeline"],
        help=(
            "Transcription engine: 'pipeline' (default; streams through a real "
            "pipecat agent pipeline at real-time pace, also reporting TTFS "
            "latency) or 'direct' (faster; per-provider SDK, no latency)."
        ),
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help=(
            "Concurrent clips per provider, both engines (default: pipeline 1, "
            "direct 4; or $CALIBRATE_STT_MAX_CONCURRENCY). Pipeline defaults to "
            "1 to keep TTFS latency uncontended."
        ),
    )


def add_stt_eval_args(parser, *, include_max_parallel):
    """Add every shared STT flag, in the canonical order the three sites use.

    ``include_max_parallel`` is ``True`` for the multi-provider entry points
    (``cli`` / ``benchmark``) and ``False`` for the single-provider eval.
    """
    add_stt_skip_llm_judges_arg(parser)
    if include_max_parallel:
        add_stt_max_parallel_arg(parser)
    add_stt_engine_args(parser)
