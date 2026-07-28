"""Shared argparse builders for the STT/TTS eval/benchmark CLIs.

The ``--skip-llm-judges``, ``--judges``, ``--max-parallel``, ``--engine`` and
``--max-concurrency`` flags are accepted by three entry points that must stay in
lock-step: the top-level ``calibrate-agent stt`` parser (``calibrate_agent.cli``),
the multi-provider benchmark (``calibrate_agent.stt.benchmark``) and the
single-provider eval (``calibrate_agent.stt.eval``). ``--yes`` is accepted by a
wider set of entry points — the top-level ``stt``/``tts`` parsers plus both
their benchmarks — since both eval kinds show a judge cost estimate before
running. Defining each flag once here keeps their help text, defaults and
choices from drifting apart.

Deliberately stdlib-only (imports nothing): ``calibrate_agent.cli`` builds its
parser by calling these, and the CLI must build its parser *without* importing
the heavy ``calibrate_agent.utils`` module — that shifts scipy/numpy init order
and trips a scipy ``_CopyMode`` incompatibility in the voice path. Like
``calibrate_agent._env``, this module must never grow a non-stdlib import.
"""

STT_LLM_JUDGES = ("intent", "llm_wer", "semantic_wer")
DEFAULT_STT_LLM_JUDGES = frozenset(STT_LLM_JUDGES)


def parse_stt_llm_judges(value: str) -> frozenset[str]:
    """Parse a comma-separated ``--judges`` value into a frozenset of names.

    Raises ``ValueError`` on an empty list or unknown name so argparse
    surfaces a clear usage error (argparse converts ``ValueError`` from a
    ``type=`` callable into an argument error).
    """
    names = {part.strip() for part in value.split(",") if part.strip()}
    if not names:
        raise ValueError(
            "expected a comma-separated list of judge names; "
            f"choose from: {', '.join(STT_LLM_JUDGES)}"
        )
    unknown = names - DEFAULT_STT_LLM_JUDGES
    if unknown:
        raise ValueError(
            f"unknown judge name(s): {', '.join(sorted(unknown))}; "
            f"choose from: {', '.join(STT_LLM_JUDGES)}"
        )
    return frozenset(names)


def resolve_stt_llm_judges(
    *, skip_llm_judges: bool, judges: frozenset[str] | None
) -> frozenset[str]:
    """Resolve which built-in STT LLM judges to run.

    ``--skip-llm-judges`` and ``--judges`` are mutually exclusive. Omitting
    both returns all three judges (today's default). Passing
    ``--skip-llm-judges`` returns an empty set. Passing ``--judges`` returns
    that subset.
    """
    if skip_llm_judges and judges is not None:
        raise SystemExit(
            "error: --judges and --skip-llm-judges are mutually exclusive"
        )
    if skip_llm_judges:
        return frozenset()
    if judges is None:
        return DEFAULT_STT_LLM_JUDGES
    return judges


def add_stt_skip_llm_judges_arg(parser):
    """Add ``--skip-llm-judges`` (skip all three built-in LLM judges)."""
    parser.add_argument(
        "--skip-llm-judges",
        action="store_true",
        help=(
            "Skip all three built-in LLM judges (Sarvam intent & entity "
            "preservation, Sarvam LLM-WER/CER, and pipecat-style semantic WER). "
            "Mutually exclusive with --judges. Config evaluators are unaffected."
        ),
    )


def add_stt_judges_arg(parser):
    """Add ``--judges`` (opt-in subset of the three built-in LLM judges)."""
    parser.add_argument(
        "--judges",
        type=parse_stt_llm_judges,
        default=None,
        metavar="NAMES",
        help=(
            "Comma-separated subset of built-in LLM judges to run: "
            "intent, llm_wer, semantic_wer. Omit to run all three. "
            "Mutually exclusive with --skip-llm-judges. Config evaluators "
            "are unaffected."
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


def add_assume_yes_arg(parser):
    """Add ``--yes`` (skip the judge cost confirmation prompt)."""
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help=(
            "Proceed with the LLM-as-judge run without the interactive "
            "cost-confirmation prompt."
        ),
    )


def add_stt_eval_args(parser, *, include_max_parallel):
    """Add every shared STT flag, in the canonical order the three sites use.

    ``include_max_parallel`` is ``True`` for the multi-provider entry points
    (``cli`` / ``benchmark``) and ``False`` for the single-provider eval.
    """
    add_stt_skip_llm_judges_arg(parser)
    add_stt_judges_arg(parser)
    if include_max_parallel:
        add_stt_max_parallel_arg(parser)
    add_stt_engine_args(parser)
