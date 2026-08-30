"""
CLI entry point for calibrate_agent package.

Usage:
    # Interactive mode (recommended):
    calibrate-agent                                        # Main menu
    calibrate-agent stt                                    # Interactive STT evaluation
    calibrate-agent tts                                    # Interactive TTS evaluation
    calibrate-agent llm                                    # Interactive LLM tests
    calibrate-agent simulations                            # Interactive simulations
    calibrate-agent status                                  # Check provider connectivity

    # Direct mode:
    calibrate-agent llm -c config.json -m openai/gpt-4.1 -p openrouter -o ./out
    calibrate-agent simulations --type text -c config.json -m openai/gpt-4.1 -p openrouter -o ./out
    calibrate-agent simulations --type voice -c config.json -o ./out
"""

import sys
import argparse
import asyncio
import runpy
import os
import json
from importlib.metadata import version as get_version
from dotenv import find_dotenv, load_dotenv

# ``calibrate_agent._env`` and ``calibrate_agent._cli_args`` are stdlib-only, so
# these imports are safe at parser-build time — unlike ``calibrate_agent.utils``,
# which pulls in scipy/numpy/pipecat and trips a scipy ``_CopyMode``
# incompatibility in the voice path if imported here.
from calibrate_agent._env import env_int
from calibrate_agent._cli_args import add_stt_eval_args


def _args_to_argv(args, exclude_keys=None, flag_mapping=None):
    """Convert argparse namespace to sys.argv format.

    Args:
        args: argparse.Namespace object
        exclude_keys: set of keys to exclude from conversion
        flag_mapping: dict mapping attribute names to their original flag names
                     (e.g., {'debug_count': '--debug_count', 'input_dir': '--input-dir'})
    """
    exclude_keys = exclude_keys or set()
    flag_mapping = flag_mapping or {}
    argv = []

    for key, value in vars(args).items():
        if key in exclude_keys or value is None:
            continue

        # Use mapping if available, otherwise convert underscores to hyphens
        if key in flag_mapping:
            flag = flag_mapping[key]
        else:
            # Default: convert underscores to hyphens (for flags like --input-dir)
            flag = f"--{key.replace('_', '-')}"

        if isinstance(value, bool):
            if value:  # Only add flag if True
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])

    return argv


def _load_cli_dotenv() -> None:
    """Load .env from the directory where the calibrate-agent command is run."""
    dotenv_path = find_dotenv(usecwd=True)
    load_dotenv(dotenv_path, override=True)


def _launch_ink_ui(mode: str):
    """Launch the bundled Ink UI for interactive TTS/STT evaluation."""
    import shutil
    from pathlib import Path

    node_bin = shutil.which("node")
    if not node_bin:
        print(
            f"Error: Node.js is required for the interactive {mode.upper()} UI.\n"
            "Install it from https://nodejs.org/ or via your package manager."
        )
        sys.exit(1)

    bundle_path = Path(__file__).parent / "ui" / "cli.bundle.mjs"
    if not bundle_path.exists():
        print(
            f"Error: UI bundle not found at {bundle_path}\n"
            "Run 'cd ui && npm run bundle' to build it."
        )
        sys.exit(1)

    import subprocess

    result = subprocess.run([node_bin, str(bundle_path), mode])
    sys.exit(result.returncode)


def _print_sample_output(result: dict) -> None:
    """Print sample_output from a verify result, if present."""
    sample = result.get("sample_output")
    if sample is None:
        return
    print("\n  Sample output received from agent:")
    if isinstance(sample, dict):
        if sample.get("response") is not None:
            print(f"  response: {sample['response']}")
        if sample.get("tool_calls"):
            print(f"  tool_calls: {json.dumps(sample['tool_calls'], indent=2)}")
    else:
        print(f"  {sample}")


def _run_agent_verify(
    agent_url: str,
    agent_headers_raw: str | None,
    models: list[str] | None = None,
    agent_inputs_raw: str | None = None,
    agent_type: str | None = None,
) -> None:
    """Verify an external agent connection and print the result."""
    from calibrate_agent.connections import DEFAULT_AGENT_TYPE, TextAgentConnection

    headers = None
    if agent_headers_raw:
        try:
            headers = json.loads(agent_headers_raw)
        except json.JSONDecodeError:
            print("✗ --agent-headers is not valid JSON")
            sys.exit(1)

    default_inputs = None
    if agent_inputs_raw:
        try:
            default_inputs = json.loads(agent_inputs_raw)
        except json.JSONDecodeError:
            print("✗ --agent-inputs is not valid JSON")
            sys.exit(1)

    try:
        agent = TextAgentConnection(
            url=agent_url,
            headers=headers,
            default_inputs=default_inputs,
            type=agent_type or DEFAULT_AGENT_TYPE,
        )
    except ValueError as e:
        print(f"✗ --agent-type: {e}")
        sys.exit(1)

    # If models provided, send the first model name in the verify request
    model_hint: str | None = models[0] if models else None

    body_preview = json.dumps(
        agent.build_body(
            [{"role": "user", "content": "Hi"}], model=model_hint
        )
    )

    print(f"\nVerifying agent connection: {agent_url}")
    print(f"Sending: {body_preview}")
    print("─" * 60)

    result = asyncio.run(agent.verify(model=model_hint))

    if result["ok"]:
        print("✓ Connection verified — response format is correct")
        _print_sample_output(result)
    else:
        print(f"✗ Verification failed: {result['error']}")
        _print_sample_output(result)
        sys.exit(1)


def main():
    """Main CLI entry point that dispatches to component-specific scripts."""
    from calibrate_agent.connections import AGENT_TYPES, DEFAULT_AGENT_TYPE

    # Load environment variables from .env file
    _load_cli_dotenv()

    parser = argparse.ArgumentParser(
        prog="calibrate-agent",
        usage="calibrate-agent [-h] [-v] {stt,tts,llm,simulations,general,status} ...",
        description="Voice agent evaluation and benchmarking toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    calibrate-agent                                        # Main menu (interactive)
    calibrate-agent stt                                    # Interactive STT evaluation
    calibrate-agent tts                                    # Interactive TTS evaluation
    calibrate-agent llm                                    # Interactive LLM tests
    calibrate-agent llm -c config.json                     # Run LLM tests directly
    calibrate-agent simulations                            # Interactive simulations
    calibrate-agent simulations --type text -c config.json # Run text simulation directly
    calibrate-agent general --dataset data.json -c config.json  # Score input/output pairs
    calibrate-agent status                                 # Check provider connectivity
        """,
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {get_version('calibrate-agent')}",
    )

    subparsers = parser.add_subparsers(
        dest="component",
        help="Component to run",
        metavar="{stt,tts,llm,simulations,general,status}",
    )
    subparsers.required = False  # Allow `calibrate-agent` alone for main menu

    # ── STT ───────────────────────────────────────────────────────
    # `calibrate-agent stt` with no args → interactive UI
    # `calibrate-agent stt -p provider1 provider2 ... -i input-dir ...` → run benchmark (multi) or eval (single)
    stt_parser = subparsers.add_parser(
        "stt",
        help="Speech-to-text evaluation",
    )
    stt_parser.add_argument(
        "-p",
        "--provider",
        type=str,
        nargs="+",
        help="STT provider(s) to evaluate (space-separated for multiple)",
    )
    stt_parser.add_argument("-l", "--language", type=str, default="english")
    stt_parser.add_argument("-i", "--input-dir", type=str)
    stt_parser.add_argument("-o", "--output-dir", type=str, default="./out")
    stt_parser.add_argument("-f", "--input-file-name", type=str, default="stt.csv")
    stt_parser.add_argument("-d", "--debug", action="store_true")
    stt_parser.add_argument("-dc", "--debug_count", type=int, default=5)
    stt_parser.add_argument("--ignore_retry", action="store_true")
    stt_parser.add_argument("--overwrite", action="store_true")
    stt_parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Generate leaderboard after evaluation (for single provider)",
    )
    stt_parser.add_argument("-s", "--save-dir", type=str)
    stt_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to optional JSON config file with judge settings (model, prompt)",
    )
    stt_parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip STT inference and run evaluators directly on a dataset of (gt, pred) pairs",
    )
    stt_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSON (list of {id, gt, pred}). Required with --eval-only.",
    )
    add_stt_eval_args(stt_parser, include_max_parallel=True)

    # ── TTS ───────────────────────────────────────────────────────
    # `calibrate-agent tts` with no args → interactive UI
    # `calibrate-agent tts -p provider -i input ...` → single provider (eval.py)
    # `calibrate-agent tts -p provider1 provider2 -i input ...` → multi-provider (benchmark.py)
    tts_parser = subparsers.add_parser(
        "tts",
        help="Text-to-speech evaluation",
    )
    tts_parser.add_argument(
        "-p",
        "--provider",
        type=str,
        nargs="+",
        help="TTS provider(s) to use for evaluation (space-separated for multiple)",
    )
    tts_parser.add_argument("-l", "--language", type=str, default="english")
    tts_parser.add_argument("-i", "--input", type=str)
    tts_parser.add_argument("-o", "--output-dir", type=str, default="./out")
    tts_parser.add_argument("-d", "--debug", action="store_true")
    tts_parser.add_argument("-dc", "--debug_count", type=int, default=5)
    tts_parser.add_argument("--overwrite", action="store_true")
    tts_parser.add_argument(
        "--max-parallel",
        type=int,
        default=env_int("CALIBRATE_TTS_MAX_PARALLEL", 2),
        help=(
            "Number of providers to evaluate in parallel (default: 2, or "
            "$CALIBRATE_TTS_MAX_PARALLEL)."
        ),
    )
    tts_parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Generate leaderboard after evaluation (for single provider)",
    )
    tts_parser.add_argument("-s", "--save-dir", type=str)
    tts_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to optional JSON config file with judge settings (model, prompt)",
    )
    tts_parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip synthesis and run the audio judge on a prior run's audio (see --dataset)",
    )
    tts_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Prior TTS run directory (e.g. ./out/run/openai) whose results.csv "
            "is read directly. Required with --eval-only."
        ),
    )

    # ── LLM tests ───────────────────────────────────────────────
    # `calibrate-agent llm` with no args → interactive UI
    # `calibrate-agent llm -c config.json -m model ...` → single model (run_tests.py)
    # `calibrate-agent llm -c config.json -m model1 model2 ...` → multi-model (benchmark.py)
    # `calibrate-agent llm --verify --agent-url URL` → verify external agent connection
    llm_parser = subparsers.add_parser(
        "llm",
        help="LLM evaluation — test agent responses and tool calls",
    )
    llm_parser.add_argument(
        "-c", "--config", type=str, default=None, help="Path to test config JSON file"
    )
    llm_parser.add_argument(
        "-o", "--output-dir", type=str, default="./out", help="Output directory"
    )
    llm_parser.add_argument(
        "-m",
        "--model",
        type=str,
        nargs="+",
        help="Model(s) to use for evaluation (space-separated for multiple)",
    )
    llm_parser.add_argument(
        "-p",
        "--provider",
        type=str,
        default="openrouter",
        choices=["openai", "openrouter"],
        help="LLM provider",
    )
    llm_parser.add_argument(
        "-n",
        "--parallel",
        type=int,
        default=None,
        help="Number of test cases to evaluate in parallel per model",
    )
    llm_parser.add_argument(
        "--max-parallel",
        type=int,
        default=env_int("CALIBRATE_LLM_MAX_PARALLEL", 2),
        help=(
            "Number of models to evaluate in parallel (default: 2, or "
            "$CALIBRATE_LLM_MAX_PARALLEL)."
        ),
    )
    llm_parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Debug mode: evaluate only the first N test cases (see --debug_count)",
    )
    llm_parser.add_argument(
        "-dc",
        "--debug_count",
        type=int,
        default=5,
        help="Number of test cases to evaluate in debug mode (default: 5)",
    )
    llm_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force a clean run instead of resuming completed test cases from a prior results.json",
    )
    llm_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an external agent connection by sending a preset message and checking the response format",
    )
    llm_parser.add_argument(
        "--agent-url",
        type=str,
        default=None,
        help="External agent endpoint URL (required with --verify)",
    )
    llm_parser.add_argument(
        "--agent-headers",
        type=str,
        default=None,
        help='HTTP headers for the agent as a JSON string, e.g. \'{"Authorization": "Bearer sk-..."}\'',
    )
    llm_parser.add_argument(
        "--agent-inputs",
        type=str,
        default=None,
        help='Extra input fields sent with the verify request as a JSON string, e.g. \'{"condition_area": "cardiology"}\'',
    )
    llm_parser.add_argument(
        "--agent-type",
        type=str,
        default=DEFAULT_AGENT_TYPE,
        choices=list(AGENT_TYPES),
        help="What the agent expects in the request body: 'conversation' sends the exchange so far as {\"messages\": [...]}, 'general' sends only the latest user text as {\"input\": \"...\"}",
    )
    llm_parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip agent connection verification (used internally when already verified)",
    )
    llm_parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip LLM inference and run evaluators on a dataset of (test_case, output) pairs",
    )
    llm_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSON for --eval-only (list of {test_case, output} items)",
    )
    # ── Simulations ─────────────────────────────────────────────
    # `calibrate-agent simulations` with no args → interactive UI
    # `calibrate-agent simulations --type text -c config.json ...` → run directly
    # `calibrate-agent simulations --verify --agent-url URL` → verify external agent
    sim_parser = subparsers.add_parser(
        "simulations",
        help="Run text or voice simulations",
    )
    sim_parser.add_argument(
        "-t",
        "--type",
        type=str,
        default=None,
        choices=["text", "voice"],
        help="Simulation type: text or voice",
    )
    sim_parser.add_argument(
        "-c", "--config", type=str, default=None, help="Path to simulation config JSON"
    )
    sim_parser.add_argument(
        "-o", "--output-dir", type=str, default="./out", help="Output directory"
    )
    sim_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model name (text simulations)",
    )
    sim_parser.add_argument(
        "-p",
        "--provider",
        type=str,
        default="openrouter",
        choices=["openai", "openrouter"],
        help="LLM provider (text simulations)",
    )
    sim_parser.add_argument(
        "-n",
        "--parallel",
        type=int,
        default=1,
        help="Number of simulations to run in parallel",
    )
    sim_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an external agent connection by sending a preset message and checking the response format",
    )
    sim_parser.add_argument(
        "--agent-url",
        type=str,
        default=None,
        help="External agent endpoint URL (required with --verify)",
    )
    sim_parser.add_argument(
        "--agent-headers",
        type=str,
        default=None,
        help='HTTP headers for the agent as a JSON string, e.g. \'{"Authorization": "Bearer sk-..."}\'',
    )
    sim_parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip agent connection verification (used internally when already verified)",
    )
    sim_parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip simulation and run evaluators on a dataset of pre-existing transcripts (text simulation only)",
    )
    sim_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSON for --eval-only (list of {conversation_history, name?})",
    )

    # Hidden internal subcommand for simulation leaderboard
    sim_subparsers = sim_parser.add_subparsers(dest="sim_subcmd", metavar="")
    sim_lb_parser = sim_subparsers.add_parser("leaderboard")
    sim_lb_parser.add_argument("-o", "--output-dir", type=str, required=True)
    sim_lb_parser.add_argument("-s", "--save-dir", type=str, required=True)

    # ── General task eval ───────────────────────────────────────
    # `calibrate-agent general --dataset data.json --config config.json` →
    # score a dataset of {id, input, output} rows with the general
    # (non-conversational) task judge.
    general_parser = subparsers.add_parser(
        "general",
        help="General task evaluation — judge arbitrary input/output pairs",
    )
    general_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSON (list of {id, input, output})",
    )
    general_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file defining the `evaluators` list",
    )
    general_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./out",
        help="Output directory",
    )

    # ── Status ────────────────────────────────────────────────────
    status_parser = subparsers.add_parser(
        "status",
        help="Check API key configuration and provider connectivity",
    )
    status_parser.add_argument(
        "--table",
        action="store_true",
        default=False,
        help="Display results as a formatted table instead of JSON",
    )

    # ── Agent test (hidden — interactive voice testing) ─────────
    agent_parser = subparsers.add_parser("agent")
    agent_subparsers = agent_parser.add_subparsers(dest="command", help="Agent command")
    agent_subparsers.required = True

    agent_test_parser = agent_subparsers.add_parser(
        "test", help="Run interactive agent test"
    )
    agent_test_parser.add_argument("-c", "--config", type=str, required=True)
    agent_test_parser.add_argument("-o", "--output-dir", type=str, default="./out")

    # ─────────────────────────────────────────────────────────────
    args = parser.parse_args()

    # No component specified → launch main menu UI
    if args.component is None:
        _launch_ink_ui("menu")

    # ── Dispatch ────────────────────────────────────────────────
    if args.component == "stt":
        # eval-only: skip STT inference and run evaluators on a (gt, pred) dataset.
        # benchmark: --provider given → run inference + evaluators.
        # neither: launch interactive UI.
        if args.eval_only:
            from calibrate_agent.stt.benchmark import main as stt_benchmark_main

            if not args.dataset:
                print("\033[31mError: --dataset is required with --eval-only\033[0m")
                sys.exit(1)

            argv = ["calibrate-agent", "--eval-only", "--dataset", args.dataset]
            argv.extend(["-l", args.language])
            argv.extend(["-o", args.output_dir])
            if args.config:
                argv.extend(["--config", args.config])
            if args.skip_llm_judges:
                argv.append("--skip-llm-judges")

            sys.argv = argv
            asyncio.run(stt_benchmark_main())
        elif args.provider is not None:
            from calibrate_agent.stt.benchmark import main as stt_benchmark_main

            providers = args.provider
            argv = ["calibrate-agent", "-p"] + providers
            argv.extend(["-l", args.language])
            argv.extend(["-i", args.input_dir])
            argv.extend(["-o", args.output_dir])
            argv.extend(["-f", args.input_file_name])
            if args.debug:
                argv.append("-d")
            argv.extend(["-dc", str(args.debug_count)])
            if args.ignore_retry:
                argv.append("--ignore_retry")
            if args.overwrite:
                argv.append("--overwrite")
            if args.save_dir:
                argv.extend(["-s", args.save_dir])
            if args.config:
                argv.extend(["--config", args.config])
            if args.skip_llm_judges:
                argv.append("--skip-llm-judges")
            # Only forward the parallelism knobs when the user set them
            # explicitly; otherwise let the benchmark resolve the per-engine
            # default (pipeline 1/1, direct 2/4).
            if args.max_parallel is not None:
                argv.extend(["--max-parallel", str(args.max_parallel)])
            if args.max_concurrency is not None:
                argv.extend(["--max-concurrency", str(args.max_concurrency)])
            argv.extend(["--engine", args.engine])

            sys.argv = argv
            asyncio.run(stt_benchmark_main())
        else:
            _launch_ink_ui("stt")

    elif args.component == "tts":
        # eval-only: skip synthesis and run the audio judge on a dataset.
        # provider given → run synthesis + evaluators.
        # neither: launch interactive UI.
        if args.eval_only:
            from calibrate_agent.tts.benchmark import main as tts_benchmark_main

            if not args.dataset:
                print("\033[31mError: --dataset is required with --eval-only\033[0m")
                sys.exit(1)

            argv = ["calibrate-agent", "--eval-only", "--dataset", args.dataset]
            argv.extend(["-o", args.output_dir])
            if args.config:
                argv.extend(["--config", args.config])

            sys.argv = argv
            asyncio.run(tts_benchmark_main())
        elif args.provider is not None:
            from calibrate_agent.tts.benchmark import main as tts_benchmark_main

            providers = args.provider
            argv = ["calibrate-agent", "-p"] + providers
            argv.extend(["-l", args.language])
            argv.extend(["-i", args.input])
            argv.extend(["-o", args.output_dir])
            if args.debug:
                argv.append("-d")
            argv.extend(["-dc", str(args.debug_count)])
            if args.overwrite:
                argv.append("--overwrite")
            if args.config:
                argv.extend(["--config", args.config])
            argv.extend(["--max-parallel", str(args.max_parallel)])

            sys.argv = argv
            asyncio.run(tts_benchmark_main())
        else:
            _launch_ink_ui("tts")

    elif args.component == "llm":
        if getattr(args, "eval_only", False):
            if args.config is None:
                print("Error: --config is required with --eval-only")
                sys.exit(1)
            if not getattr(args, "dataset", None):
                print("Error: --dataset is required with --eval-only")
                sys.exit(1)

            from calibrate_agent.llm.run_tests import main as llm_run_tests_main

            argv = [
                "calibrate-agent",
                "-c",
                args.config,
                "-o",
                args.output_dir,
                "--eval-only",
                "--dataset",
                args.dataset,
            ]
            if getattr(args, "parallel", None) is not None:
                argv.extend(["-n", str(args.parallel)])
            if getattr(args, "debug", False):
                argv.append("-d")
                argv.extend(["-dc", str(args.debug_count)])
            sys.argv = argv
            asyncio.run(llm_run_tests_main())
        elif getattr(args, "verify", False):
            if not args.agent_url:
                print("Error: --agent-url is required with --verify")
                sys.exit(1)
            _run_agent_verify(
                args.agent_url,
                args.agent_headers,
                models=args.model,
                agent_inputs_raw=args.agent_inputs,
                agent_type=args.agent_type,
            )
        elif args.config is None:
            # No config → interactive mode
            _launch_ink_ui("llm")
        else:
            # Direct mode: run tests with provided config
            import json as _json
            from calibrate_agent.utils import apply_debug_limit

            with open(args.config) as _f:
                _config = _json.load(_f)

            if getattr(args, "debug", False) and _config.get("test_cases"):
                _config["test_cases"] = apply_debug_limit(
                    _config["test_cases"], True, args.debug_count
                )

            if _config.get("agent_url"):
                # Agent connection path
                from calibrate_agent.connections import TextAgentConnection
                from calibrate_agent.llm import tests as _tests

                try:
                    _agent = TextAgentConnection(
                        url=_config["agent_url"],
                        headers=_config.get("agent_headers"),
                        default_inputs=_config.get("agent_default_inputs"),
                        type=_config.get("agent_type", DEFAULT_AGENT_TYPE),
                    )
                except ValueError as _e:
                    print(f"✗ config agent_type: {_e}")
                    sys.exit(1)
                _models = args.model if args.model else []

                # Verify once per model (skip if already verified upstream e.g. interactive UI)
                if not getattr(args, "skip_verify", False):
                    _models_to_verify = _models if _models else [None]
                    for _m in _models_to_verify:
                        _label = f"model: {_m}" if _m else "connection"
                        print(f"\nVerifying agent {_label}: {_config['agent_url']}")
                        _verify_result = asyncio.run(_agent.verify(model=_m))
                        if not _verify_result["ok"]:
                            print(f"✗ Verification failed: {_verify_result['error']}")
                            _print_sample_output(_verify_result)
                            sys.exit(1)
                        print(f"✓ Verified")
                        _print_sample_output(_verify_result)
                        print()

                from calibrate_agent.llm.tests_leaderboard import generate_leaderboard
                from calibrate_agent.llm._output import print_benchmark_summary
                from calibrate_agent.llm.run_tests import exit_if_run_failed

                # Run — all models together (tests.run fans them out in parallel)
                if _models:
                    print(f"\n\033[92m{'='*60}\033[0m")
                    print(f"\033[92m  Models: {', '.join(_models)}\033[0m")
                    print(f"\033[92m{'='*60}\033[0m\n")
                    _results = asyncio.run(
                        _tests.run(
                            agent=_agent,
                            test_cases=_config["test_cases"],
                            output_dir=args.output_dir,
                            models=_models,
                            evaluators=_config.get("evaluators"),
                            max_parallel=args.max_parallel,
                            test_parallel=args.parallel,
                            overwrite=args.overwrite,
                        )
                    )
                    _model_results = {
                        _m: _results.get(_m, _results) for _m in _models
                    }

                    _lb_dir = os.path.join(args.output_dir, "leaderboard")
                    generate_leaderboard(output_dir=args.output_dir, save_dir=_lb_dir)
                    _has_errors = print_benchmark_summary(
                        models=_models,
                        model_results=_model_results,
                        leaderboard_dir=_lb_dir,
                    )
                    if _has_errors:
                        sys.exit(1)
                    exit_if_run_failed(_model_results.values())
                else:
                    _run_result = asyncio.run(
                        _tests.run(
                            agent=_agent,
                            test_cases=_config["test_cases"],
                            output_dir=args.output_dir,
                            evaluators=_config.get("evaluators"),
                            test_parallel=args.parallel,
                            overwrite=args.overwrite,
                        )
                    )
                    exit_if_run_failed([_run_result])
            else:
                from calibrate_agent.llm.benchmark import main as llm_benchmark_main

                models = args.model if args.model else ["gpt-4.1"]

                argv = ["calibrate-agent", "-c", args.config]
                argv.extend(["-o", args.output_dir])
                argv.extend(["-m"] + models)
                argv.extend(["-p", args.provider])
                if getattr(args, "parallel", None) is not None:
                    argv.extend(["-n", str(args.parallel)])
                argv.extend(["--max-parallel", str(args.max_parallel)])
                if getattr(args, "overwrite", False):
                    argv.append("--overwrite")
                if getattr(args, "debug", False):
                    argv.append("-d")
                    argv.extend(["-dc", str(args.debug_count)])

                sys.argv = argv
                asyncio.run(llm_benchmark_main())

    elif args.component == "simulations":
        if getattr(args, "verify", False):
            if not args.agent_url:
                print("Error: --agent-url is required with --verify")
                sys.exit(1)
            _model_str = getattr(args, "model", None)
            _run_agent_verify(
                args.agent_url,
                args.agent_headers,
                models=[_model_str] if _model_str else None,
            )
        # Hidden leaderboard subcommand (used by Ink UI)
        elif getattr(args, "sim_subcmd", None) == "leaderboard":
            from calibrate_agent.llm.simulation_leaderboard import (
                main as leaderboard_main,
            )

            sys.argv = ["calibrate-agent"] + _args_to_argv(
                args,
                exclude_keys={
                    "component",
                    "sim_subcmd",
                    "type",
                    "config",
                    "model",
                    "provider",
                    "parallel",
                    "port",
                },
            )
            leaderboard_main()
        elif args.type is None or args.config is None:
            # Missing type or config → interactive mode
            _launch_ink_ui("simulations")
        elif args.type == "text":
            from calibrate_agent.llm.run_simulation import main as llm_simulation_main

            # Eval-only: skip agent verification entirely; the dataset already
            # contains the transcripts and we only run evaluators on them.
            if getattr(args, "eval_only", False):
                if not getattr(args, "dataset", None):
                    print("Error: --dataset is required with --eval-only")
                    sys.exit(1)
                sys.argv = ["calibrate-agent"] + _args_to_argv(
                    args,
                    exclude_keys={
                        "component",
                        "sim_subcmd",
                        "type",
                        "skip_verify",
                    },
                )
                asyncio.run(llm_simulation_main())
                return

            # Pre-verify agent connection if config has agent_url
            if args.config:
                import json as _json

                with open(args.config) as _f:
                    _sim_config = _json.load(_f)
                if _sim_config.get("agent_type", DEFAULT_AGENT_TYPE) != DEFAULT_AGENT_TYPE:
                    print(
                        "✗ Text simulation is not supported for an agent of type "
                        f"'{_sim_config['agent_type']}'. The simulator sends the "
                        "conversation so far each turn, so the agent must be of "
                        "type 'conversation'."
                    )
                    sys.exit(1)
                if _sim_config.get("agent_default_inputs"):
                    print(
                        "✗ Text simulation is not supported for an agent configured "
                        "with custom inputs (agent_default_inputs). The simulator "
                        "sends only conversation messages each turn."
                    )
                    sys.exit(1)
                if _sim_config.get("agent_url") and not getattr(
                    args, "skip_verify", False
                ):
                    from calibrate_agent.connections import TextAgentConnection

                    _sim_agent = TextAgentConnection(
                        url=_sim_config["agent_url"],
                        headers=_sim_config.get("agent_headers"),
                    )
                    print(f"\nVerifying agent connection: {_sim_config['agent_url']}")
                    _verify = asyncio.run(_sim_agent.verify())
                    if not _verify["ok"]:
                        print(f"✗ Verification failed: {_verify['error']}")
                        _print_sample_output(_verify)
                        sys.exit(1)
                    print("✓ Verified")
                    _print_sample_output(_verify)
                    print()

            sys.argv = ["calibrate-agent"] + _args_to_argv(
                args, exclude_keys={"component", "sim_subcmd", "type", "skip_verify"}
            )
            asyncio.run(llm_simulation_main())
        elif args.type == "voice":
            from calibrate_agent.agent.run_simulation import main as agent_main

            # Pre-verify external WebSocket voice agent if config has a
            # ws:// or wss:// agent_url.
            if args.config:
                import json as _json
                from urllib.parse import urlparse as _urlparse

                with open(args.config) as _f:
                    _voice_config = _json.load(_f)
                _voice_url = _voice_config.get("agent_url")
                if (
                    _voice_url
                    and _urlparse(_voice_url).scheme in ("ws", "wss")
                    and not getattr(args, "skip_verify", False)
                ):
                    from calibrate_agent.connections import WebSocketAgentConnection

                    _voice_agent = WebSocketAgentConnection(url=_voice_url)
                    print(f"\nVerifying agent connection: {_voice_url}")
                    _verify = asyncio.run(_voice_agent.verify())
                    if not _verify["ok"]:
                        print(f"✗ Verification failed: {_verify['error']}")
                        sys.exit(1)
                    print("✓ Verified")
                    print()

            sys.argv = ["calibrate-agent"] + _args_to_argv(
                args,
                exclude_keys={
                    "component",
                    "sim_subcmd",
                    "type",
                    "model",
                    "provider",
                    "verify",
                    "skip_verify",
                    "agent_url",
                    "agent_headers",
                    "eval_only",
                    "dataset",
                },
            )
            asyncio.run(agent_main())

    elif args.component == "general":
        from calibrate_agent.general.eval import main as general_eval_main

        if not args.dataset:
            print("\033[31mError: --dataset is required\033[0m")
            sys.exit(1)
        if not args.config:
            print("\033[31mError: --config is required\033[0m")
            sys.exit(1)

        argv = ["calibrate-agent", "--dataset", args.dataset, "-c", args.config]
        argv.extend(["-o", args.output_dir])
        sys.argv = argv
        asyncio.run(general_eval_main())

    elif args.component == "status":
        from calibrate_agent.status import run_status_live

        table_mode = getattr(args, "table", False)
        asyncio.run(run_status_live(table=table_mode))

    elif args.component == "agent":
        if args.command == "test":
            test_args = _args_to_argv(args, exclude_keys={"component", "command"})
            test_args = [
                arg.replace("--output-dir", "--output_dir") for arg in test_args
            ]

            test_module_path = os.path.join(
                os.path.dirname(__file__), "agent", "test.py"
            )
            sys.argv = ["calibrate-agent-test"] + test_args
            runpy.run_path(test_module_path, run_name="__main__")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
