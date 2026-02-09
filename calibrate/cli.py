"""
CLI entry point for calibrate package.

Usage:
    # Interactive mode (recommended):
    calibrate                                        # Main menu
    calibrate stt                                    # Interactive STT evaluation
    calibrate tts                                    # Interactive TTS evaluation
    calibrate llm                                    # Interactive LLM tests
    calibrate simulations                            # Interactive simulations
    calibrate status                                  # Check provider connectivity

    # Direct mode:
    calibrate llm -c config.json -m gpt-4.1 -p openrouter -o ./out
    calibrate simulations --type text -c config.json -m gpt-4.1 -p openrouter -o ./out
    calibrate simulations --type voice -c config.json -o ./out
"""

import sys
import argparse
import asyncio
import runpy
import os
import json
from importlib.metadata import version as get_version
from dotenv import load_dotenv


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


def main():
    """Main CLI entry point that dispatches to component-specific scripts."""
    # Load environment variables from .env file
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        prog="calibrate",
        usage="calibrate [-h] [-v] {stt,tts,llm,simulations,status} ...",
        description="Voice agent evaluation and benchmarking toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    calibrate                                        # Main menu (interactive)
    calibrate stt                                    # Interactive STT evaluation
    calibrate tts                                    # Interactive TTS evaluation
    calibrate llm                                    # Interactive LLM tests
    calibrate llm -c config.json                     # Run LLM tests directly
    calibrate simulations                            # Interactive simulations
    calibrate simulations --type text -c config.json # Run text simulation directly
    calibrate status                                 # Check provider connectivity
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
        metavar="{stt,tts,llm,simulations,status}",
    )
    subparsers.required = False  # Allow `calibrate` alone for main menu

    # ── STT ───────────────────────────────────────────────────────
    # `calibrate stt` with no args → interactive UI
    # `calibrate stt -p provider1 provider2 ... -i input-dir ...` → run benchmark (multi) or eval (single)
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

    # ── TTS ───────────────────────────────────────────────────────
    # `calibrate tts` with no args → interactive UI
    # `calibrate tts -p provider -i input ...` → single provider (eval.py)
    # `calibrate tts -p provider1 provider2 -i input ...` → multi-provider (benchmark.py)
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
        "--leaderboard",
        action="store_true",
        help="Generate leaderboard after evaluation (for single provider)",
    )
    tts_parser.add_argument("-s", "--save-dir", type=str)

    # ── LLM tests ───────────────────────────────────────────────
    # `calibrate llm` with no args → interactive UI
    # `calibrate llm -c config.json -m model ...` → single model (run_tests.py)
    # `calibrate llm -c config.json -m model1 model2 ...` → multi-model (benchmark.py)
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
    # ── Simulations ─────────────────────────────────────────────
    # `calibrate simulations` with no args → interactive UI
    # `calibrate simulations --type text -c config.json ...` → run directly
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
        default="gpt-4.1",
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

    # Hidden internal subcommand for simulation leaderboard
    sim_subparsers = sim_parser.add_subparsers(dest="sim_subcmd", metavar="")
    sim_lb_parser = sim_subparsers.add_parser("leaderboard")
    sim_lb_parser.add_argument("-o", "--output-dir", type=str, required=True)
    sim_lb_parser.add_argument("-s", "--save-dir", type=str, required=True)

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
        # If provider is given, run evaluation directly; otherwise launch interactive UI
        if args.provider is not None:
            from calibrate.stt.benchmark import main as stt_benchmark_main

            providers = args.provider
            argv = ["calibrate", "-p"] + providers
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

            sys.argv = argv
            asyncio.run(stt_benchmark_main())
        else:
            _launch_ink_ui("stt")

    elif args.component == "tts":
        # If provider is given, run evaluation directly; otherwise launch interactive UI
        if args.provider is not None:
            from calibrate.tts.benchmark import main as tts_benchmark_main

            providers = args.provider
            argv = ["calibrate", "-p"] + providers
            argv.extend(["-l", args.language])
            argv.extend(["-i", args.input])
            argv.extend(["-o", args.output_dir])
            if args.debug:
                argv.append("-d")
            argv.extend(["-dc", str(args.debug_count)])
            if args.overwrite:
                argv.append("--overwrite")

            sys.argv = argv
            asyncio.run(tts_benchmark_main())
        else:
            _launch_ink_ui("tts")

    elif args.component == "llm":
        if args.config is None:
            # No config → interactive mode
            _launch_ink_ui("llm")
        else:
            # Direct mode: run tests with provided config
            from calibrate.llm.benchmark import main as llm_benchmark_main

            models = (
                args.model if args.model else ["gpt-4.1"]
            )  # Default to gpt-4.1 if not specified

            argv = ["calibrate", "-c", args.config]
            argv.extend(["-o", args.output_dir])
            argv.extend(["-m"] + models)
            argv.extend(["-p", args.provider])

            sys.argv = argv
            asyncio.run(llm_benchmark_main())

    elif args.component == "simulations":
        # Hidden leaderboard subcommand (used by Ink UI)
        if getattr(args, "sim_subcmd", None) == "leaderboard":
            from calibrate.llm.simulation_leaderboard import (
                main as leaderboard_main,
            )

            sys.argv = ["calibrate"] + _args_to_argv(
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
            from calibrate.llm.run_simulation import main as llm_simulation_main

            sys.argv = ["calibrate"] + _args_to_argv(
                args, exclude_keys={"component", "sim_subcmd", "type"}
            )
            asyncio.run(llm_simulation_main())
        elif args.type == "voice":
            from calibrate.agent.run_simulation import main as agent_main

            sys.argv = ["calibrate"] + _args_to_argv(
                args,
                exclude_keys={
                    "component",
                    "sim_subcmd",
                    "type",
                    "model",
                    "provider",
                },
            )
            asyncio.run(agent_main())

    elif args.component == "status":
        from calibrate.status import main as status_main

        table_mode = getattr(args, "table", False)
        result = asyncio.run(status_main(quiet=not table_mode))
        if not table_mode:
            print(json.dumps(result, indent=2))

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
