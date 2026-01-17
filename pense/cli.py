"""
CLI entry point for pense package.

Usage:
    # New style (recommended) - with subcommands:
    pense stt eval -p deepgram -l english -i ./data -o ./out
    pense stt leaderboard -o ./out -s ./leaderboard
    pense tts eval -p openai -i ./data -o ./out
    pense tts leaderboard -o ./out -s ./leaderboard
    pense llm tests run -c ./config.json -o ./out
    pense llm tests leaderboard -o ./out -s ./leaderboard
    pense llm simulations run -c ./config.json -o ./out
    pense llm simulations leaderboard -o ./out -s ./leaderboard
    pense agent simulation -c ./config.json -o ./out
"""

import sys
import argparse
import asyncio
import runpy
import os
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


def main():
    """Main CLI entry point that dispatches to component-specific scripts."""
    # Load environment variables from .env file
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        prog="pense",
        description="Voice agent evaluation and benchmarking toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    pense stt eval -p deepgram -l english -i ./data -o ./out
    pense stt leaderboard -o ./out -s ./leaderboard
    pense tts eval -p openai -i ./data -o ./out
    pense llm tests run -c ./config.json -o ./out
    pense llm tests leaderboard -o ./out -s ./leaderboard
    pense llm simulations run -c ./config.json -o ./out
    pense llm simulations leaderboard -o ./out -s ./leaderboard
    pense agent test -c ./config.json -o ./out
    pense agent simulation -c ./config.json -o ./out
        """,
    )

    subparsers = parser.add_subparsers(dest="component", help="Component to run")
    subparsers.required = True

    # STT subcommands
    stt_parser = subparsers.add_parser("stt", help="Speech-to-text evaluation")
    stt_subparsers = stt_parser.add_subparsers(dest="command", help="STT command")
    stt_subparsers.required = True

    stt_eval_parser = stt_subparsers.add_parser("eval", help="Run STT evaluation")
    stt_eval_parser.add_argument("-p", "--provider", type=str, required=True)
    stt_eval_parser.add_argument("-l", "--language", type=str, default="english")
    stt_eval_parser.add_argument("-i", "--input-dir", type=str, required=True)
    stt_eval_parser.add_argument("-o", "--output-dir", type=str, default="./out")
    stt_eval_parser.add_argument("-f", "--input-file-name", type=str, default="stt.csv")
    stt_eval_parser.add_argument("-d", "--debug", action="store_true")
    stt_eval_parser.add_argument("-dc", "--debug_count", type=int, default=5)
    stt_eval_parser.add_argument("--ignore_retry", action="store_true")
    stt_eval_parser.add_argument("--port", type=int, default=8765)

    stt_leaderboard_parser = stt_subparsers.add_parser(
        "leaderboard", help="Generate STT leaderboard"
    )
    stt_leaderboard_parser.add_argument("-o", "--output-dir", type=str, required=True)
    stt_leaderboard_parser.add_argument("-s", "--save-dir", type=str, required=True)

    # TTS subcommands
    tts_parser = subparsers.add_parser("tts", help="Text-to-speech evaluation")
    tts_subparsers = tts_parser.add_subparsers(dest="command", help="TTS command")
    tts_subparsers.required = True

    tts_eval_parser = tts_subparsers.add_parser("eval", help="Run TTS evaluation")
    tts_eval_parser.add_argument("-p", "--provider", type=str, default="smallest")
    tts_eval_parser.add_argument("-l", "--language", type=str, default="english")
    tts_eval_parser.add_argument("-i", "--input", type=str, required=True)
    tts_eval_parser.add_argument("-o", "--output-dir", type=str, default="./out")
    tts_eval_parser.add_argument("-d", "--debug", action="store_true")
    tts_eval_parser.add_argument("-dc", "--debug_count", type=int, default=5)
    tts_eval_parser.add_argument("--port", type=int, default=8765)

    tts_leaderboard_parser = tts_subparsers.add_parser(
        "leaderboard", help="Generate TTS leaderboard"
    )
    tts_leaderboard_parser.add_argument("-o", "--output-dir", type=str, required=True)
    tts_leaderboard_parser.add_argument("-s", "--save-dir", type=str, required=True)

    # LLM subcommands
    llm_parser = subparsers.add_parser("llm", help="LLM evaluation")
    llm_subparsers = llm_parser.add_subparsers(dest="llm_type", help="LLM type")
    llm_subparsers.required = True

    # LLM tests subcommands
    llm_tests_parser = llm_subparsers.add_parser("tests", help="LLM tests")
    llm_tests_subparsers = llm_tests_parser.add_subparsers(
        dest="command", help="Tests command"
    )
    llm_tests_subparsers.required = True

    llm_tests_run_parser = llm_tests_subparsers.add_parser("run", help="Run LLM tests")
    llm_tests_run_parser.add_argument(
        "-c", "--config", type=str, default="examples/tests.json"
    )
    llm_tests_run_parser.add_argument("-o", "--output-dir", type=str, default="./out")
    llm_tests_run_parser.add_argument("-m", "--model", type=str, default="gpt-4.1")
    llm_tests_run_parser.add_argument(
        "-p", "--provider", type=str, default="openrouter"
    )

    llm_tests_leaderboard_parser = llm_tests_subparsers.add_parser(
        "leaderboard", help="Generate LLM tests leaderboard"
    )
    llm_tests_leaderboard_parser.add_argument(
        "-o", "--output-dir", type=str, required=True
    )
    llm_tests_leaderboard_parser.add_argument(
        "-s", "--save-dir", type=str, required=True
    )

    # LLM simulations subcommands
    llm_simulations_parser = llm_subparsers.add_parser(
        "simulations", help="LLM simulations"
    )
    llm_simulations_subparsers = llm_simulations_parser.add_subparsers(
        dest="command", help="Simulations command"
    )
    llm_simulations_subparsers.required = True

    llm_simulations_run_parser = llm_simulations_subparsers.add_parser(
        "run", help="Run LLM simulation"
    )
    llm_simulations_run_parser.add_argument("-c", "--config", type=str, required=True)
    llm_simulations_run_parser.add_argument(
        "-o", "--output-dir", type=str, default="./out"
    )
    llm_simulations_run_parser.add_argument(
        "-m", "--model", type=str, default="gpt-4.1"
    )
    llm_simulations_run_parser.add_argument(
        "-p", "--provider", type=str, default="openrouter"
    )
    llm_simulations_run_parser.add_argument(
        "-n",
        "--parallel",
        type=int,
        default=1,
        help="Number of simulations to run in parallel",
    )

    llm_simulations_leaderboard_parser = llm_simulations_subparsers.add_parser(
        "leaderboard", help="Generate LLM simulation leaderboard"
    )
    llm_simulations_leaderboard_parser.add_argument(
        "-o", "--output-dir", type=str, required=True
    )
    llm_simulations_leaderboard_parser.add_argument(
        "-s", "--save-dir", type=str, required=True
    )

    # Agent subcommands
    agent_parser = subparsers.add_parser("agent", help="Agent simulation")
    agent_subparsers = agent_parser.add_subparsers(dest="command", help="Agent command")
    agent_subparsers.required = True

    agent_test_parser = agent_subparsers.add_parser(
        "test", help="Run interactive agent test"
    )
    agent_test_parser.add_argument("-c", "--config", type=str, required=True)
    agent_test_parser.add_argument("-o", "--output-dir", type=str, default="./out")

    agent_simulation_parser = agent_subparsers.add_parser(
        "simulation", help="Run agent simulation"
    )
    agent_simulation_parser.add_argument("-c", "--config", type=str, required=True)
    agent_simulation_parser.add_argument(
        "-o", "--output-dir", type=str, default="./out"
    )
    agent_simulation_parser.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()

    # Dispatch to the appropriate module
    # Note: sys.argv[0] should be just the program name, not include subcommands
    # The submodule parsers will parse the remaining arguments
    if args.component == "stt":
        if args.command == "eval":
            from pense.stt.eval import main as stt_main

            # Map attribute names to their original flag names (preserve underscores)
            flag_mapping = {
                "debug_count": "--debug_count",
                "ignore_retry": "--ignore_retry",
            }
            sys.argv = ["pense"] + _args_to_argv(
                args, exclude_keys={"component", "command"}, flag_mapping=flag_mapping
            )
            asyncio.run(stt_main())
        elif args.command == "leaderboard":
            from pense.stt.leaderboard import main as leaderboard_main

            sys.argv = ["pense"] + _args_to_argv(
                args, exclude_keys={"component", "command"}
            )
            leaderboard_main()
    elif args.component == "tts":
        if args.command == "eval":
            from pense.tts.eval import main as tts_main

            # Map attribute names to their original flag names (preserve underscores)
            flag_mapping = {
                "debug_count": "--debug_count",
            }
            sys.argv = ["pense"] + _args_to_argv(
                args, exclude_keys={"component", "command"}, flag_mapping=flag_mapping
            )
            asyncio.run(tts_main())
        elif args.command == "leaderboard":
            from pense.tts.leaderboard import main as leaderboard_main

            sys.argv = ["pense"] + _args_to_argv(
                args, exclude_keys={"component", "command"}
            )
            leaderboard_main()
    elif args.component == "llm":
        if args.llm_type == "tests":
            if args.command == "run":
                from pense.llm.run_tests import main as llm_tests_main

                sys.argv = ["pense"] + _args_to_argv(
                    args, exclude_keys={"component", "llm_type", "command"}
                )
                asyncio.run(llm_tests_main())
            elif args.command == "leaderboard":
                from pense.llm.tests_leaderboard import main as leaderboard_main

                sys.argv = ["pense"] + _args_to_argv(
                    args, exclude_keys={"component", "llm_type", "command"}
                )
                leaderboard_main()
        elif args.llm_type == "simulations":
            if args.command == "run":
                from pense.llm.run_simulation import main as llm_simulation_main

                sys.argv = ["pense"] + _args_to_argv(
                    args, exclude_keys={"component", "llm_type", "command"}
                )
                asyncio.run(llm_simulation_main())
            elif args.command == "leaderboard":
                from pense.llm.simulation_leaderboard import (
                    main as leaderboard_main,
                )

                sys.argv = ["pense"] + _args_to_argv(
                    args, exclude_keys={"component", "llm_type", "command"}
                )
                leaderboard_main()
    elif args.component == "agent":
        if args.command == "test":
            # The test.py script uses pipecat's runner which expects specific sys.argv format
            # We need to convert output-dir to output_dir for the test script
            test_args = _args_to_argv(args, exclude_keys={"component", "command"})
            # Convert --output-dir to --output_dir for test.py compatibility
            test_args = [
                arg.replace("--output-dir", "--output_dir") for arg in test_args
            ]

            test_module_path = os.path.join(
                os.path.dirname(__file__), "agent", "test.py"
            )
            sys.argv = ["pense-agent-test"] + test_args
            runpy.run_path(test_module_path, run_name="__main__")
        elif args.command == "simulation":
            from pense.agent.run_simulation import main as agent_main

            sys.argv = ["pense"] + _args_to_argv(
                args, exclude_keys={"component", "command"}
            )
            asyncio.run(agent_main())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
