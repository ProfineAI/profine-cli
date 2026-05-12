"""profine CLI — run individual optimization tools.

Usage:
    profine read train.py
    profine profile train.py --hardware 1x_a100
    profine interpret --profile-dir output/
    profine suggest --profile-dir output/
    profine edit train.py --suggestion-dir output/ --optimization torch_compile
    profine benchmark train.py --optimized edited_train.py --hardware 1x_a100
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

# Force-kill on Ctrl+C — httpx blocks normal signal handling
signal.signal(signal.SIGINT, lambda *_: os._exit(1))
from pathlib import Path

from profine.cli.commands import (
    cmd_read,
    cmd_profile,
    cmd_interpret,
    cmd_suggest,
    cmd_edit,
    cmd_benchmark,
    cmd_run_all,
)
from profine.cli.errors import is_debug_mode, print_user_error
from profine.config.settings import DEFAULTS

# Commands that require an LLM backend
_LLM_COMMANDS = {"read", "profile", "interpret", "suggest", "edit", "benchmark", "run-all"}


def build_parser() -> argparse.ArgumentParser:
    # Shared flags live in a parent parser so they work before OR after the
    # subcommand (e.g. `profine read train.py -o out` and
    # `profine -o out read train.py` both work).
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--provider", default="openai",
                        choices=["openai", "anthropic", "local"],
                        help="LLM provider: 'openai', 'anthropic', or 'local' (OpenAI-compatible local server)")
    shared.add_argument("--api-key", default=None, help="API key override")
    shared.add_argument("--model", default=None, help="Model name override (required for --provider local)")
    shared.add_argument("--base-url", default=None,
                        help="OpenAI-compatible endpoint URL (for --provider local; defaults to "
                             "http://localhost:11434/v1 for Ollama). Env: PROFINE_LOCAL_BASE_URL")
    shared.add_argument("--output", "-o", default="profine_output", help="Output directory")
    shared.add_argument("--prefs", default=None, help="Path to user preferences markdown")

    parser = argparse.ArgumentParser(
        prog="profine",
        description="Agentic ML Training Optimizer",
        parents=[shared],
    )

    sub = parser.add_subparsers(dest="command", help="Tool to run")

    # read
    p_read = sub.add_parser("read", help="Read and analyze a training script", parents=[shared], conflict_handler="resolve")
    p_read.add_argument("script", help="Path to the training script")

    # profile
    p_profile = sub.add_parser("profile", help="Profile a training script on Modal", parents=[shared], conflict_handler="resolve")
    p_profile.add_argument("script", help="Path to the training script")
    p_profile.add_argument("--hardware", default=DEFAULTS.default_hardware, help="Hardware preset")
    p_profile.add_argument("--steps", type=int, default=DEFAULTS.default_steps, help="Total steps")
    p_profile.add_argument("--warmup", type=int, default=DEFAULTS.default_warmup_steps, help="Warmup steps")
    p_profile.add_argument("--timeout", type=int, default=DEFAULTS.default_modal_timeout,
                           help=f"Modal container timeout in seconds (default: {DEFAULTS.default_modal_timeout})")
    p_profile.add_argument("--warmstart", action="store_true", help="Reuse deployed Modal app between runs")

    # interpret
    p_interpret = sub.add_parser("interpret", help="Interpret a profile into bottlenecks", parents=[shared], conflict_handler="resolve")
    p_interpret.add_argument("--profile-dir", required=True, help="Directory with profile output")

    # suggest
    p_suggest = sub.add_parser("suggest", help="Suggest optimizations", parents=[shared], conflict_handler="resolve")
    p_suggest.add_argument("--interpret-dir", required=True, help="Directory with interpret output (bottleneck_report.json)")
    p_suggest.add_argument("--arch-dir", default=None, help="Directory with architecture_record.json (default: auto-detect)")
    p_suggest.add_argument("--profile-dir", default=None, help="Directory with profile output (default: auto-detect)")

    # edit
    p_edit = sub.add_parser("edit", help="Apply an optimization to the source", parents=[shared], conflict_handler="resolve")
    p_edit.add_argument("script", nargs="?", default=None, help="Path to the training script (auto-detected from prior steps if omitted)")
    p_edit.add_argument("--suggestion-dir", required=True, help="Directory with suggestion output")
    p_edit.add_argument("--optimization", default=None, help="Optimization ID to apply (default: top-ranked)")
    p_edit.add_argument("--top", type=int, default=None,
                        help="Apply the top N ranked optimizations sequentially, "
                             "each stacked on the previous edit. Cumulative output "
                             "lands at <output>/edit/ for `profine benchmark`; "
                             "per-step artifacts go in <output>/edit/NN_<entry_id>/.")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Benchmark original vs. optimized", parents=[shared], conflict_handler="resolve")
    p_bench.add_argument("script", nargs="?", default=None, help="Path to the original training script (auto-detected from prior steps if omitted)")
    p_bench.add_argument("--optimized", default=None, help="Path to the optimized script (default: <output>/edit/edited_train.py)")
    p_bench.add_argument("--hardware", default=DEFAULTS.default_hardware, help="Hardware preset")
    p_bench.add_argument("--steps", type=int, default=DEFAULTS.default_steps, help="Total steps")
    p_bench.add_argument("--warmup", type=int, default=DEFAULTS.default_warmup_steps, help="Warmup steps")
    p_bench.add_argument("--rtol", type=float, default=1e-2, help="Loss rtol")
    p_bench.add_argument("--atol", type=float, default=1e-4, help="Loss atol")
    p_bench.add_argument("--timeout", type=int, default=DEFAULTS.default_modal_timeout,
                         help=f"Modal container timeout in seconds (default: {DEFAULTS.default_modal_timeout})")
    p_bench.add_argument("--warmstart", action="store_true", help="Reuse deployed Modal app between runs")
    p_bench.add_argument("--edit-dir", default=None,
                          help="Directory of editor output (default: <output>/edit). "
                               "Extra files under <edit-dir>/files/ are overlaid onto the "
                               "Modal workspace before the optimized run, so multi-file "
                               "edits actually take effect.")

    # run-all
    p_all = sub.add_parser("run-all", help="Run the full pipeline: read → profile → interpret → suggest → edit → benchmark",
                           parents=[shared], conflict_handler="resolve")
    p_all.add_argument("script", help="Path to the training script")
    p_all.add_argument("--hardware", default=DEFAULTS.default_hardware, help="Hardware preset")
    p_all.add_argument("--steps", type=int, default=DEFAULTS.default_steps, help="Total steps")
    p_all.add_argument("--warmup", type=int, default=DEFAULTS.default_warmup_steps, help="Warmup steps")
    p_all.add_argument("--timeout", type=int, default=DEFAULTS.default_modal_timeout,
                       help=f"Modal container timeout in seconds (default: {DEFAULTS.default_modal_timeout})")
    p_all.add_argument("--warmstart", action="store_true", help="Reuse deployed Modal app between runs")
    p_all.add_argument("--top", type=int, default=None,
                       help="Apply top N optimizations (default: all ranked candidates)")
    p_all.add_argument("--rtol", type=float, default=1e-2, help="Loss relative tolerance for correctness")
    p_all.add_argument("--atol", type=float, default=1e-4, help="Loss absolute tolerance for correctness")

    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # Load user preferences if provided
    user_prefs: str | None = None
    if args.prefs:
        user_prefs = Path(args.prefs).read_text(encoding="utf-8")

    output_dir = Path(args.output)

    dispatch = {
        "read": cmd_read,
        "profile": cmd_profile,
        "interpret": cmd_interpret,
        "suggest": cmd_suggest,
        "edit": cmd_edit,
        "benchmark": cmd_benchmark,
        "run-all": cmd_run_all,
    }

    # Check for API key before running any LLM command. The "local" provider talks
    # to an OpenAI-compatible local server (Ollama, vLLM, LM Studio, ...) — no key
    # required, but --model is.
    if args.command in _LLM_COMMANDS:
        provider = args.provider
        if provider == "local":
            if not args.model:
                print("Error: --provider local requires --model.\n")
                print("Examples:")
                print("  profine --provider local --model llama3.1:8b <command>            # Ollama")
                print("  profine --provider local --model meta-llama/Llama-3.1-8B-Instruct \\")
                print("          --base-url http://localhost:8000/v1 <command>             # vLLM")
                return 1
        elif not args.api_key:
            if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
                print("Error: No OpenAI API key found.\n")
                print("Set it with one of:")
                print("  export OPENAI_API_KEY=sk-...")
                print("  python -m profine --api-key sk-... <command>")
                print("\nOr switch to Anthropic (--provider anthropic) or a local LLM (--provider local).")
                return 1
            if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
                print("Error: No Anthropic API key found.\n")
                print("Set it with one of:")
                print("  export ANTHROPIC_API_KEY=sk-ant-...")
                print("  python -m profine --api-key sk-ant-... <command>")
                print("\nOr switch to OpenAI (--provider openai) or a local LLM (--provider local).")
                return 1

    handler = dispatch[args.command]
    try:
        return handler(args, output_dir, user_prefs)
    except Exception as exc:
        if is_debug_mode():
            raise
        code = print_user_error(exc)
        if code < 0:
            # Unknown failure — let the traceback through so it doesn't get hidden.
            raise
        return code


if __name__ == "__main__":
    sys.exit(main())
