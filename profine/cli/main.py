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
    cmd_telemetry,
    cmd_env,
)
from profine.cli.errors import is_debug_mode, print_user_error
from profine.config.settings import DEFAULTS

_LLM_COMMANDS = {"read", "profile", "interpret", "suggest", "edit", "benchmark", "run-all"}


def _ensure_utf8_stdout() -> None:
    """Reconfigure stdout/stderr to UTF-8 so non-ASCII help text (arrows,
    em-dashes) doesn't crash on Windows consoles defaulting to CP1252.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _add_shared(p: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Attach the shared flags to `p`.

    `suppress=True` makes the argparse defaults SUPPRESS, so the action does
    not write to the namespace when the user didn't pass the flag. We use
    that for subparsers so that a flag set at the top level (e.g.
    `profine --provider local read x`) isn't clobbered when the subparser
    parses into its fresh namespace and copies attrs back. The top-level
    parser keeps real defaults so the value is always present after parse.
    """
    NONE = argparse.SUPPRESS if suppress else None
    OUT = argparse.SUPPRESS if suppress else "profine_output"
    PROV = argparse.SUPPRESS if suppress else "openai"
    p.add_argument("--provider", default=PROV,
                   choices=["openai", "anthropic", "local"],
                   help="LLM provider: 'openai', 'anthropic', or 'local' (OpenAI-compatible local server)")
    p.add_argument("--api-key", default=NONE, help="API key override")
    p.add_argument("--model", default=NONE, help="Model name override (required for --provider local)")
    p.add_argument("--base-url", default=NONE,
                   help="OpenAI-compatible endpoint URL (for --provider local; defaults to "
                        "http://localhost:11434/v1 for Ollama). Env: PROFINE_LOCAL_BASE_URL")
    p.add_argument("--seed", type=int, default=NONE,
                   help="Seed for the LLM provider (best-effort; OpenAI honors it, Anthropic "
                        "ignores it and relies on temperature=0). Use to make optimization "
                        "rankings reproducible across runs.")
    p.add_argument("--output", "-o", default=OUT, help="Output directory")
    p.add_argument("--prefs", default=NONE, help="Path to user preferences markdown")
    # Anonymous telemetry opt-out. The first interactive run will prompt
    # for consent if neither this flag nor PROFINE_NO_TELEMETRY is set;
    # paying customers' opt-out is server-side, this flag is the OSS lever.
    if suppress:
        p.add_argument("--no-telemetry", action="store_true", default=argparse.SUPPRESS,
                       help="Disable anonymous telemetry for this invocation")
    else:
        p.add_argument("--no-telemetry", action="store_true", default=False,
                       help="Disable anonymous telemetry for this invocation")


def build_parser() -> argparse.ArgumentParser:
    # Shared flags work before OR after the subcommand. Subparsers use
    # SUPPRESS defaults so they don't clobber values set at the top level.
    shared_suppress = argparse.ArgumentParser(add_help=False)
    _add_shared(shared_suppress, suppress=True)

    parser = argparse.ArgumentParser(
        prog="profine",
        description="Agentic ML Training Optimizer",
    )
    _add_shared(parser, suppress=False)

    sub = parser.add_subparsers(dest="command", help="Tool to run")
    shared = shared_suppress

    p_read = sub.add_parser("read", help="Read and analyze a training script", parents=[shared], conflict_handler="resolve")
    p_read.add_argument("script", help="Path to the training script")

    p_profile = sub.add_parser("profile", help="Profile a training script on Modal", parents=[shared], conflict_handler="resolve")
    p_profile.add_argument("script", help="Path to the training script")
    p_profile.add_argument("--hardware", default=DEFAULTS.default_hardware, help="Hardware preset")
    p_profile.add_argument("--steps", type=int, default=DEFAULTS.default_steps, help="Total steps")
    p_profile.add_argument("--warmup", type=int, default=DEFAULTS.default_warmup_steps, help="Warmup steps")
    p_profile.add_argument("--timeout", type=int, default=DEFAULTS.default_modal_timeout,
                           help=f"Modal container timeout in seconds (default: {DEFAULTS.default_modal_timeout})")
    p_profile.add_argument("--warmstart", action="store_true", help="Reuse deployed Modal app between runs")

    p_interpret = sub.add_parser("interpret", help="Interpret a profile into bottlenecks", parents=[shared], conflict_handler="resolve")
    p_interpret.add_argument("--profile-dir", required=True, help="Directory with profile output")

    p_suggest = sub.add_parser("suggest", help="Suggest optimizations", parents=[shared], conflict_handler="resolve")
    p_suggest.add_argument("--interpret-dir", required=True, help="Directory with interpret output (bottleneck_report.json)")
    p_suggest.add_argument("--arch-dir", default=None, help="Directory with architecture_record.json (default: auto-detect)")
    p_suggest.add_argument("--profile-dir", default=None, help="Directory with profile output (default: auto-detect)")

    p_edit = sub.add_parser("edit", help="Apply an optimization to the source", parents=[shared], conflict_handler="resolve")
    p_edit.add_argument("script", nargs="?", default=None, help="Path to the training script (auto-detected from prior steps if omitted)")
    p_edit.add_argument("--suggestion-dir", required=True, help="Directory with suggestion output")
    p_edit.add_argument("--optimization", default=None, help="Optimization ID to apply (default: top-ranked)")
    p_edit.add_argument("--top", type=int, default=None,
                        help="Apply the top N ranked optimizations sequentially, "
                             "each stacked on the previous edit. Cumulative output "
                             "lands at <output>/edit/ for `profine benchmark`; "
                             "per-step artifacts go in <output>/edit/NN_<entry_id>/.")

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

    # telemetry — view/toggle anonymous data collection
    p_telem = sub.add_parser(
        "telemetry",
        help="Manage anonymous telemetry consent",
        parents=[shared],
        conflict_handler="resolve",
    )
    p_telem.add_argument(
        "action",
        choices=["status", "enable", "disable"],
        help="status: show current state; enable/disable: change OSS consent",
    )

    # env — print all PROFINE_* env vars with their resolved values
    sub.add_parser(
        "env",
        help="Show every PROFINE_* env var profine reads (with current values)",
        parents=[shared],
        conflict_handler="resolve",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdout()
    try:
        # When profine is installed as a console script, find_dotenv()'s default
        # (usecwd=False) walks up from the entry-script's directory — e.g.
        # ~/anaconda3/bin/, which never has a .env. Force the search to start at
        # the user's cwd so `.env` in their project root is actually loaded.
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

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
        "telemetry": cmd_telemetry,
        "env": cmd_env,
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
