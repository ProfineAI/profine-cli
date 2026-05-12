"""Styled CLI output helpers backed by `rich`.

We keep this module tiny on purpose: the rest of the CLI calls semantic helpers
(`success`, `error`, `warn`, `path`, ...) and this module decides the styling.
Falls back to plain text when stdout isn't a TTY (e.g. when piped to `tee` for
a log file) so log artifacts stay clean.

Conventions:
  - cyan/bold       — section headers and stage banners
  - green           — success / completion / "ship it"
  - red             — failures and regressions
  - yellow          — warnings and skips
  - magenta         — file paths the user might want to open
  - dim             — secondary metadata (counts, durations, etc.)
"""

from __future__ import annotations

import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Force-no-color when not a TTY OR when NO_COLOR is set (https://no-color.org/).
# `force_terminal=False` lets rich auto-detect; we explicitly disable color only
# when the user has redirected stdout, which keeps `tee run.log` files readable.
_console = Console(
    no_color=not sys.stdout.isatty() or bool(os.environ.get("NO_COLOR")),
    highlight=False,
    soft_wrap=True,
)


def header(text: str, *, step: str | None = None) -> None:
    """Banner for a major pipeline stage (read / profile / interpret / ...)."""
    title = Text(text, style="bold cyan")
    if step:
        title = Text.assemble((f"  [{step}] ", "bold cyan dim"), (text, "bold cyan"))
    _console.print()
    _console.print(Panel(title, border_style="cyan", padding=(0, 2)))


def status(text: str) -> None:
    """Inline progress line (e.g. 'Reading nanoGPT/train.py...')."""
    _console.print(text)


def success(text: str) -> None:
    _console.print(f"[bold green]✓[/bold green] {text}")


def error(text: str) -> None:
    _console.print(f"[bold red]✗[/bold red] {text}", style="red")


def warn(text: str) -> None:
    _console.print(f"[bold yellow]⚠[/bold yellow]  {text}")


def info(text: str) -> None:
    _console.print(f"[bold blue]ℹ[/bold blue]  {text}")


def path(label: str, p: str | object) -> None:
    """`<label>: <path>` line with the path highlighted."""
    _console.print(f"  {label}: [magenta]{p}[/magenta]")


def dim(text: str) -> None:
    _console.print(f"[dim]{text}[/dim]")


def rule(text: str = "") -> None:
    _console.rule(f"[cyan]{text}[/cyan]" if text else "")


def plain(text: str = "") -> None:
    """Pass-through for content we don't want to style (e.g. LLM markdown excerpts)."""
    _console.print(text)
