"""Remote execution entry point for Modal containers.

This module is imported (not serialized) by Modal in the container,
which decouples local and remote Python versions. Keep imports
minimal at module level — heavy imports happen inside the function.
"""

from __future__ import annotations


def run_profile(
    source: str,
    script_rel: str,
    total_steps: int,
    timeout: int,
    args: list[str] | None,
    overlay_files: dict[str, str] | None = None,
) -> str:
    """Entry point called by Modal. Delegates to _remote_execute."""
    from profine.modal.executor import _remote_execute
    return _remote_execute(source, script_rel, total_steps, timeout, args, overlay_files)
