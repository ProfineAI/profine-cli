"""Modal execution — ephemeral and warmstart paths.

Runs an instrumented profiler script on a Modal GPU and returns
the raw results payload.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from profine.modal.app_registry import (
    build_deployment_signature,
    resolve_app_name,
    save_deployment_cache,
    signature_matches,
)
from profine.modal.discovery import (
    discover_dependencies,
    discover_project_root,
    discover_python_version,
    discover_system_packages,
)
from profine.modal.image_builder import ModalImageBuilder, WORKSPACE_MOUNT
from profine.profiler.hooks import RESULTS_SENTINEL
from profine.schema.hardware import HardwareConfig, ModalRuntimeConfig

from profine.config.yaml_loader import get_transient_error_patterns

_TRANSIENT_ERROR_PATTERNS = get_transient_error_patterns()
_MAX_TRANSIENT_RETRIES = 3


@dataclass(slots=True)
class ExecutionResult:
    """Raw result from Modal execution."""
    payload: dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    success: bool = False
    error: str | None = None
    runtime_seconds: float = 0.0


def _ensure_utf8_stdout() -> None:
    """Reconfigure local stdout/stderr to UTF-8 on Windows.

    Modal streams container output to the local terminal. If the
    container prints Unicode (e.g. ✓ from transformers), this fails
    on Windows with charmap encoding.
    """
    import sys
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


class ModalExecutor:
    """Runs profiler scripts on Modal GPUs."""

    def __init__(
        self,
        modal_config: ModalRuntimeConfig | None = None,
        hf_token: str | None = None,
    ) -> None:
        self._config = modal_config or ModalRuntimeConfig()
        self._hf_token = hf_token
        self._image_builder = ModalImageBuilder(self._config)

    def execute(
        self,
        *,
        instrumented_source: str,
        script_path: Path,
        hardware: HardwareConfig,
        total_steps: int = 60,
        script_args: list[str] | None = None,
        dependencies: list[str] | None = None,
        overlay_files: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute an instrumented script on Modal.

        Args:
            instrumented_source: The profiler-instrumented Python source.
            script_path: Original script path (for project root discovery).
            hardware: Target GPU hardware.
            total_steps: Authoritative step limit (overrides whatever the LLM wrote).
            script_args: Optional CLI arguments for the script.
            dependencies: Pre-computed pip dependencies. If None, discovers from script.

        Returns:
            ExecutionResult with parsed payload or error info.
        """
        _ensure_utf8_stdout()
        try:
            import modal
        except ImportError:
            return ExecutionResult(
                success=False,
                error="modal package not installed. Install with: pip install modal",
            )

        project_root = discover_project_root(script_path)
        if dependencies is None:
            dependencies = discover_dependencies(script_path)
        python_version = discover_python_version(script_path) or self._config.python_version
        system_packages = discover_system_packages(dependencies)

        print(f"  [modal] project_root: {project_root}")
        print(f"  [modal] dependencies: {dependencies}")
        print(f"  [modal] python: {python_version}")

        build_flash = hardware.flash_attention_supported and _needs_flash_attention(dependencies)

        image = self._image_builder.build(
            modal,
            project_root=project_root,
            dependencies=dependencies,
            system_packages=system_packages,
            python_version=python_version,
            hardware=hardware,
            build_flash_attention=build_flash,
            hf_token=self._hf_token,
        )

        volumes = self._image_builder.build_volumes(modal)

        cls_kwargs = self._image_builder.build_cls_kwargs(
            hardware=hardware,
            volumes=volumes,
        )

        if self._config.enable_warmstart:
            return self._execute_warmstart(
                modal, image, cls_kwargs,
                instrumented_source=instrumented_source,
                script_path=script_path,
                project_root=project_root,
                hardware=hardware,
                dependencies=dependencies,
                python_version=python_version,
                total_steps=total_steps,
                script_args=script_args,
                overlay_files=overlay_files,
            )
        return self._execute_ephemeral(
            modal, image, cls_kwargs,
            instrumented_source=instrumented_source,
            script_path=script_path,
            project_root=project_root,
            total_steps=total_steps,
            script_args=script_args,
            overlay_files=overlay_files,
        )

    def _execute_ephemeral(
        self,
        modal: Any,
        image: Any,
        cls_kwargs: dict,
        *,
        instrumented_source: str,
        script_path: Path,
        project_root: Path,
        total_steps: int,
        script_args: list[str] | None,
        overlay_files: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Fresh Modal app for each run."""
        from profine.modal.remote import run_profile

        app = modal.App(name="profine-profile-ephemeral", image=image)
        rel_path = script_path.resolve().relative_to(project_root.resolve()).as_posix()

        remote_fn = app.function(**cls_kwargs)(run_profile)

        start = time.monotonic()
        try:
            with modal.enable_output():
                with app.run():
                    raw_output = remote_fn.remote(instrumented_source, rel_path, total_steps, self._config.timeout_seconds, script_args, overlay_files)
            return _parse_output(raw_output, time.monotonic() - start)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                runtime_seconds=time.monotonic() - start,
            )

    def _execute_warmstart(
        self,
        modal: Any,
        image: Any,
        cls_kwargs: dict,
        *,
        instrumented_source: str,
        script_path: Path,
        project_root: Path,
        hardware: HardwareConfig,
        dependencies: list[str],
        python_version: str,
        total_steps: int,
        script_args: list[str] | None,
        overlay_files: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Reuse a deployed Modal app if the signature matches."""
        from profine.modal.remote import run_profile

        app_name = resolve_app_name("profine", project_root, script_path, hardware)
        current_sig = build_deployment_signature(
            project_root=project_root,
            dependencies=dependencies,
            python_version=python_version,
            hardware=hardware,
        )

        rel_path = script_path.resolve().relative_to(project_root.resolve()).as_posix()

        if signature_matches(project_root, app_name, current_sig):
            try:
                remote_fn = modal.Function.from_name(app_name, "run_profile")
                start = time.monotonic()
                raw_output = remote_fn.remote(instrumented_source, rel_path, total_steps, self._config.timeout_seconds, script_args, overlay_files)
                return _parse_output(raw_output, time.monotonic() - start)
            except Exception:
                pass  # Fall through to redeploy

        app = modal.App(name=app_name, image=image)
        remote_fn = app.function(**cls_kwargs)(run_profile)

        start = time.monotonic()
        try:
            app.deploy(name=app_name)
            save_deployment_cache(project_root, app_name, current_sig)

            raw_output = remote_fn.remote(instrumented_source, rel_path, total_steps, self._config.timeout_seconds, script_args, overlay_files)
            return _parse_output(raw_output, time.monotonic() - start)
        except Exception as e:
            return self._execute_ephemeral(
                modal, image, cls_kwargs,
                instrumented_source=instrumented_source,
                script_path=script_path,
                project_root=project_root,
                total_steps=total_steps,
                script_args=script_args,
                overlay_files=overlay_files,
            )


def _remote_execute(
    source: str,
    script_rel_path: str,
    total_steps: int,
    timeout: int,
    args: list[str] | None,
    overlay_files: dict[str, str] | None = None,
) -> str:
    """Runs inside the Modal container. Executes the instrumented script."""
    import os
    import signal
    import sys
    import runpy

    # Set authoritative step limit so the StepController uses this
    # regardless of what the LLM wrote in install_hooks()
    os.environ["PROFINE_TOTAL_STEPS"] = str(total_steps)

    # Force acc_events=True on the torch profiler so CUDA times survive
    # the schedule cycle — the LLM healer tends to drop this flag.
    import torch.profiler
    _orig_profile_init = torch.profiler.profile.__init__

    def _patched_profile_init(self_prof, *a, **kw):
        kw.setdefault("acc_events", True)
        _orig_profile_init(self_prof, *a, **kw)

    torch.profiler.profile.__init__ = _patched_profile_init

    # Wall-clock watchdog: Modal's container timeout + 60s buffer
    _WATCHDOG_SECONDS = timeout + 60

    def _watchdog_handler(signum, frame):
        raise TimeoutError(
            f"Script exceeded {_WATCHDOG_SECONDS}s wall-clock limit "
            "(likely stuck in data download or model init)"
        )

    signal.signal(signal.SIGALRM, _watchdog_handler)
    signal.alarm(_WATCHDOG_SECONDS)

    work_dir = os.path.join(WORKSPACE_MOUNT, os.path.dirname(script_rel_path))
    os.makedirs(work_dir, exist_ok=True)

    # Apply overlay files BEFORE writing the script, so that any
    # local-package edits are in
    # place when the script's imports resolve. Paths are project-
    # relative; we materialize them under the workspace mount.
    if overlay_files:
        import importlib
        for rel, content in overlay_files.items():
            # Defense in depth: refuse absolute or escaping paths so a
            # malformed editor result can't write outside /workspace.
            norm = os.path.normpath(rel.lstrip("/"))
            if norm.startswith("..") or os.path.isabs(norm):
                continue
            target = os.path.join(WORKSPACE_MOUNT, norm)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w") as ovf:
                ovf.write(content)
        # If any already-imported module was overlaid, drop its cached
        # bytecode so the next import picks up the new source.
        importlib.invalidate_caches()

    script_name = os.path.basename(script_rel_path)
    script_full = os.path.join(work_dir, f"_profine_profile_{script_name}")
    with open(script_full, "w") as f:
        f.write(source)

    # Set up sys.argv and make local packages importable. runpy.run_path
    # only adds the script's own dir; scripts that live in subprojects
    # need the workspace root on sys.path too.
    sys.argv = [script_full] + (args or [])
    os.chdir(work_dir)
    for p in (work_dir, WORKSPACE_MOUNT):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Force UTF-8 stdout to avoid charmap encoding errors from
    # Unicode characters (e.g. ✓) in library output
    import io
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    captured = io.StringIO()
    old_stdout = sys.stdout
    # Tee: write to both captured buffer and the original stdout.
    class Tee:
        def write(self, data):
            captured.write(data)
            try:
                old_stdout.write(data)
            except UnicodeEncodeError:
                old_stdout.write(data.encode("utf-8", errors="replace").decode("ascii", errors="replace"))
        def flush(self):
            captured.flush()
            old_stdout.flush()
        def isatty(self):
            return False
        def __getattr__(self, name):
            return getattr(old_stdout, name)

    sys.stdout = Tee()
    try:
        runpy.run_path(script_full, run_name="__main__")
    except Exception as e:
        if "StepLimitReached" in type(e).__name__:
            pass
        elif isinstance(e, TimeoutError):
            # Salvage partial profiling data collected before timeout
            _emit_partial_results(captured, status="timeout", error=str(e))
        else:
            # Wrap with the full traceback so the orchestrator's
            # heal LLM sees file/line/exception type, not just str(e).
            import traceback
            tb = traceback.format_exc()
            raise RuntimeError(f"Script crashed:\n{tb}") from e
    finally:
        signal.alarm(0)  # Cancel watchdog
        sys.stdout = old_stdout

    return captured.getvalue()


def _emit_partial_results(captured: Any, *, status: str, error: str) -> None:
    """Write whatever profiling data was collected to the capture stream.

    Called when execution is interrupted (timeout, etc.) so partial
    step times, GPU samples, and memory data aren't lost.
    """
    try:
        from profine.profiler.hooks import get_active_context, RESULTS_SENTINEL
        ctx = get_active_context()
        if ctx is None:
            return
        results = ctx.results()
        results["status"] = status
        results["error"] = error
        captured.write(RESULTS_SENTINEL + json.dumps(results, default=str) + "\n")
    except Exception:
        pass  # Best-effort — don't mask the original error


def _parse_output(raw_output: str, runtime_seconds: float) -> ExecutionResult:
    """Extract the results JSON from the script's stdout.

    Uses the LAST sentinel match — the instrumented script may emit
    partial results first (emit_results) then a complete payload
    with profiler_events appended.
    """
    payload = None
    for line in raw_output.splitlines():
        if RESULTS_SENTINEL in line:
            json_str = line.split(RESULTS_SENTINEL, 1)[1]
            try:
                payload = json.loads(json_str)
            except json.JSONDecodeError:
                pass

    if payload is not None:
        payload["runtime_seconds"] = runtime_seconds
        return ExecutionResult(
            payload=payload,
            stdout=raw_output,
            success=True,
            runtime_seconds=runtime_seconds,
        )

    return ExecutionResult(
        stdout=raw_output,
        success=False,
        error="No results sentinel found in output. Script may have crashed.",
        runtime_seconds=runtime_seconds,
    )


def _needs_flash_attention(dependencies: list[str]) -> bool:
    """Check if flash-attn is in the dependency list."""
    return any("flash-attn" in d.lower() or "flash_attn" in d.lower() for d in dependencies)
