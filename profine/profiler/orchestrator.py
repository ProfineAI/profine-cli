"""Profile orchestrator — ties everything together.

Usage:
    from profine.profiler.orchestrator import ProfileOrchestrator

    profiler = ProfileOrchestrator(provider="openai")
    result = profiler.profile("train.py", hardware="1x_a100")

    print(result.markdown)
    result.save("output/")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from profine.config.settings import DEFAULTS
from profine.config.yaml_loader import get_import_to_pip, get_known_pypi_toplevel
from profine.llm.backend import create_backend
from profine.profiler.event_collector import (
    compute_scale_factor,
    parse_events_from_payload,
)
from profine.profiler.heuristics import (
    classify_gpu_pattern,
    compute_arithmetic_intensity,
    compute_category_breakdown,
    compute_communication_overhead,
    compute_dataloader_stall_pct,
    compute_gpu_mean,
    compute_memory_headroom,
    compute_phase_breakdown,
    compute_top_kernels,
    detect_attention_impl,
    detect_precision,
)
from profine.profiler.instrumentor import ScriptInstrumentor
from profine.profiler.report import generate_report
from profine.profiler.stabilization import detect_stabilization_point
from profine.reader.extractor import extract
from profine.schema.hardware import HardwareConfig, ModalRuntimeConfig, get_hardware
from profine.schema.profile_record import ProfileRecord


@dataclass(slots=True)
class ProfileResult:
    """Output of the profiler: both human and machine representations."""
    record: ProfileRecord
    markdown: str
    instrumented_source: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        record_path = out / "profile_record.json"
        report_path = out / "profile_report.md"

        record_path.write_text(
            json.dumps(_record_to_dict(self.record), indent=2, default=str),
            encoding="utf-8",
        )
        report_path.write_text(self.markdown, encoding="utf-8")

        return {"record": record_path, "report": report_path}


class ProfileOrchestrator:
    """Main entry point for the Profiler tool."""

    def __init__(
        self,
        provider: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        modal_config: ModalRuntimeConfig | None = None,
        hf_token: str | None = None,
        max_retries: int = 3,
    ) -> None:
        llm_kwargs: dict[str, Any] = {}
        if api_key:
            llm_kwargs["api_key"] = api_key
        if model:
            llm_kwargs["model"] = model
        if base_url:
            llm_kwargs["base_url"] = base_url
        self._backend = create_backend(provider, **llm_kwargs)
        self._instrumentor = ScriptInstrumentor(self._backend)
        self._modal_config = modal_config
        self._hf_token = hf_token
        self._max_retries = max_retries

    def profile(
        self,
        script_path: str | Path,
        *,
        hardware: str | HardwareConfig = DEFAULTS.default_hardware,
        steps: int = DEFAULTS.default_steps,
        warmup_steps: int = DEFAULTS.default_warmup_steps,
        script_args: list[str] | None = None,
    ) -> ProfileResult:
        """Profile a training script on Modal.

        Args:
            script_path: Path to the training script.
            hardware: Hardware preset name or HardwareConfig.
            steps: Total optimizer steps to run.
            warmup_steps: Steps to discard as warmup.
            script_args: Optional CLI arguments for the script.

        Returns:
            ProfileResult with both human and machine outputs.
        """
        path = Path(script_path)
        source = path.read_text(encoding="utf-8")
        hw = get_hardware(hardware) if isinstance(hardware, str) else hardware

        # Step 1: Extract facts
        print("  [1/4] Extracting code facts...")
        facts = extract(source, path.name)

        # Step 2: Instrument the script
        print("  [2/4] Instrumenting script (LLM)...")
        instrumented = self._instrumentor.instrument(
            source=source,
            facts=facts,
            total_steps=steps,
            warmup_steps=warmup_steps,
        )

        # Save instrumented source for debugging
        debug_dir = Path("profine_output") / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "instrumented.py").write_text(instrumented.source, encoding="utf-8")

        # Step 3: Discover dependencies (AST → LLM validation)
        print("  [3/4] Discovering dependencies...")
        from profine.modal.discovery import discover_dependencies, discover_project_root
        from profine.profiler.prompts import build_dependency_validation_prompt

        dependencies = discover_dependencies(path)
        print(f"         AST found: {dependencies}")

        project_root = discover_project_root(path)
        local_names = {
            p.name for p in project_root.iterdir()
            if (p.is_dir() and (p / "__init__.py").exists())
            or (p.suffix == ".py" and p.is_file())
        }

        print("         Validating with LLM...")
        system_prompt, user_prompt = build_dependency_validation_prompt(source, dependencies)
        try:
            raw = self._backend.call(system_prompt, user_prompt)
            extra = _parse_dep_list(raw)
            if extra:
                existing = {d.split(">=")[0].split("==")[0].lower() for d in dependencies}
                added = []
                for dep in extra:
                    base = dep.split(">=")[0].split("==")[0]
                    base_lower = base.lower()
                    if base_lower in existing:
                        continue
                    if base in local_names or base.replace("-", "_") in local_names:
                        continue  # local package, not a PyPI dep
                    dependencies.append(dep)
                    added.append(dep)
                print(f"         LLM additions: {added}" if added
                      else "         LLM additions filtered (all local or duplicate)")
            else:
                print("         LLM confirmed deps are complete")
        except Exception as e:
            print(f"         LLM validation failed ({e}), using AST deps only")

        # Step 4: Execute on Modal
        print("  [4/4] Launching on Modal...")
        from profine.modal.executor import ModalExecutor
        executor = ModalExecutor(
            modal_config=self._modal_config,
            hf_token=self._hf_token,
        )

        result = None
        last_error = ""
        try:
            for attempt in range(self._max_retries):
                print(f"  [attempt {attempt + 1}/{self._max_retries}] Executing on Modal...")
                result = executor.execute(
                    instrumented_source=instrumented.source,
                    script_path=path,
                    hardware=hw,
                    total_steps=steps,
                    script_args=script_args,
                    dependencies=dependencies,
                )

                # Treat a "successful but ran on CPU" run as a recoverable
                # failure — we asked for a GPU host but the script never
                # touched it, so the profile data is meaningless.
                if result.success and _looks_like_cpu_run(result.payload, hw.name):
                    last_error = (
                        "Run completed on CPU despite GPU host. "
                        "Force CUDA device in the trainer/model config."
                    )
                    print(f"  [attempt {attempt + 1}] Ran on CPU silently — healing for device.")
                    if attempt < self._max_retries - 1:
                        instrumented = self._heal(
                            instrumented, last_error, debug_dir, attempt,
                            hint="Force device='cuda' (or torch.cuda.current_device()) "
                                 "everywhere the script picks 'cpu'. Make sure model and "
                                 "tensors move to CUDA before training.",
                        )
                    continue

                if result.success:
                    print(f"  [attempt {attempt + 1}] Success.")
                    break

                error_text = result.error or result.stderr or "Unknown error"
                kind = _classify_error(error_text)
                print(f"  [attempt {attempt + 1}] Failed [{kind}]: {error_text[:200]}")

                # Bail out if the previous attempt failed with the same
                # signature — re-running an identical container wastes
                # Modal time and LLM calls.
                if _same_error_signature(error_text, last_error):
                    print("  [healing] Same failure as previous attempt; aborting.")
                    last_error = error_text
                    break
                last_error = error_text

                if attempt >= self._max_retries - 1:
                    continue

                if kind == "missing_dep":
                    new_deps = _extract_missing_dep_names(error_text)
                    new_deps = {d for d in new_deps
                                if d.lower() not in {x.lower() for x in dependencies}}
                    if new_deps:
                        dependencies.extend(sorted(new_deps))
                        print(f"  [healing] Adding missing deps: {sorted(new_deps)}")
                    else:
                        print(f"  [healing] Missing dep but no install name; falling back to source heal.")
                        instrumented = self._heal(
                            instrumented, error_text, debug_dir, attempt,
                        )
                elif kind == "deps":
                    bad = _extract_bad_pip_names(error_text)
                    if bad:
                        dependencies, dropped = _drop_bad_deps(dependencies, bad)
                        print(f"  [healing] Dropped non-installable deps: {dropped}")
                    else:
                        print("  [healing] Build failed but no dep name extracted; "
                              "letting source heal try.")
                        instrumented = self._heal(
                            instrumented, error_text, debug_dir, attempt,
                        )
                elif kind == "infra":
                    print("  [healing] Treating as transient infra error; retrying.")
                else:
                    hint = _local_package_hint(error_text, local_names)
                    instrumented = self._heal(
                        instrumented, error_text, debug_dir, attempt, hint=hint,
                    )
        except KeyboardInterrupt:
            print("\n  Cancelled by user.")
            if result is None:
                from profine.modal.executor import ExecutionResult
                result = ExecutionResult(success=False, error="Cancelled by user")
            last_error = "Cancelled by user"

        # Step 5: Build ProfileRecord from payload
        payload = result.payload if result and result.success else {}
        record = _build_record(payload, path, hw, steps, warmup_steps)

        if not result or not result.success:
            record.status = "crash"
            record.error = last_error

        # Step 6: Compute heuristics
        _enrich_with_heuristics(record, hw)

        # Step 7: Generate report
        markdown = generate_report(record)

        warnings = []
        if record.status != "ok":
            warnings.append(f"Profile status: {record.status}")
        if record.error:
            warnings.append(f"Error: {record.error}")

        return ProfileResult(
            record=record,
            markdown=markdown,
            instrumented_source=instrumented.source,
            raw_payload=payload,
            warnings=warnings,
        )

    def _heal(self, instrumented, error, debug_dir, attempt, hint=""):
        """Run the LLM source healer with an optional hint and persist
        the healed copy for debugging. Returns the (possibly unchanged)
        instrumented script."""
        print(f"  [healing] Asking LLM to fix the script{f' ({hint[:60]})' if hint else ''}...")
        try:
            healed = self._instrumentor.heal(instrumented, error, hint=hint)
            (debug_dir / f"instrumented_heal{attempt + 1}.py").write_text(
                healed.source, encoding="utf-8",
            )
            return healed
        except Exception as heal_err:
            print(f"  [healing] LLM heal failed: {heal_err}")
            return instrumented


def _build_record(
    payload: dict[str, Any],
    script_path: Path,
    hardware: HardwareConfig,
    steps: int,
    warmup_steps: int,
) -> ProfileRecord:
    """Convert raw Modal payload into a ProfileRecord."""
    all_step_times = payload.get("step_times_ms", [])
    steps_completed = payload.get("steps_completed", len(all_step_times))

    # Detect stabilization
    effective_warmup = detect_stabilization_point(all_step_times, min_warmup=warmup_steps)
    steady_times = all_step_times[effective_warmup:]
    warmup_times = all_step_times[:effective_warmup]

    # Parse profiler events
    raw_events = payload.get("profiler_events", [])
    events = parse_events_from_payload(raw_events) if isinstance(raw_events, list) else []

    record = ProfileRecord(
        status=payload.get("status", "ok"),
        script_path=str(script_path),
        hardware_name=hardware.name,
        steps_requested=steps,
        steps_completed=steps_completed,
        warmup_steps_requested=warmup_steps,
        warmup_steps_effective=effective_warmup,
        runtime_seconds=payload.get("runtime_seconds", 0.0),
        step_times_ms=steady_times,
        warmup_step_times_ms=warmup_times,
        loss_values=payload.get("loss_values", []),
        gpu_util_samples=payload.get("gpu_utilization_samples", []),
        memory_samples_bytes=payload.get("memory_samples_bytes", []),
        memory_peak_bytes=payload.get("memory_peak_bytes", 0),
        profiler_events=events,
        metadata=payload.get("metadata", {}),
    )
    return record


def _enrich_with_heuristics(record: ProfileRecord, hardware: HardwareConfig) -> None:
    """Compute and attach all heuristics to a ProfileRecord."""
    events = record.profiler_events
    step_total_us = sum(record.step_times_ms) * 1000 if record.step_times_ms else 0

    # Trim init/teardown idle periods, compute summary, then discard raw samples
    samples = record.gpu_util_samples
    # Strip leading/trailing zeros (model init + post-training teardown)
    start = 0
    while start < len(samples) and samples[start] == 0.0:
        start += 1
    end = len(samples)
    while end > start and samples[end - 1] == 0.0:
        end -= 1
    trimmed = samples[start:end]

    record.gpu_util_mean = compute_gpu_mean(trimmed)
    record.gpu_util_pattern = classify_gpu_pattern(trimmed)

    # Downsample to windowed means (chunks of 10 → ~1/10th the size)
    chunk = 10
    record.gpu_util_samples = [
        round(sum(trimmed[i:i + chunk]) / len(trimmed[i:i + chunk]), 1)
        for i in range(0, len(trimmed), chunk)
    ]

    # Compute memory headroom before converting to GB
    record.memory_headroom_pct = compute_memory_headroom(record.memory_peak_bytes, hardware.vram_gb)

    # Compact memory samples: keep unique values only (typically all the same after warmup)
    seen = set()
    unique_mem = []
    for b in record.memory_samples_bytes:
        if b not in seen:
            seen.add(b)
            unique_mem.append(b)
    record.memory_samples_bytes = unique_mem

    record.top_kernels = compute_top_kernels(events)
    record.kernel_breakdown = compute_category_breakdown(events)
    record.phase_breakdown = compute_phase_breakdown(events)
    record.dataloader_stall_pct = compute_dataloader_stall_pct(events, step_total_us)
    record.arithmetic_intensity = compute_arithmetic_intensity(events)
    record.attention_impl = detect_attention_impl(events)
    record.precision = detect_precision(events)

    comm_overhead, comm_overlap = compute_communication_overhead(events)
    record.communication_overhead_pct = comm_overhead
    record.communication_overlapped = comm_overlap


_PIP_NOT_FOUND = re.compile(
    r"(?:no matching distribution found for|could not find a version that satisfies the requirement)\s+([A-Za-z0-9_.\-]+)",
    re.IGNORECASE,
)
_MODULE_NOT_FOUND = re.compile(
    r"No module named ['\"]([A-Za-z0-9_]+)", re.IGNORECASE,
)
_EXCEPTION_CLASS = re.compile(r"([A-Z][A-Za-z]+(?:Error|Exception)):")


def _classify_error(error: str) -> str:
    """Route a Modal failure to the right healer.

    Returns: "missing_dep" (importable package not installed in image),
    "deps" (bad pip name during build), "infra" (Modal/network), or
    "script" (script-level bug fixable by rewriting source)."""
    e = (error or "").lower()
    if "image build" in e and "failed" in e:
        return "deps"
    if "no matching distribution" in e or "could not find a version" in e:
        return "deps"
    if any(s in e for s in ("connectionreset", "connection reset",
                              "connectiontimeout", "connection timed out",
                              "network is unreachable", "modal app failed to")):
        return "infra"
    # ModuleNotFoundError naming a known PyPI top-level → missing dep,
    # not a script bug. Don't ask the LLM to rewrite the source.
    known = get_known_pypi_toplevel()
    for missing in _MODULE_NOT_FOUND.findall(error or ""):
        if missing.lower() in known:
            return "missing_dep"
    return "script"


def _extract_missing_dep_names(error: str) -> set[str]:
    """Pull pip-installable names from `ModuleNotFoundError` messages."""
    known = get_known_pypi_toplevel()
    mapping = get_import_to_pip()
    found: set[str] = set()
    for missing in _MODULE_NOT_FOUND.findall(error or ""):
        if missing.lower() in known:
            found.add(mapping.get(missing, missing))
    return found


def _extract_bad_pip_names(error: str) -> set[str]:
    """Pull pip package names out of a build-failure traceback so we
    can drop them from the deps list before retrying."""
    return {m.lower() for m in _PIP_NOT_FOUND.findall(error or "")}


def _drop_bad_deps(dependencies: list[str], bad: set[str]) -> tuple[list[str], list[str]]:
    """Return (kept, dropped) after filtering out names in `bad`."""
    kept, dropped = [], []
    for dep in dependencies:
        base = dep.split(">=")[0].split("==")[0].split("[")[0].lower()
        (dropped if base in bad else kept).append(dep)
    return kept, dropped


def _same_error_signature(a: str, b: str) -> bool:
    """Cheap stable signature: error class + first missing-module name +
    trimmed message. Catches the 'identical failure 3x' loop without
    false-positives on transient changes."""
    if not a or not b:
        return False
    def sig(s: str) -> str:
        cls = _EXCEPTION_CLASS.search(s)
        mod = _MODULE_NOT_FOUND.search(s)
        # Fallback: if no exception class is named (e.g. raw str(e) = "'O'"),
        # use the trimmed message itself so identical-message crashes still
        # collapse to one signature.
        msg = (s or "").strip()[:120]
        return f"{cls.group(1) if cls else ''}|{mod.group(1) if mod else ''}|{msg if not cls else ''}"
    return sig(a) == sig(b) and sig(a) != "||"


def _local_package_hint(error: str, local_names: set[str]) -> str:
    """If a ModuleNotFoundError names a package that exists in the
    project tree, tell the healer it's a path problem — don't let it
    invent replacement libraries."""
    for missing in _MODULE_NOT_FOUND.findall(error or ""):
        if missing in local_names:
            return (
                f"`{missing}` is a LOCAL package in this project (already on "
                f"sys.path at /workspace). Do NOT replace it with a different "
                f"library (e.g. yacs). Keep the import as-is."
            )
    return ""


def _looks_like_cpu_run(payload: dict[str, Any], hardware_name: str) -> bool:
    """Detect runs that completed but never used the GPU. Triggered
    when the user asked for a GPU host but the script defaulted to
    CPU"""
    if "cpu" in hardware_name.lower():
        return False
    samples = payload.get("gpu_utilization_samples") or []
    peak_bytes = payload.get("memory_peak_bytes") or 0
    if not samples:
        return False
    mean_util = sum(samples) / len(samples)
    # < 5% util AND < 64 MB GPU memory peak → script never touched the GPU.
    return mean_util < 5.0 and peak_bytes < 64 * 1024 * 1024


def _parse_dep_list(raw: str) -> list[str]:
    """Parse a JSON array of pip package names from LLM response."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    deps = json.loads(text)
    if isinstance(deps, list):
        return [str(d) for d in deps if isinstance(d, str) and d]
    return []


def _record_to_dict(record: ProfileRecord) -> dict[str, Any]:
    """Convert a ProfileRecord to a JSON-serializable dict."""
    from dataclasses import asdict
    d = asdict(record)
    # ProfilerEvent and other nested dataclasses are already dicts via asdict
    return d
