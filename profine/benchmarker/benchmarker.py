"""Benchmarker orchestrator (plan 4.6).

Runs baseline vs. optimized scripts on Modal, compares metrics,
and verifies correctness via loss curve matching.

Usage:
    from profine.benchmarker.benchmarker import Benchmarker

    benchmarker = Benchmarker(provider="openai")
    result = benchmarker.benchmark(
        baseline_path="train.py",
        optimized_source=edited_source,
        hardware="1x_a100",
    )

    print(result.markdown)
    print(result.comparison.speedup_pct)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from profine.benchmarker.comparator import BenchmarkComparison, compare_payloads
from profine.benchmarker.report import generate_report
from profine.config.settings import DEFAULTS
from profine.llm.backend import create_backend
from profine.profiler.instrumentor import ScriptInstrumentor
from profine.profiler.stabilization import detect_stabilization_point
from profine.reader.extractor import extract
from profine.schema.hardware import HardwareConfig, ModalRuntimeConfig, get_hardware


@dataclass(slots=True)
class BenchmarkResult:
    """Output of the Benchmarker tool."""
    comparison: BenchmarkComparison
    markdown: str
    baseline_payload: dict[str, Any] = field(default_factory=dict)
    candidate_payload: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    optimization_name: str = ""

    @property
    def passed(self) -> bool:
        return self.comparison.correctness.passed

    @property
    def speedup_pct(self) -> float:
        return self.comparison.speedup_pct

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        report_path = out / "benchmark_report.md"
        comparison_path = out / "benchmark_comparison.json"

        report_path.write_text(self.markdown, encoding="utf-8")
        comparison_path.write_text(
            json.dumps(_comparison_to_dict(self.comparison), indent=2, default=str),
            encoding="utf-8",
        )

        return {"report": report_path, "comparison": comparison_path}


class Benchmarker:
    """Runs baseline vs. optimized on Modal and compares."""

    def __init__(
        self,
        provider: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        seed: int | None = None,
        modal_config: ModalRuntimeConfig | None = None,
        hf_token: str | None = None,
        max_retries: int = 2,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
        if seed is not None:
            kwargs["seed"] = seed
        self._backend = create_backend(provider, **kwargs)
        self._instrumentor = ScriptInstrumentor(self._backend)
        self._modal_config = modal_config
        self._hf_token = hf_token
        self._max_retries = max_retries

    def benchmark(
        self,
        baseline_path: str | Path,
        optimized_source: str,
        *,
        hardware: str | HardwareConfig = DEFAULTS.default_hardware,
        steps: int = DEFAULTS.default_steps,
        warmup_steps: int = DEFAULTS.default_warmup_steps,
        rtol: float = 1e-2,
        atol: float = 1e-4,
        optimization_name: str = "",
        optimization_category: str = "",
        optimization_categories: list[str] | None = None,
        script_args: list[str] | None = None,
        overlay_files: dict[str, str] | None = None,
    ) -> BenchmarkResult:
        """Run A/B benchmark: baseline vs. optimized.

        Args:
            baseline_path: Path to the original training script.
            optimized_source: Source code of the optimized script.
            hardware: Hardware preset name or HardwareConfig.
            steps: Total optimizer steps per run.
            warmup_steps: Steps to discard as warmup.
            rtol: Relative tolerance for loss curve correctness check.
            atol: Absolute tolerance for loss curve correctness check.
            optimization_name: Name of the optimization being tested.
            script_args: Optional CLI arguments for the scripts.

        Returns:
            BenchmarkResult with comparison, verdict, and markdown report.
        """
        path = Path(baseline_path)
        baseline_source = path.read_text(encoding="utf-8")
        hw = get_hardware(hardware) if isinstance(hardware, str) else hardware

        warnings: list[str] = []

        # Discover dependencies once from the original script
        from profine.modal.discovery import discover_dependencies
        dependencies = discover_dependencies(path)

        # Instrument the baseline. CRITICAL: when the optimized entry
        # script is identical to the baseline (multi-file edit — the
        # change lives in overlay_files), reuse the baseline's
        # instrumented source for the optimized run. Otherwise the LLM
        # rewrites each script independently and may produce different
        # synthetic data shapes/vocabs, making the loss comparison
        # meaningless. Re-instrumenting only when the entry source
        # actually differs preserves data parity for the common case.
        baseline_instrumented = self._instrument(baseline_source, path.name, steps, warmup_steps)
        if baseline_instrumented is None:
            warnings.append("Failed to instrument baseline script")

        if optimized_source == baseline_source:
            optimized_instrumented = baseline_instrumented
        else:
            optimized_instrumented = self._instrument(optimized_source, path.name, steps, warmup_steps)
            if optimized_instrumented is None:
                warnings.append("Failed to instrument optimized script")

        # Execute both on Modal. Overlay files (e.g. patched library
        # modules from the editor) are applied to the optimized run only.
        print(f"  [1/2] Running baseline on {hw.name}...")
        baseline_payload = self._execute(
            baseline_instrumented or baseline_source, baseline_source,
            path, hw, steps, script_args, dependencies, label="baseline",
        )
        if _is_failed_payload(baseline_payload):
            raise RuntimeError(
                "Baseline run failed on all retry attempts; cannot produce a benchmark "
                "comparison. The optimized run was not attempted. Check the [baseline] "
                "errors above — common causes: missing data files (e.g. `input.txt`), "
                "dependencies not in requirements, or a non-deterministic crash. "
                "If you fixed the underlying issue, re-run `profine benchmark`."
            )

        print(f"  [2/2] Running optimized on {hw.name}...")
        candidate_payload = self._execute(
            optimized_instrumented or optimized_source, optimized_source,
            path, hw, steps, script_args, dependencies, label="optimized",
            overlay_files=overlay_files,
        )
        if _is_failed_payload(candidate_payload):
            raise RuntimeError(
                "Optimized run failed on all retry attempts; cannot produce a benchmark "
                "comparison. (Baseline succeeded.) The optimization may have introduced a "
                "regression that the healer couldn't recover from. Review "
                "`profine_output/edit/edited_train.py` and the [optimized] errors above."
            )

        # Strip warmup from both
        _strip_warmup(baseline_payload, warmup_steps)
        _strip_warmup(candidate_payload, warmup_steps)

        # Pick correctness tolerance based on the optimization class.
        # Mixed-precision math legitimately perturbs losses by ~1-3%
        # per step, far above the default rtol=1e-2 used for algebraic
        # equivalences. Without this, every BF16/FP16 benchmark fails
        # correctness even when the optimization is correct.
        # For stacked edits, take the loosest tolerance across all
        # applied optimization classes. A single BF16 + compile stack,
        # for example, should use BF16's wider tolerance, not compile's.
        cats = list(optimization_categories or [])
        if optimization_category and optimization_category not in cats:
            cats.insert(0, optimization_category)
        eff_rtol, eff_atol = rtol, atol
        chosen_cat = ""
        for cat in cats or [""]:
            r, a = _resolve_tolerance(cat, rtol, atol)
            if (r, a) > (eff_rtol, eff_atol):
                eff_rtol, eff_atol, chosen_cat = r, a, cat
        if (eff_rtol, eff_atol) != (rtol, atol):
            label = chosen_cat or "stacked"
            warnings.append(
                f"Loss tolerance widened for category '{label}': "
                f"rtol {rtol} -> {eff_rtol}, atol {atol} -> {eff_atol}"
            )

        # Compare
        comparison = compare_payloads(baseline_payload, candidate_payload, rtol=eff_rtol, atol=eff_atol)

        # Generate report
        markdown = generate_report(
            comparison,
            optimization_name,
            hardware=hw.name if hasattr(hw, "name") else None,
            cost_per_hour=getattr(hw, "cost_per_hour", None),
        )

        return BenchmarkResult(
            comparison=comparison,
            markdown=markdown,
            baseline_payload=baseline_payload,
            candidate_payload=candidate_payload,
            warnings=warnings,
            optimization_name=optimization_name,
        )

    def _instrument(
        self, source: str, filename: str, steps: int, warmup_steps: int,
    ) -> str | None:
        """Instrument a script for benchmarking. Returns None on failure."""
        try:
            facts = extract(source, filename)
            result = self._instrumentor.instrument(
                source=source,
                facts=facts,
                total_steps=steps,
                warmup_steps=warmup_steps,
                benchmark_mode=True,
            )
            return result.source
        except Exception:
            return None

    def _execute(
        self,
        instrumented_source: str,
        original_source: str,
        script_path: Path,
        hardware: HardwareConfig,
        total_steps: int,
        script_args: list[str] | None,
        dependencies: list[str] | None = None,
        label: str = "run",
        overlay_files: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute on Modal with retry + healing. Returns the raw payload."""
        from profine.modal.executor import ModalExecutor

        executor = ModalExecutor(
            modal_config=self._modal_config,
            hf_token=self._hf_token,
        )

        current_source = instrumented_source
        for attempt in range(self._max_retries + 1):
            print(f"  [{label}] attempt {attempt + 1}/{self._max_retries + 1}...")
            result = executor.execute(
                instrumented_source=current_source,
                script_path=script_path,
                hardware=hardware,
                total_steps=total_steps,
                script_args=script_args,
                dependencies=dependencies,
                overlay_files=overlay_files,
            )
            if result.success:
                print(f"  [{label}] Success.")
                return result.payload

            error = result.error or result.stderr or "Unknown error"
            print(f"  [{label}] Failed: {error[:200]}")

            if attempt < self._max_retries:
                print(f"  [{label}] Healing...")
                try:
                    from profine.profiler.instrumentor import InstrumentedScript
                    healed = self._instrumentor.heal(
                        InstrumentedScript(
                            source=current_source,
                            original_source=original_source,
                            active_steps=0,
                            scale_factor=1.0,
                        ),
                        error,
                    )
                    current_source = healed.source
                except Exception as e:
                    print(f"  [{label}] Heal failed: {e}")

        print(f"  [{label}] All attempts exhausted.")
        return {}


def _is_failed_payload(payload: dict[str, Any] | None) -> bool:
    """Detect when _execute exhausted retries and returned a useless payload.

    A successful run returns a dict with non-empty step_times_ms. An exhausted
    run returns {} (or a dict missing those fields). Distinguishing the two
    matters because compare_payloads happily produces a meaningless 0%-change
    report when given empty inputs.
    """
    if not payload:
        return True
    step_times = payload.get("step_times_ms")
    return not step_times


def _resolve_tolerance(
    category: str,
    user_rtol: float,
    user_atol: float,
) -> tuple[float, float]:
    """Pick (rtol, atol) given the optimization class.

    If the user passed non-default tolerances we trust them. Otherwise
    we widen for classes where the optimization legitimately changes
    the numerics (BF16, quantization). Returning the inputs unchanged
    when no override applies keeps single-class refactor optimizations
    on the strict default. Overrides live in catalog.yaml.
    """
    from profine.config.yaml_loader import get_category_tolerances
    DEFAULT_RTOL, DEFAULT_ATOL = 1e-2, 1e-4
    if user_rtol != DEFAULT_RTOL or user_atol != DEFAULT_ATOL:
        return user_rtol, user_atol
    cat = (category or "").lower()
    for key, (rtol, atol) in get_category_tolerances().items():
        if key in cat:
            return rtol, atol
    return user_rtol, user_atol


def _strip_warmup(payload: dict[str, Any], warmup_steps: int) -> None:
    """Remove warmup steps from `step_times_ms` only.

    Loss values are deliberately left untouched so that baseline and candidate
    can be aligned by **original training-step index** in the correctness check.
    Stripping losses per-payload caused a real bug: baseline stabilized at step 10
    while the torch.compile'd candidate stabilized at step 15, so post-strip step 0
    compared baseline-step-10 against candidate-step-15 — five training steps
    apart — and reported a spurious correctness FAIL (~0.18 loss divergence).
    Step times still get stripped because median-step-time math wants only the
    steady-state region; loss curves want raw alignment.
    """
    step_times = payload.get("step_times_ms", [])
    if not step_times:
        return

    effective_warmup = detect_stabilization_point(step_times, min_warmup=warmup_steps)
    payload["step_times_ms"] = step_times[effective_warmup:]


def _comparison_to_dict(comparison: BenchmarkComparison) -> dict[str, Any]:
    from dataclasses import asdict
    return asdict(comparison)
