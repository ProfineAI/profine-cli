"""CLI command handlers — one function per subcommand."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from profine.cli import _console as ui


def cmd_read(args: Namespace, output_dir: Path, user_prefs: str | None) -> int:
    from profine.reader.reader import CodeReader

    print(f"Reading {args.script}...")
    reader = CodeReader(provider=args.provider, api_key=args.api_key, model=args.model, base_url=args.base_url, seed=args.seed)
    result = reader.read(args.script, debug_dir=output_dir / "_debug")

    out = output_dir / "read"
    paths = result.save(out)
    ui.path("Architecture record", paths["record"])
    ui.path("Architecture brief ", paths["brief"])

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"  - {w}")

    print(f"\n{result.markdown_brief[:500]}...")
    return 0


def cmd_profile(args: Namespace, output_dir: Path, user_prefs: str | None) -> int:
    from profine.profiler import profile
    from profine.schema.hardware import ModalRuntimeConfig
    from profine.config.settings import DEFAULTS

    timeout = getattr(args, "timeout", DEFAULTS.default_modal_timeout)
    modal_config = ModalRuntimeConfig(timeout_seconds=timeout)
    if getattr(args, "warmstart", False):
        modal_config.enable_warmstart = True

    print(f"Profiling {args.script} on {args.hardware}...")
    result = profile(
        args.script,
        hardware=args.hardware,
        steps=args.steps,
        warmup_steps=args.warmup,
        provider=args.provider,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        seed=args.seed,
        modal_config=modal_config,
    )

    out = output_dir / "profile"
    paths = result.save(out)
    ui.path("Profile record", paths["record"])
    ui.path("Profile report", paths["report"])

    if result.warnings:
        for w in result.warnings:
            print(f"  Warning: {w}")

    print(f"\n{result.markdown[:500]}...")
    return 0


def cmd_interpret(args: Namespace, output_dir: Path, user_prefs: str | None) -> int:
    from profine.interpreter.interpreter import ProfileInterpreter
    from profine.schema.profile_record import ProfileRecord

    profile_dir = Path(args.profile_dir)
    record_path = profile_dir / "profile_record.json"
    arch_path = profile_dir.parent / "read" / "architecture_record.json"

    record_data = json.loads(record_path.read_text(encoding="utf-8"))
    arch_data = json.loads(arch_path.read_text(encoding="utf-8")) if arch_path.exists() else None

    # Build a minimal ProfileRecord for the interpreter
    profile_record = _dict_to_profile_record(record_data)

    print("Interpreting profile...")
    interpreter = ProfileInterpreter(provider=args.provider, api_key=args.api_key, model=args.model, base_url=args.base_url, seed=args.seed)
    result = interpreter.interpret(profile_record, arch_data, user_prefs, debug_dir=output_dir / "_debug")

    out = output_dir / "interpret"
    paths = result.save(out)
    ui.path("Bottleneck report", paths["report"])
    ui.path("Bottleneck brief ", paths["brief"])

    print(f"\n{result.markdown[:500]}...")
    return 0


def cmd_suggest(args: Namespace, output_dir: Path, user_prefs: str | None) -> int:
    from profine.suggester.suggester import OptimizationSuggester
    from profine.schema.bottleneck_report import BottleneckReport

    interpret_dir = Path(args.interpret_dir)

    # Load architecture record
    if args.arch_dir:
        arch_path = Path(args.arch_dir) / "architecture_record.json"
    else:
        arch_path = interpret_dir.parent / "read" / "architecture_record.json"
    arch_data = json.loads(arch_path.read_text(encoding="utf-8")) if arch_path.exists() else {}

    # Load bottleneck report + profile summary from interpreter output
    bn_path = interpret_dir / "bottleneck_report.json"
    bottleneck_report = None
    profile_summary = None
    if bn_path.exists():
        interpret_data = json.loads(bn_path.read_text(encoding="utf-8"))
        if "bottleneck_report" in interpret_data:
            bottleneck_report = _dict_to_bottleneck_report(interpret_data["bottleneck_report"])
            profile_summary = interpret_data.get("profile_summary")
        else:
            bottleneck_report = _dict_to_bottleneck_report(interpret_data)

    print("Suggesting optimizations...")
    suggester = OptimizationSuggester(provider=args.provider, api_key=args.api_key, model=args.model, base_url=args.base_url, seed=args.seed)
    result = suggester.suggest(arch_data, bottleneck_report, user_prefs, profile_summary,
                                debug_dir=output_dir / "_debug")

    out = output_dir / "suggest"
    paths = result.save(out)
    ui.path("Suggestion report", paths["report"])
    ui.path("Suggestion brief ", paths["brief"])

    print(f"\n{result.markdown[:500]}...")
    return 0


def cmd_edit(args: Namespace, output_dir: Path, user_prefs: str | None) -> int:
    from profine.editor.editor import CodeEditor

    # Auto-detect script path from prior steps if not provided
    if not args.script:
        args.script = _auto_detect_script(output_dir)
        if not args.script:
            print("Error: no script specified and could not auto-detect from output directory.\n"
                  "Either pass the script path or run `profine read` first.")
            return 1
        print(f"Auto-detected script: {args.script}")

    suggestion_dir = Path(args.suggestion_dir)
    report_path = suggestion_dir / "suggestion_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    candidates = report.get("candidates", [])
    if not candidates:
        print("No candidates in suggestion report.")
        return 1

    # Resolve which candidates to apply. --top wins over --optimization;
    # otherwise fall back to a single target picked by --optimization or
    # the top-ranked entry.
    top_n = getattr(args, "top", None)
    if top_n and top_n > 0:
        targets = candidates[: top_n]
    elif args.optimization:
        opt = args.optimization
        if opt.isdigit():
            idx = int(opt) - 1
            if not (0 <= idx < len(candidates)):
                print(f"Rank {opt} out of range. Available: 1-{len(candidates)}")
                return 1
            targets = [candidates[idx]]
        else:
            t = next((c for c in candidates if c["entry_id"] == opt), None)
            if not t:
                print(f"Optimization '{opt}' not found. "
                      f"Available: {[c['entry_id'] for c in candidates]}")
                return 1
            targets = [t]
    else:
        targets = [candidates[0]]

    # Load entry source and architecture record.
    script_path = Path(args.script).resolve()
    source = script_path.read_text(encoding="utf-8")
    arch_path = output_dir / "read" / "architecture_record.json"
    arch_data = json.loads(arch_path.read_text(encoding="utf-8")) if arch_path.exists() else None

    # Discover local modules so the editor can patch imported files.
    from profine.modal.discovery import discover_local_modules, discover_project_root
    project_root = discover_project_root(script_path)
    base_local_modules = discover_local_modules(script_path)
    try:
        entry_rel = script_path.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        entry_rel = script_path.name

    editor = CodeEditor(provider=args.provider, api_key=args.api_key, model=args.model, base_url=args.base_url, seed=args.seed)
    out = output_dir / "edit"

    # Stacked-edit state. Each iteration sees the cumulative result so
    # far as its "input" (entry source + local modules), so optimization
    # K is applied on top of optimizations 1..K-1.
    current_source = source
    cumulative_extras: dict[str, str] = {}  # path -> latest edited source
    applied_ids: list[str] = []
    applied_exclusive_groups: set[int] = set()  # track which groups are taken
    skipped: list[tuple[str, str]] = []     # (entry_id, reason)
    all_warnings: list[str] = []
    last_explanation = ""

    for i, target_dict in enumerate(targets, start=1):
        candidate = _dict_to_candidate(target_dict)
        prefix = f"[{i}/{len(targets)}] " if len(targets) > 1 else ""

        # Skip if an entry from the same exclusive group was already applied
        excl_group = target_dict.get("exclusive_group", 0)
        if excl_group and excl_group in applied_exclusive_groups:
            reason = (f"exclusive group {excl_group} — "
                      f"conflicts with already-applied optimization")
            skipped.append((candidate.entry_id, reason))
            print(f"{prefix}Skipped {candidate.name}: {reason}")
            continue

        print(f"{prefix}Applying optimization: {candidate.name}...")

        # Layer cumulative edits over the discovered baseline modules
        # so the LLM sees the up-to-date state of every file.
        effective_locals = dict(base_local_modules)
        effective_locals.update(cumulative_extras)
        if effective_locals and i == 1:
            print(f"  (with {len(effective_locals)} local module(s) as context)")

        result = editor.edit(
            current_source, candidate, arch_data, user_prefs,
            entry_path=entry_rel, local_modules=effective_locals,
            debug_dir=output_dir / "_debug",
        )

        # Per-iteration artifacts under NN_<entry_id>/ for traceability.
        iter_dir = out / f"{i:02d}_{candidate.entry_id}"
        result.save(iter_dir)

        all_warnings.extend(result.warnings)

        if result.applied:
            current_source = result.edited_source
            for fe in result.extra_file_edits:
                cumulative_extras[fe.path] = fe.edited_source
            applied_ids.append(candidate.entry_id)
            if excl_group:
                applied_exclusive_groups.add(excl_group)
            last_explanation = result.explanation
            print(f"  applied. (entry diff: {_diff_line_count(result.diff)} lines, "
                  f"extra files: {len(result.extra_file_edits)})")
        else:
            reason = result.not_applicable_reason or "no edit returned"
            skipped.append((candidate.entry_id, reason))
            print(f"  skipped: {reason}")

    # Materialize the cumulative result at the standard top-level paths
    # so `profine benchmark --optimized <output>/edit/edited_train.py`
    # picks up both the edited entry and every patched library file
    # without any extra flags.
    out.mkdir(parents=True, exist_ok=True)
    edited_path = out / "edited_train.py"
    diff_path = out / "edited_train.py.diff"
    manifest_path = out / "change_manifest.json"
    files_dir = out / "files"

    edited_path.write_text(current_source, encoding="utf-8")
    diff_path.write_text(_unified_diff(source, current_source), encoding="utf-8")

    # Rewrite files/ from scratch so a previous run's overlays don't leak.
    if files_dir.exists():
        import shutil
        shutil.rmtree(files_dir)
    for rel, content in cumulative_extras.items():
        target = files_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        diff_target = target.with_suffix(target.suffix + ".diff")
        diff_target.write_text(
            _unified_diff(base_local_modules.get(rel, ""), content,
                          from_label=rel, to_label=f"{rel} (optimized)"),
            encoding="utf-8",
        )

    manifest = {
        # Keep `optimization_id` (singular) for back-compat with any
        # consumer reading the field by that name; for stacked edits
        # it's the FIRST applied id, and `applied_ids` gives the full
        # list.
        "optimization_id": applied_ids[0] if applied_ids else "",
        "applied_ids": applied_ids,
        "skipped": [{"entry_id": eid, "reason": r} for eid, r in skipped],
        "warnings": all_warnings,
        "extra_files": sorted(cumulative_extras.keys()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Summary
    print()
    ui.path("Edited script", edited_path)
    ui.path("Diff         ", diff_path)
    ui.path("Manifest     ", manifest_path)
    if cumulative_extras:
        print(f"Extra files edited ({len(cumulative_extras)}):")
        for rel in sorted(cumulative_extras):
            print(f"  - {rel} -> {files_dir / rel}")
    if applied_ids:
        print(f"\nApplied: {', '.join(applied_ids)}")
        if last_explanation:
            print(f"\n{last_explanation}")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for eid, reason in skipped:
            print(f"  - {eid}: {reason}")
    if not applied_ids:
        return 1
    return 0


def _diff_line_count(diff_text: str) -> int:
    """Count +/- payload lines in a unified diff (excluding ---/+++ headers)."""
    n = 0
    for line in diff_text.splitlines():
        if (line.startswith("+") and not line.startswith("+++")) or \
           (line.startswith("-") and not line.startswith("---")):
            n += 1
    return n


def _unified_diff(
    original: str,
    edited: str,
    from_label: str = "original",
    to_label: str = "optimized",
) -> str:
    import difflib
    return "".join(difflib.unified_diff(
        original.splitlines(keepends=True),
        edited.splitlines(keepends=True),
        fromfile=from_label, tofile=to_label, lineterm="",
    ))


def cmd_benchmark(args: Namespace, output_dir: Path, user_prefs: str | None) -> int:
    from profine.benchmarker.benchmarker import Benchmarker
    from profine.schema.hardware import ModalRuntimeConfig
    from profine.config.settings import DEFAULTS

    # Auto-detect script path from prior steps if not provided
    if not args.script:
        args.script = _auto_detect_script(output_dir)
        if not args.script:
            print("Error: no script specified and could not auto-detect from output directory.\n"
                  "Either pass the script path or run `profine read` first.")
            return 1
        print(f"Auto-detected script: {args.script}")

    # Auto-detect optimized script from edit output if not provided
    if not args.optimized:
        default_optimized = output_dir / "edit" / "edited_train.py"
        if default_optimized.exists():
            args.optimized = str(default_optimized)
            print(f"Auto-detected optimized script: {args.optimized}")
        else:
            print("Error: no --optimized script specified and "
                  f"{default_optimized} does not exist.\n"
                  "Run `profine edit` first or pass --optimized explicitly.")
            return 1

    optimized_path = Path(args.optimized)
    if not optimized_path.is_file():
        print(f"Error: '{args.optimized}' is not a file."
              + (" Did you mean to pass the edited script, e.g. "
                 f"'{args.optimized}/edited_train.py'?"
                 if optimized_path.is_dir() else ""))
        return 1

    timeout = getattr(args, "timeout", DEFAULTS.default_modal_timeout)
    modal_config = ModalRuntimeConfig(timeout_seconds=timeout)
    if getattr(args, "warmstart", False):
        modal_config.enable_warmstart = True

    optimized_source = optimized_path.read_text(encoding="utf-8")

    # Pick up multi-file editor output: any patched library modules
    # under <edit-dir>/files/ are overlaid onto the Modal workspace at
    # the optimized run. Without this, edits to imported modules
    # are silently ignored and the benchmark
    # measures the original code on both sides.
    edit_dir = Path(args.edit_dir) if getattr(args, "edit_dir", None) else (output_dir / "edit")
    overlay_files = _load_overlay_files(edit_dir)

    # Read the applied optimization(s) from the change manifest so we
    # can widen loss tolerance for classes that legitimately perturb
    # the loss trajectory (BF16, quantization). For stacked edits we
    # collect every applied id's category and let the benchmarker use
    # the loosest applicable tolerance.
    optimization_category = ""
    optimization_name = ""
    optimization_categories: list[str] = []
    manifest_path = edit_dir / "change_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        applied_ids = manifest.get("applied_ids") or (
            [manifest["optimization_id"]] if manifest.get("optimization_id") else []
        )
        if applied_ids:
            sugg = output_dir / "suggest" / "suggestion_report.json"
            if sugg.exists():
                try:
                    candidates = json.loads(sugg.read_text(encoding="utf-8")).get("candidates", [])
                except json.JSONDecodeError:
                    candidates = []
                by_id = {c.get("entry_id"): c for c in candidates}
                names = []
                for oid in applied_ids:
                    c = by_id.get(oid)
                    if c:
                        if c.get("category"):
                            optimization_categories.append(c["category"])
                        if c.get("name"):
                            names.append(c["name"])
                optimization_name = " + ".join(names)
                # Single-category compatibility: pass first as the
                # primary category; benchmarker handles the list too.
                optimization_category = optimization_categories[0] if optimization_categories else ""

    print(f"Benchmarking {args.script} vs {args.optimized} on {args.hardware}...")
    if overlay_files:
        print(f"  Overlaying {len(overlay_files)} edited file(s) onto the optimized run.")
    if optimization_categories:
        print(f"  Optimization class(es): {', '.join(optimization_categories)}")
    benchmarker = Benchmarker(
        provider=args.provider, api_key=args.api_key, model=args.model,
        base_url=args.base_url,
        seed=args.seed,
        modal_config=modal_config,
    )
    try:
        result = benchmarker.benchmark(
            args.script,
            optimized_source,
            hardware=args.hardware,
            steps=args.steps,
            warmup_steps=args.warmup,
            rtol=args.rtol,
            atol=args.atol,
            optimization_name=optimization_name,
            optimization_category=optimization_category,
            optimization_categories=optimization_categories,
            overlay_files=overlay_files,
        )
    except RuntimeError as exc:
        # Benchmarker raises RuntimeError when baseline (or optimized) exhausts
        # all retries — we surface that as a clean CLI error instead of a traceback,
        # so run-all aborts with an actionable message.
        ui.error(f"Benchmark aborted: {exc}")
        return 1

    out = output_dir / "benchmark"
    paths = result.save(out)
    ui.path("Benchmark report    ", paths["report"])
    ui.path("Benchmark comparison", paths["comparison"])
    if result.warnings:
        for w in result.warnings:
            print(f"  Warning: {w}")
    print(f"\n{result.markdown[:500]}...")
    return 0


def _load_overlay_files(edit_dir: Path) -> dict[str, str] | None:
    """Load <edit_dir>/files/** as a project-relative-path -> source map.

    Returns None when nothing is staged so the benchmarker can skip
    overlay logic entirely on the single-file edit path.
    """
    files_dir = edit_dir / "files"
    if not files_dir.is_dir():
        return None
    overlay: dict[str, str] = {}
    for p in files_dir.rglob("*"):
        if not p.is_file() or p.suffix == ".diff":
            continue
        rel = p.relative_to(files_dir).as_posix()
        try:
            overlay[rel] = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return overlay or None



def _get_val(arch: dict, key: str) -> str:
    """Extract a value from an architecture record dict."""
    field = arch.get(key)
    if isinstance(field, dict) and "value" in field:
        return str(field["value"])
    return "unknown"


def _dict_to_profile_record(d: dict[str, Any]) -> Any:
    """Build a ProfileRecord from a saved JSON dict."""
    from profine.schema.profile_record import (
        KernelCategoryBreakdown,
        KernelSummary,
        PhaseBreakdown,
        ProfileRecord,
        ProfilerEvent,
    )

    # Parse nested dataclass lists
    profiler_events = [
        ProfilerEvent(**{k: v for k, v in e.items() if k in ProfilerEvent.__dataclass_fields__})
        for e in d.get("profiler_events", [])
    ]
    top_kernels = [
        KernelSummary(**{k: v for k, v in k_dict.items() if k in KernelSummary.__dataclass_fields__})
        for k_dict in d.get("top_kernels", [])
    ]

    # Parse nested dataclass dicts
    kb = d.get("kernel_breakdown")
    kernel_breakdown = KernelCategoryBreakdown(**kb) if kb else None

    pb = d.get("phase_breakdown")
    phase_breakdown = PhaseBreakdown(**pb) if pb else None

    return ProfileRecord(
        status=d.get("status", "ok"),
        error=d.get("error"),
        script_path=d.get("script_path", ""),
        hardware_name=d.get("hardware_name", ""),
        steps_requested=d.get("steps_requested", 0),
        steps_completed=d.get("steps_completed", 0),
        warmup_steps_requested=d.get("warmup_steps_requested", 0),
        warmup_steps_effective=d.get("warmup_steps_effective", 0),
        runtime_seconds=d.get("runtime_seconds", 0.0),
        step_times_ms=d.get("step_times_ms", []),
        warmup_step_times_ms=d.get("warmup_step_times_ms", []),
        loss_values=d.get("loss_values", []),
        gpu_util_samples=d.get("gpu_util_samples", []),
        gpu_util_mean=d.get("gpu_util_mean", 0.0),
        gpu_util_pattern=d.get("gpu_util_pattern", ""),
        memory_samples_bytes=d.get("memory_samples_bytes", []),
        memory_peak_bytes=d.get("memory_peak_bytes", 0),
        profiler_events=profiler_events,
        top_kernels=top_kernels,
        kernel_breakdown=kernel_breakdown,
        phase_breakdown=phase_breakdown,
        dataloader_stall_pct=d.get("dataloader_stall_pct", 0.0),
        arithmetic_intensity=d.get("arithmetic_intensity"),
        communication_overhead_pct=d.get("communication_overhead_pct", 0.0),
        communication_overlapped=d.get("communication_overlapped", False),
        attention_impl=d.get("attention_impl", "unknown"),
        precision=d.get("precision", "unknown"),
        memory_headroom_pct=d.get("memory_headroom_pct", 0.0),
        metadata=d.get("metadata", {}),
    )


def _dict_to_bottleneck_report(d: dict[str, Any]) -> Any:
    """Build a BottleneckReport from a saved JSON dict."""
    from profine.schema.bottleneck_report import BottleneckEntry, BottleneckReport
    entries = [
        BottleneckEntry(
            category=b.get("category", ""),
            location=b.get("location", ""),
            time_share_pct=b.get("time_share_pct", 0.0),
            est_headroom_pct=b.get("est_headroom_pct", 0.0),
            confidence=b.get("confidence", "inferred"),
        )
        for b in d.get("bottlenecks", [])
    ]
    return BottleneckReport(
        bottlenecks=entries,
        compute_bound=d.get("compute_bound", False),
        memory_bandwidth_bound=d.get("memory_bandwidth_bound", False),
        memory_capacity_bound=d.get("memory_capacity_bound", False),
        data_pipeline_bound=d.get("data_pipeline_bound", False),
        communication_bound=d.get("communication_bound", False),
        summary=d.get("summary", ""),
    )


def _dict_to_candidate(d: dict[str, Any]) -> Any:
    """Build an OptimizationCandidate from a saved JSON dict."""
    from profine.schema.optimization_candidate import OptimizationCandidate
    return OptimizationCandidate(
        entry_id=d.get("entry_id", ""),
        category=d.get("category", ""),
        name=d.get("name", ""),
        description=d.get("description", ""),
        rank=d.get("rank", 0),
        priority=d.get("priority", "medium"),
        est_speedup_low_pct=d.get("est_speedup_low_pct", 0.0),
        est_speedup_high_pct=d.get("est_speedup_high_pct", 0.0),
        rationale=d.get("rationale", ""),
        bottlenecks_addressed=d.get("bottlenecks_addressed", []),
        risks=d.get("risks", []),
        code_pattern=d.get("code_pattern", ""),
        estimated_effort=d.get("estimated_effort", ""),
    )


def cmd_run_all(args: Namespace, output_dir: Path, user_prefs: str | None) -> int:
    """Run the full pipeline: read → profile → interpret → suggest → edit → benchmark."""
    script = args.script
    steps = [
        ("read", "Reading script"),
        ("profile", "Profiling on Modal"),
        ("interpret", "Interpreting bottlenecks"),
        ("suggest", "Suggesting optimizations"),
        ("edit", "Applying optimizations"),
        ("benchmark", "Benchmarking"),
    ]

    def _step_header(idx: int, name: str, desc: str) -> None:
        ui.header(desc, step=f"{idx}/{len(steps)}")

    # 1. Read
    _step_header(1, *steps[0])
    read_args = Namespace(
        script=script, provider=args.provider, api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        seed=args.seed, output=args.output, prefs=args.prefs,
    )
    rc = cmd_read(read_args, output_dir, user_prefs)
    if rc != 0:
        ui.error("Read failed — aborting pipeline.")
        return rc

    # 2. Profile
    _step_header(2, *steps[1])
    profile_args = Namespace(
        script=script, hardware=args.hardware, steps=args.steps,
        warmup=args.warmup, timeout=args.timeout,
        warmstart=getattr(args, "warmstart", False),
        provider=args.provider, api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        seed=args.seed, output=args.output, prefs=args.prefs,
    )
    rc = cmd_profile(profile_args, output_dir, user_prefs)
    if rc != 0:
        ui.error("Profile failed — aborting pipeline.")
        return rc

    # Check profile succeeded (not crash status)
    profile_record = output_dir / "profile" / "profile_record.json"
    if profile_record.exists():
        data = json.loads(profile_record.read_text(encoding="utf-8"))
        if data.get("status") == "crash":
            print(f"\nProfile crashed: {data.get('error', 'unknown')[:200]}")
            ui.error("Cannot continue without valid profile data — aborting pipeline.")
            return 1

    # 3. Interpret
    _step_header(3, *steps[2])
    interpret_args = Namespace(
        profile_dir=str(output_dir / "profile"),
        provider=args.provider, api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        seed=args.seed, output=args.output, prefs=args.prefs,
    )
    rc = cmd_interpret(interpret_args, output_dir, user_prefs)
    if rc != 0:
        ui.error("Interpret failed — aborting pipeline.")
        return rc

    # 4. Suggest
    _step_header(4, *steps[3])
    suggest_args = Namespace(
        interpret_dir=str(output_dir / "interpret"),
        arch_dir=None, profile_dir=None,
        provider=args.provider, api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        seed=args.seed, output=args.output, prefs=args.prefs,
    )
    rc = cmd_suggest(suggest_args, output_dir, user_prefs)
    if rc != 0:
        ui.error("Suggest failed — aborting pipeline.")
        return rc

    # 5. Edit — apply all ranked candidates (or --top N)
    _step_header(5, *steps[4])
    top_n = getattr(args, "top", None)
    if top_n is None:
        # Default: apply all ranked candidates
        sugg_path = output_dir / "suggest" / "suggestion_report.json"
        if sugg_path.exists():
            sugg = json.loads(sugg_path.read_text(encoding="utf-8"))
            top_n = len(sugg.get("candidates", []))
        else:
            top_n = 10
    edit_args = Namespace(
        script=script,
        suggestion_dir=str(output_dir / "suggest"),
        optimization=None, top=top_n,
        provider=args.provider, api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        seed=args.seed, output=args.output, prefs=args.prefs,
    )
    rc = cmd_edit(edit_args, output_dir, user_prefs)
    if rc != 0:
        ui.error("Edit failed (no optimizations applied) — aborting pipeline.")
        return rc

    # 6. Benchmark
    _step_header(6, *steps[5])
    benchmark_args = Namespace(
        script=script,
        optimized=str(output_dir / "edit" / "edited_train.py"),
        hardware=args.hardware, steps=args.steps, warmup=args.warmup,
        rtol=getattr(args, "rtol", 1e-2), atol=getattr(args, "atol", 1e-4),
        timeout=args.timeout,
        warmstart=getattr(args, "warmstart", False),
        edit_dir=None,
        provider=args.provider, api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        seed=args.seed, output=args.output, prefs=args.prefs,
    )
    rc = cmd_benchmark(benchmark_args, output_dir, user_prefs)

    # Write consolidated SUMMARY.md that aggregates every step's key findings.
    try:
        from profine.cli.run_all_summary import write_summary
        summary_path = write_summary(output_dir, script, args.hardware)
        if summary_path:
            ui.success(f"Summary written to: [magenta]{summary_path}[/magenta]")
    except Exception as exc:
        # Summary generation is best-effort; never fail the pipeline because of it.
        print(f"(Summary generation failed: {exc})")

    if rc == 0:
        ui.success(f"Pipeline complete. Results in: [magenta]{output_dir}[/magenta]")
    else:
        ui.error(f"Pipeline finished with errors (exit {rc}). Partial results in: {output_dir}")
    return rc


def _auto_detect_script(output_dir: Path) -> str | None:
    """Try to find script_path from read or profile output."""
    for sub in ("read/architecture_record.json", "profile/profile_record.json"):
        path = output_dir / sub
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("script_path"):
                    return data["script_path"]
            except (json.JSONDecodeError, KeyError):
                continue
    return None
