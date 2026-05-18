"""CLI command handlers — one function per subcommand."""

from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

from profine.cli import _console as ui


def _resolve_client_version() -> str:
    """Best-effort lookup of the installed profine package version.

    Returns an empty string when the package isn't installed (e.g.
    dev runs from a non-editable checkout) so telemetry still works;
    the version is informational, not required.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version("profine")
    except PackageNotFoundError:
        return ""
    except Exception:  # noqa: BLE001
        return ""


def _emit_telemetry_after(
    args: Namespace,
    output_dir: Path,
    pipeline_callable,
) -> int:
    """Common wrapper: build a recorder, run the pipeline, emit telemetry,
    close the recorder. Used by cmd_profile, cmd_benchmark, cmd_run_all
    so each contributes a row to the flywheel exactly once.

    If telemetry is enabled but the read step hasn't produced an
    architecture_record.json yet (common when the user invokes
    `profine profile` standalone), we run the reader first so the
    fingerprint can be computed at emit time. Skipped entirely when
    telemetry is off so opted-out users never pay the LLM cost.
    """
    from profine.telemetry import emit_run
    from profine.telemetry.builder import build_recorder

    recorder = build_recorder(args, client_version=_resolve_client_version())
    # run-all does its own read stage; calling the pre-step there would
    # double-bill the LLM and make the "skipped read on resume" line a lie.
    if recorder.enabled and getattr(args, "command", None) != "run-all":
        _ensure_read_output_for_telemetry(args, output_dir)
    try:
        return pipeline_callable()
    finally:
        try:
            emit_run(output_dir, recorder, hardware_name=getattr(args, "hardware", None))
        except Exception:  # noqa: BLE001 — telemetry must never crash the run
            pass
        recorder.close()


def _ensure_read_output_for_telemetry(args: Namespace, output_dir: Path) -> None:
    """Run cmd_read upfront if no architecture_record.json exists.

    `emit_run` needs the architecture record to build a fingerprint;
    without it the run can't contribute to priors. Calling read here
    fills the gap for `profine profile` / `profine benchmark` users
    who skip the explicit read step. Idempotent (no-op when the file
    is already on disk).

    Failures are silent — a missing OPENAI_API_KEY or LLM error here
    simply means this run won't contribute telemetry. We never crash
    the calling pipeline because of the reader.
    """
    arch_path = output_dir / "read" / "architecture_record.json"
    if arch_path.exists():
        return

    script = getattr(args, "script", None)
    if not script or not Path(script).exists():
        return

    print("  [telemetry] running reader for fingerprint…")
    read_args = Namespace(
        script=script,
        provider=getattr(args, "provider", "openai"),
        api_key=getattr(args, "api_key", None),
        model=getattr(args, "model", None),
        base_url=getattr(args, "base_url", None),
        seed=getattr(args, "seed", None),
        output=getattr(args, "output", "profine_output"),
        prefs=getattr(args, "prefs", None),
    )
    try:
        cmd_read(read_args, output_dir, None)
    except Exception as e:  # noqa: BLE001
        print(f"  [telemetry] reader failed ({type(e).__name__}); "
              "telemetry will be skipped for this run.")


def cmd_env(args: Namespace, output_dir: Path, user_prefs: str | None) -> int:
    """Print every PROFINE_* env var the codebase reads, with its
    current resolved value. Secrets are redacted."""
    from profine.env_vars import by_category

    grouped = by_category()
    for category in sorted(grouped):
        print(f"\n# {category}")
        print("-" * (len(category) + 2))
        for entry in grouped[category]:
            value = entry.resolved_by()
            if value is None and entry.default is not None:
                value = f"(default) {entry.default}"
            elif value is None:
                value = "(unset)"
            print(f"  {entry.name}")
            print(f"    description : {entry.description}")
            print(f"    current     : {value}")
            if entry.referenced_in:
                refs = ", ".join(entry.referenced_in[:3])
                if len(entry.referenced_in) > 3:
                    refs += f", +{len(entry.referenced_in) - 3} more"
                print(f"    referenced  : {refs}")
    print()
    return 0


def cmd_auth(args: Namespace, output_dir: Path, user_prefs: str | None) -> int:
    """Manage saved API keys in ~/.profine/auth.json.

    `output_dir` and `user_prefs` are unused; kept for dispatcher symmetry.
    """
    import getpass

    from profine import auth

    action = args.action

    if action == "status":
        saved = auth.load()
        path = auth.auth_path()
        if not saved:
            print(f"No saved credentials. ({path} is empty or missing.)")
            print("Run `profine auth login` to add some.")
            return 0
        print(f"Saved credentials ({path}):")
        for name in auth.MANAGED_KEYS:
            if name in saved:
                env_set = bool(os.environ.get(name))
                marker = "  (env var also set — env wins)" if env_set else ""
                print(f"  {name:<22} {auth.redact(saved[name])}{marker}")
        return 0

    if action == "login":
        print("Paste each credential (leave blank to skip; existing values shown redacted).")
        print(f"Saved to: {auth.auth_path()}\n")
        existing = auth.load()
        saved_now = 0
        for name in auth.MANAGED_KEYS:
            current = existing.get(name)
            hint = f" [{auth.redact(current)}]" if current else ""
            try:
                value = getpass.getpass(f"  {name}{hint}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return 1
            if value:
                auth.save_key(name, value)
                saved_now += 1
        if saved_now:
            print(f"\nSaved {saved_now} credential(s) to {auth.auth_path()}.")
        else:
            print("\nNo changes.")
        return 0

    if action == "set":
        name = args.key
        if not name:
            print("Usage: profine auth set <KEY> [VALUE]")
            print(f"Known keys: {', '.join(auth.MANAGED_KEYS)}")
            return 1
        if name not in auth.MANAGED_KEYS:
            print(f"Unknown key '{name}'. Known: {', '.join(auth.MANAGED_KEYS)}")
            return 1
        value = args.value
        if value is None:
            try:
                value = getpass.getpass(f"{name}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return 1
        if not value:
            print("Empty value; not saved.")
            return 1
        auth.save_key(name, value)
        print(f"Saved {name} to {auth.auth_path()}.")
        return 0

    if action == "logout":
        name = args.key
        if name is None:
            if auth.clear_all():
                print(f"Cleared {auth.auth_path()}.")
            else:
                print("Nothing to clear.")
            return 0
        if name not in auth.MANAGED_KEYS:
            print(f"Unknown key '{name}'. Known: {', '.join(auth.MANAGED_KEYS)}")
            return 1
        if auth.clear_key(name):
            print(f"Removed {name}.")
        else:
            print(f"{name} was not saved.")
        return 0

    print(f"Unknown auth action: {action}")
    return 1


def cmd_telemetry(args: Namespace, output_dir: Path, user_prefs: str | None) -> int:
    """Manage anonymous telemetry consent.

    `output_dir` and `user_prefs` are unused here — the subcommand
    only touches `~/.profine/telemetry_consent.json`. They're in the
    signature for dispatcher symmetry with every other cmd_*.
    """
    from profine.telemetry.consent import (
        consent_path,
        env_opted_out,
        load_consent,
        save_consent,
    )

    action = args.action

    if action == "status":
        record = load_consent()
        if record is None:
            print("Telemetry consent: not yet decided.")
            print("Next interactive `profine` run will prompt you.")
        elif record.granted:
            print(f"Telemetry consent: GRANTED (install_id: {record.install_id})")
            print(f"Stored at: {consent_path()}")
        else:
            print("Telemetry consent: DECLINED")
            print(f"Stored at: {consent_path()}")
        if env_opted_out():
            print("Note: PROFINE_NO_TELEMETRY is set in the environment — "
                  "telemetry is disabled regardless of the stored consent.")
        if os.environ.get("PROFINE_API_KEY"):
            print("Note: PROFINE_API_KEY is set — runs send paid telemetry "
                  "via that key, bypassing OSS consent.")
        return 0

    if action == "enable":
        record = save_consent(True)
        print(f"Telemetry enabled. install_id: {record.install_id}")
        print(f"Stored at: {consent_path()}")
        return 0

    if action == "disable":
        save_consent(False)
        print("Telemetry disabled. No data will be sent.")
        print(f"Stored at: {consent_path()}")
        return 0

    if action == "doctor":
        return _cmd_telemetry_doctor()

    print(f"Unknown telemetry action: {action}")
    return 1


def _cmd_telemetry_doctor() -> int:
    """Synchronously probe the telemetry endpoint and print a verdict.

    Reports consent state, endpoint URL, and the result of one POST
    attempt with a minimal probe payload. Bypasses the background-thread
    machinery so failures are visible instead of swallowed.
    """
    import time
    from urllib.error import URLError
    from urllib.request import Request, urlopen
    from profine.telemetry.consent import load_consent, env_opted_out
    from profine.telemetry.recorder import _HTTP_TIMEOUT_SECONDS

    record = load_consent()
    if record is None:
        print("Telemetry: consent not yet decided. Run `profine telemetry enable` first.")
        return 1
    if not record.granted:
        print("Telemetry: consent DECLINED. Nothing will be sent until you run "
              "`profine telemetry enable`.")
        return 1
    if env_opted_out():
        print("Telemetry: PROFINE_NO_TELEMETRY is set — runtime opt-out is in effect.")
        return 1

    api_url = "https://api.profine.ai"
    endpoint = f"{api_url}/api/telemetry/anon"
    print(f"Telemetry doctor")
    print(f"  install_id: {record.install_id}")
    print(f"  endpoint:   {endpoint}")
    print(f"  timeout:    {_HTTP_TIMEOUT_SECONDS:.0f}s per attempt, 1 retry")

    # Send a shape the server will accept (matches _drain_payload's schema).
    # The probe deliberately uses a synthetic fingerprint_hash so probes are
    # distinguishable from real runs by anyone querying the DB.
    import hashlib
    probe_marker = f"doctor_probe_{record.install_id}"
    fp_hash = hashlib.sha256(probe_marker.encode()).hexdigest()
    payload = json.dumps({
        "client_version": "doctor-probe",
        "install_id": record.install_id,
        "fingerprint": {
            "arch_class": "probe",
            "param_bucket": "probe",
            "hardware_class": "probe",
            "precision": "probe",
            "optimizer_class": "probe",
            "has_compile": False,
            "has_distributed": False,
            "fingerprint_hash": fp_hash,
            "framework": "probe",
        },
        "outcomes": [],
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    for attempt in (1, 2):
        t0 = time.perf_counter()
        try:
            req = Request(endpoint, data=payload, headers=headers, method="POST")
            with urlopen(req, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
                body = resp.read(512)
                dt = time.perf_counter() - t0
                print(f"  attempt {attempt}: HTTP {resp.status} in {dt:.2f}s")
                if body:
                    print(f"    body: {body[:200]!r}")
                print("Result: OK — endpoint reachable and accepted the probe.")
                return 0
        except URLError as e:
            dt = time.perf_counter() - t0
            print(f"  attempt {attempt}: FAILED in {dt:.2f}s ({type(e).__name__}: {e})")
            if attempt == 1:
                print(f"  (backend may have been asleep; retrying after 2s)")
                time.sleep(2.0)
        except Exception as e:  # noqa: BLE001
            dt = time.perf_counter() - t0
            print(f"  attempt {attempt}: FAILED in {dt:.2f}s ({type(e).__name__}: {e})")
            break

    print("Result: FAILED — telemetry rows from this machine are being dropped.")
    print("Common causes: backend asleep (free-tier Render dyno), DNS/TLS issue, "
          "or local firewall blocking outbound HTTPS to api.profine.ai.")
    return 1


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
    return _emit_telemetry_after(
        args, output_dir,
        lambda: _cmd_profile_body(args, output_dir, user_prefs),
    )


def _cmd_profile_body(args: Namespace, output_dir: Path, user_prefs: str | None) -> int:
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

    if args.arch_dir:
        arch_path = Path(args.arch_dir) / "architecture_record.json"
    else:
        arch_path = interpret_dir.parent / "read" / "architecture_record.json"
    arch_data = json.loads(arch_path.read_text(encoding="utf-8")) if arch_path.exists() else {}

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

    script_path = Path(args.script).resolve()
    source = script_path.read_text(encoding="utf-8")
    arch_path = output_dir / "read" / "architecture_record.json"
    arch_data = json.loads(arch_path.read_text(encoding="utf-8")) if arch_path.exists() else None

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

        # Treat a failed edit on one candidate as "skipped, keep going" so
        # a single bad LLM response doesn't throw away the speedups already
        # stacked earlier in this run.
        try:
            result = editor.edit(
                current_source, candidate, arch_data, user_prefs,
                entry_path=entry_rel, local_modules=effective_locals,
                debug_dir=output_dir / "_debug",
            )
        except Exception as exc:  # noqa: BLE001 — narrowed via the skipped path
            reason = f"editor raised {type(exc).__name__}: {exc}"
            skipped.append((candidate.entry_id, reason))
            print(f"  skipped: {reason}")
            continue

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
            # Snapshot the cumulative pipeline state after this iteration so
            # the benchmarker's auto-peel can rewind to any prior point if a
            # later optimization (or this one) crashes at runtime.
            _save_cumulative_snapshot(
                iter_dir / "cumulative",
                entry_source=current_source,
                extras=cumulative_extras,
                applied_ids=applied_ids,
            )
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


def _save_cumulative_snapshot(
    snap_dir: Path,
    *,
    entry_source: str,
    extras: dict[str, str],
    applied_ids: list[str],
) -> None:
    """Write the cumulative editor state after one iteration so auto-peel
    can copy it back over `<output>/edit/` to undo later optimizations."""
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "edited_train.py").write_text(entry_source, encoding="utf-8")
    if extras:
        files_dir = snap_dir / "files"
        for rel, content in extras.items():
            target = files_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
    (snap_dir / "applied_ids.json").write_text(
        json.dumps(applied_ids), encoding="utf-8",
    )


def _restore_cumulative_snapshot(snap_dir: Path, edit_out: Path) -> list[str]:
    """Copy a snapshot onto `<output>/edit/` and return its applied_ids."""
    import shutil

    edit_out.mkdir(parents=True, exist_ok=True)
    edited_path = edit_out / "edited_train.py"
    shutil.copyfile(snap_dir / "edited_train.py", edited_path)

    files_dir = edit_out / "files"
    if files_dir.exists():
        shutil.rmtree(files_dir)
    snap_files = snap_dir / "files"
    if snap_files.exists():
        shutil.copytree(snap_files, files_dir)

    applied_path = snap_dir / "applied_ids.json"
    if applied_path.exists():
        return json.loads(applied_path.read_text(encoding="utf-8"))
    return []


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
    return _emit_telemetry_after(
        args, output_dir,
        lambda: _cmd_benchmark_body(args, output_dir, user_prefs),
    )


def _cmd_benchmark_body(args: Namespace, output_dir: Path, user_prefs: str | None) -> int:
    from profine.benchmarker.benchmarker import Benchmarker
    from profine.schema.hardware import ModalRuntimeConfig
    from profine.config.settings import DEFAULTS

    if not args.script:
        args.script = _auto_detect_script(output_dir)
        if not args.script:
            print("Error: no script specified and could not auto-detect from output directory.\n"
                  "Either pass the script path or run `profine read` first.")
            return 1
        print(f"Auto-detected script: {args.script}")

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
            # When --edit-dir points outside output_dir (re-benchmarking an
            # existing pipeline's edits), the suggest report lives next to
            # edit_dir, not under output_dir.
            if not sugg.exists():
                sugg = edit_dir.parent / "suggest" / "suggestion_report.json"
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
    return _emit_telemetry_after(
        args, output_dir,
        lambda: _cmd_run_all_pipeline(args, output_dir, user_prefs),
    )


_STAGE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "read":      ("read/architecture_record.json",),
    "profile":   ("profile/profile_record.json",),
    "interpret": ("interpret/bottleneck_report.json",),
    "suggest":   ("suggest/suggestion_report.json",),
    "edit":      ("edit/edited_train.py", "edit/change_manifest.json"),
}
_LLM_STAGES: tuple[str, ...] = ("read", "interpret", "suggest", "edit")

_PREFLIGHT_MINUTES_PER_MODAL_RUN = float(
    os.environ.get("PROFINE_PREFLIGHT_MINUTES_PER_RUN", "5")
)
_BENCHMARK_MODAL_RUNS = 2
_DEFAULT_TOP_N = 10
_COST_PROMPT_THRESHOLD_USD = float(os.environ.get("PROFINE_COST_PROMPT_THRESHOLD", "5"))


def _stage_done(output_dir: Path, stage: str) -> bool:
    return all((output_dir / rel).exists() for rel in _STAGE_ARTIFACTS[stage])


def _preflight_estimate(
    args: Namespace,
    output_dir: Path,
    skipped: list[str] | None = None,
) -> int:
    """Print a one-line cost summary before run-all kicks off.

    Pass `skipped` to mark stages already done at pipeline entry so we
    don't claim a stage was resumed when it just ran moments earlier.
    """
    import sys
    from profine.schema.hardware import get_hardware

    try:
        hw = get_hardware(args.hardware)
        hw_label = getattr(hw, "name", args.hardware)
        cost_per_hour = float(getattr(hw, "cost_per_hour", 0.0) or 0.0)
    except Exception:
        hw_label = args.hardware
        cost_per_hour = 0.0

    no_resume = bool(getattr(args, "no_resume", False))
    if skipped is None:
        skipped = [
            stage for stage in _STAGE_ARTIFACTS
            if not no_resume and _stage_done(output_dir, stage)
        ]
    llm_calls = sum(1 for s in _LLM_STAGES if s not in skipped)
    modal_runs = (0 if "profile" in skipped else 1) + _BENCHMARK_MODAL_RUNS
    minutes = modal_runs * _PREFLIGHT_MINUTES_PER_MODAL_RUN
    dollars = minutes / 60.0 * cost_per_hour

    summary = (
        f"Estimated cost: ~${dollars:.2f} on [bold]{hw_label}[/bold] "
        f"(~{minutes:.0f} min, {modal_runs} GPU run{'s' if modal_runs != 1 else ''}, "
        f"{llm_calls} LLM call{'s' if llm_calls != 1 else ''})"
    )
    if skipped:
        summary += f"  ·  resuming, skipping [dim]{', '.join(skipped)}[/dim]"
    ui.info(summary)

    if dollars < _COST_PROMPT_THRESHOLD_USD:
        return 0
    if getattr(args, "yes", False):
        return 0
    if not sys.stdin.isatty():
        return 0
    ui.info(f"[yellow]Estimate exceeds ${_COST_PROMPT_THRESHOLD_USD:.0f} threshold.[/yellow]")
    try:
        reply = input("Continue? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ui.error("Aborted.")
        return 1
    if reply in ("", "y", "yes"):
        return 0
    ui.error("Aborted by user.")
    return 1


def _cmd_run_all_pipeline(args: Namespace, output_dir: Path, user_prefs: str | None) -> int:
    """Internal pipeline body. Same logic as before; recorder wrapping
    lives in cmd_run_all() so this stays focused on orchestration."""
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

    no_resume = bool(getattr(args, "no_resume", False))
    # Snapshot at entry: the cost summary further down must reflect what
    # was on disk *before* we ran anything, not after read writes to it.
    initially_done = [
        stage for stage in _STAGE_ARTIFACTS
        if not no_resume and _stage_done(output_dir, stage)
    ]

    def _skip_if_done(idx: int, stage: str) -> bool:
        if no_resume or not _stage_done(output_dir, stage):
            return False
        _step_header(idx, *steps[idx - 1])
        first_artifact = output_dir / _STAGE_ARTIFACTS[stage][0]
        ui.success(f"Skipped — resuming from existing {stage}: [dim]{first_artifact}[/dim]")
        return True

    if not _skip_if_done(1, "read"):
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

    rc = _preflight_estimate(args, output_dir, skipped=initially_done)
    if rc != 0:
        return rc

    if not _skip_if_done(2, "profile"):
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
        # Skip the inner telemetry wrapper — run-all owns the single batch.
        rc = _cmd_profile_body(profile_args, output_dir, user_prefs)
        if rc != 0:
            ui.error("Profile failed — aborting pipeline.")
            return rc

    profile_record = output_dir / "profile" / "profile_record.json"
    if profile_record.exists():
        data = json.loads(profile_record.read_text(encoding="utf-8"))
        if data.get("status") == "crash":
            print(f"\nProfile crashed: {data.get('error', 'unknown')[:200]}")
            ui.error("Cannot continue without valid profile data — aborting pipeline.")
            return 1

    if not _skip_if_done(3, "interpret"):
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

    if not _skip_if_done(4, "suggest"):
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

    if not _skip_if_done(5, "edit"):
        _step_header(5, *steps[4])
        top_n = getattr(args, "top", None)
        if top_n is None:
            sugg_path = output_dir / "suggest" / "suggestion_report.json"
            if sugg_path.exists():
                sugg = json.loads(sugg_path.read_text(encoding="utf-8"))
                top_n = len(sugg.get("candidates", []))
            else:
                top_n = _DEFAULT_TOP_N
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
    rc = _cmd_benchmark_body(benchmark_args, output_dir, user_prefs)
    rc, peeled = _auto_peel_on_crash(rc, output_dir, benchmark_args, user_prefs)
    if peeled:
        ui.info(
            f"[yellow]Auto-peeled {len(peeled)} optimization(s) that crashed at runtime: "
            f"{', '.join(peeled)}[/yellow]"
        )

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


def _auto_peel_on_crash(
    rc: int,
    output_dir: Path,
    benchmark_args: Namespace,
    user_prefs: str | None,
) -> tuple[int, list[str]]:
    """Drop the last applied optimization and re-benchmark until success
    or only one optimization remains. Returns (final_rc, peeled_ids)."""
    if rc == 0:
        return rc, []
    edit_dir = output_dir / "edit"
    manifest_path = edit_dir / "change_manifest.json"
    if not manifest_path.exists():
        return rc, []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return rc, []
    applied = list(manifest.get("applied_ids") or [])
    if len(applied) <= 1:
        return rc, []

    peeled: list[str] = []
    while len(applied) > 1 and rc != 0:
        bad = applied[-1]
        snap_dir = _find_snapshot_for(edit_dir, applied[:-1])
        if snap_dir is None:
            break
        new_applied = _restore_cumulative_snapshot(snap_dir, edit_dir)
        ui.info(
            f"[yellow]Optimized run crashed; auto-peeling [bold]{bad}[/bold] "
            f"and retrying benchmark with {len(new_applied)} remaining "
            f"optimization(s)…[/yellow]"
        )
        peeled.append(bad)
        applied = new_applied
        rc = _cmd_benchmark_body(benchmark_args, output_dir, user_prefs)

    if peeled:
        manifest["applied_ids"] = applied
        manifest.setdefault("auto_peeled", []).extend(peeled)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return rc, peeled


def _find_snapshot_for(edit_dir: Path, target_applied: list[str]) -> Path | None:
    """Return the iteration snapshot dir whose cumulative applied_ids match."""
    if not target_applied:
        return None
    for sub in sorted(edit_dir.glob("*_*/cumulative")):
        ids_path = sub / "applied_ids.json"
        if not ids_path.exists():
            continue
        try:
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if ids == target_applied:
            return sub
    return None


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
