"""Code Editor orchestrator (plan 4.5).

Takes an optimization candidate + source code, uses the LLM to produce
the edited source, validates it, and returns a structured result.

Usage:
    from profine.editor.editor import CodeEditor

    editor = CodeEditor(provider="openai")
    result = editor.edit(source, candidate, architecture_record)

    print(result.diff)
    print(result.explanation)
    result.save("output/")
"""

from __future__ import annotations

import ast
import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from profine.llm.backend import LlmBackend, create_backend
from profine.llm.utils import call_and_parse, parse_json_response
from profine.schema.optimization_candidate import OptimizationCandidate
from profine.editor.prompts import (
    SYSTEM_PROMPT,
    HEALING_SYSTEM,
    build_edit_prompt,
    build_healing_prompt,
)


@dataclass(slots=True)
class ChangeEntry:
    """A single code change within the edit."""
    line_start: int = 0
    line_end: int = 0
    description: str = ""
    original_snippet: str = ""
    new_snippet: str = ""
    path: str = ""  # Empty = entry script; otherwise project-relative path.


@dataclass(slots=True)
class FileEdit:
    """Edit applied to a non-entry project file."""
    path: str
    original_source: str
    edited_source: str
    diff: str = ""


@dataclass(slots=True)
class EditResult:
    """Output of the Code Editor tool."""
    original_source: str
    edited_source: str
    applied: bool = False
    changes: list[ChangeEntry] = field(default_factory=list)
    explanation: str = ""
    diff: str = ""
    new_imports: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    not_applicable_reason: str = ""
    optimization_id: str = ""
    extra_file_edits: list[FileEdit] = field(default_factory=list)

    def save(self, output_dir: str | Path, filename: str = "edited_train.py") -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        edited_path = out / filename
        diff_path = out / f"{filename}.diff"
        manifest_path = out / "change_manifest.json"

        edited_path.write_text(self.edited_source, encoding="utf-8")
        diff_path.write_text(self.diff, encoding="utf-8")
        manifest_path.write_text(
            json.dumps(_result_to_dict(self), indent=2, default=str),
            encoding="utf-8",
        )

        # Persist edits to other project files under output_dir/files/<rel>
        # so the user can review them without us touching their checkout.
        extra_paths: list[Path] = []
        if self.extra_file_edits:
            files_dir = out / "files"
            for fe in self.extra_file_edits:
                rel = fe.path.lstrip("/")
                target = files_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(fe.edited_source, encoding="utf-8")
                diff_target = target.with_suffix(target.suffix + ".diff")
                diff_target.write_text(fe.diff, encoding="utf-8")
                extra_paths.append(target)

        result = {"edited": edited_path, "diff": diff_path, "manifest": manifest_path}
        if extra_paths:
            result["extra_files"] = files_dir
        return result


class CodeEditor:
    """LLM-driven code editor for applying optimizations."""

    def __init__(
        self,
        provider: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_heal_attempts: int = 2,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
        self._backend = create_backend(provider, **kwargs)
        self._max_heal_attempts = max_heal_attempts

    def edit(
        self,
        source: str,
        candidate: OptimizationCandidate,
        architecture_record: dict[str, Any] | None = None,
        user_preferences: str | None = None,
        entry_path: str | None = None,
        local_modules: dict[str, str] | None = None,
        *,
        debug_dir: str | Path | None = None,
    ) -> EditResult:
        """Apply a single optimization to source code.

        Args:
            source: Entry script source.
            candidate: The optimization to apply.
            architecture_record: Output from the Read Code tool (optional).
            user_preferences: Free-form user preferences (optional).
            entry_path: Project-relative path to the entry script (optional).
            local_modules: Project-relative path → source for other local
                modules the entry script imports. Letting the LLM see
                them lets it edit, for example, an imported Trainer or
                model class that owns the forward/backward step.

        Returns:
            EditResult with edited source, diff, and explanation.
        """
        user_msg = build_edit_prompt(
            source, candidate, architecture_record, user_preferences,
            entry_path=entry_path, local_modules=local_modules,
        )
        parsed = call_and_parse(
            self._backend, SYSTEM_PROMPT, user_msg,
            debug_dir=debug_dir, debug_label=f"editor_response_{candidate.entry_id}",
        )

        edited_source = parsed.get("edited_source", source)
        applied = parsed.get("applied", False)

        # Validate the edited entry source parses
        if applied:
            edited_source = self._validate_and_heal(edited_source)

        # Multi-file edits: validate each, drop unchanged ones (LLM
        # sometimes echoes the input), and skip anything that doesn't
        # match a known local module so a hallucinated path can't write
        # outside the editor's output directory.
        extra_edits: list[FileEdit] = []
        warnings: list[str] = list(parsed.get("warnings", []))
        for fe in parsed.get("file_edits", []) or []:
            path = (fe.get("path") or "").strip()
            new_src = fe.get("edited_source", "")
            if not path or not new_src:
                continue
            if local_modules is None or path not in local_modules:
                warnings.append(f"Ignored edit to unknown file: {path}")
                continue
            original = local_modules[path]
            if new_src == original:
                continue
            healed = self._validate_and_heal(new_src) if applied else new_src
            extra_edits.append(FileEdit(
                path=path,
                original_source=original,
                edited_source=healed,
                diff=_compute_diff(original, healed, from_label=path, to_label=f"{path} (optimized)"),
            ))

        # Guardrail against entry-script restructuring.
        #
        # When the LLM also returns file_edits, it is supposed to leave
        # the entry script alone (or only add a top-level import / flag).
        # In practice it sometimes "helps" by inlining the very classes
        # it just edited in the Local Module — duplicating Trainer/GPT
        # bodies into the entry. That breaks the run two ways: the
        # benchmarker now sees optimized_source != baseline_source and
        # re-instruments a totally different script, and the inlined
        # copy bypasses the overlay edits entirely. Detect that pattern
        # and revert the entry source to the original so only the
        # file_edits take effect.
        if applied and extra_edits and edited_source != source:
            if _is_structural_rewrite(source, edited_source):
                warnings.append(
                    "Entry script was structurally rewritten alongside file_edits "
                    "(likely inlining of Local Module code) — reverted to the original "
                    "entry source. Only the file_edits will be applied."
                )
                edited_source = source

        # Compute diff for entry script
        diff = _compute_diff(source, edited_source)

        # Build change entries
        changes = [
            ChangeEntry(
                line_start=c.get("line_start", 0),
                line_end=c.get("line_end", 0),
                description=c.get("description", ""),
                original_snippet=c.get("original_snippet", ""),
                new_snippet=c.get("new_snippet", ""),
                path=c.get("path", "") or "",
            )
            for c in parsed.get("changes", [])
        ]

        return EditResult(
            original_source=source,
            edited_source=edited_source,
            applied=applied,
            changes=changes,
            explanation=parsed.get("explanation", ""),
            diff=diff,
            new_imports=parsed.get("new_imports", []),
            warnings=warnings,
            not_applicable_reason=parsed.get("not_applicable_reason", ""),
            optimization_id=candidate.entry_id,
            extra_file_edits=extra_edits,
        )

    def edit_multiple(
        self,
        source: str,
        candidates: list[OptimizationCandidate],
        architecture_record: dict[str, Any] | None = None,
        user_preferences: str | None = None,
    ) -> list[EditResult]:
        """Apply multiple optimizations sequentially.

        Each optimization is applied to the output of the previous one.
        Returns a list of EditResults, one per candidate.
        """
        results: list[EditResult] = []
        current_source = source

        for candidate in candidates:
            result = self.edit(current_source, candidate, architecture_record, user_preferences)
            results.append(result)
            if result.applied:
                current_source = result.edited_source

        return results

    def _validate_and_heal(self, source: str) -> str:
        """Validate that edited source parses as valid Python. Heal if not."""
        for attempt in range(self._max_heal_attempts + 1):
            error = _check_syntax(source)
            if error is None:
                return source
            if attempt < self._max_heal_attempts:
                source = self._heal(source, error)
        return source  # return best effort even if still broken

    def _heal(self, broken_source: str, error: str) -> str:
        """Ask the LLM to fix a broken edit."""
        user_msg = build_healing_prompt(broken_source, error)
        raw = self._backend.call(HEALING_SYSTEM, user_msg)
        parsed = _parse_response(raw)
        return parsed.get("edited_source", broken_source)


def _parse_response(raw: str) -> dict[str, Any]:
    """Parse the editor LLM response — see profine.llm.utils."""
    return parse_json_response(raw)


def _check_syntax(source: str) -> str | None:
    """Return None if source parses, or the error message if not."""
    try:
        ast.parse(source)
        return None
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"


def _is_structural_rewrite(original: str, edited: str) -> bool:
    """Heuristic: did the LLM restructure the entry script rather than
    just add a few lines?

    We trip this guardrail when the diff drops more than a handful of
    non-blank lines, OR when the edited source contains class/def
    definitions whose names didn't exist in the original. Both signal
    that the LLM inlined code from a Local Module (duplicating it)
    rather than just adding an import or wrapping a call.

    Pure additions (new imports, a guard flag) keep the deletions near
    zero and reuse only existing names, so they pass.
    """
    import re
    orig_nonblank = [l for l in original.splitlines() if l.strip()]
    new_nonblank = [l for l in edited.splitlines() if l.strip()]
    orig_set = set(orig_nonblank)
    new_set = set(new_nonblank)

    deletions = [l for l in orig_nonblank if l not in new_set]
    if len(deletions) > 3:
        return True

    def_pat = re.compile(r"^\s*(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)")
    orig_defs = {m.group(1) for l in original.splitlines() if (m := def_pat.match(l))}
    new_defs = {m.group(1) for l in edited.splitlines() if (m := def_pat.match(l))}
    if new_defs - orig_defs:
        return True

    return False


def _compute_diff(
    original: str,
    edited: str,
    from_label: str = "original",
    to_label: str = "optimized",
) -> str:
    """Compute a unified diff between original and edited source."""
    original_lines = original.splitlines(keepends=True)
    edited_lines = edited.splitlines(keepends=True)
    diff = difflib.unified_diff(
        original_lines, edited_lines,
        fromfile=from_label, tofile=to_label,
        lineterm="",
    )
    return "".join(diff)


def _result_to_dict(result: EditResult) -> dict[str, Any]:
    """Convert EditResult to a JSON-serializable dict (excluding full source)."""
    return {
        "applied": result.applied,
        "optimization_id": result.optimization_id,
        "explanation": result.explanation,
        "changes": [
            {
                "path": c.path,
                "line_start": c.line_start,
                "line_end": c.line_end,
                "description": c.description,
                "original_snippet": c.original_snippet,
                "new_snippet": c.new_snippet,
            }
            for c in result.changes
        ],
        "new_imports": result.new_imports,
        "warnings": result.warnings,
        "not_applicable_reason": result.not_applicable_reason,
        "extra_file_edits": [
            {"path": fe.path, "diff": fe.diff}
            for fe in result.extra_file_edits
        ],
    }
