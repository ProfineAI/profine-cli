"""Read Code orchestrator (plan section 4.1).

Usage:
    from profine.reader.reader import CodeReader

    reader = CodeReader(provider="openai")  # or "anthropic"
    result = reader.read("path/to/train.py")

    print(result.markdown_brief)          # human-readable
    print(result.architecture_record)     # dict following the schema
    result.save("output_dir/")            # writes both files
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from profine.reader.extractor import CodeFacts, extract
from profine.reader.llm_analyzer import analyze

# Caps on sibling-module source fed to the reader LLM so the prompt
# stays inside any gpt-4o-class context window.
_LOCAL_MODULES_MAX_FILES = int(os.environ.get("PROFINE_READER_MAX_FILES", "12"))
_LOCAL_MODULES_MAX_CHARS = int(os.environ.get("PROFINE_READER_MAX_CHARS", "50000"))


@dataclass(slots=True)
class ReadResult:
    """Output of the code reader: both human and machine representations."""
    architecture_record: dict[str, Any]
    markdown_brief: str
    facts: CodeFacts
    source: str
    warnings: list[str] = field(default_factory=list)

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        """Write architecture_record.json and architecture_brief.md to output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        record_path = out / "architecture_record.json"
        brief_path = out / "architecture_brief.md"

        record_path.write_text(
            json.dumps(self.architecture_record, indent=2, default=str),
            encoding="utf-8",
        )
        brief_path.write_text(self.markdown_brief, encoding="utf-8")

        return {"record": record_path, "brief": brief_path}

    @property
    def guessed_fields(self) -> list[str]:
        """Return field names where confidence == 'guessed'."""
        guessed = []
        for key, val in self.architecture_record.items():
            if isinstance(val, dict) and val.get("confidence") == "guessed":
                guessed.append(key)
            elif isinstance(val, dict) and not val.get("confidence"):
                # Check nested (optimizer, dataloader, etc.)
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, dict) and sub_val.get("confidence") == "guessed":
                        guessed.append(f"{key}.{sub_key}")
        return guessed


class CodeReader:
    """Main entry point for the Read Code tool."""

    def __init__(
        self,
        provider: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.provider = provider
        self._llm_kwargs: dict[str, Any] = {}
        if api_key:
            self._llm_kwargs["api_key"] = api_key
        if model:
            self._llm_kwargs["model"] = model
        if base_url:
            self._llm_kwargs["base_url"] = base_url
        if seed is not None:
            self._llm_kwargs["seed"] = seed

    def read(
        self,
        script_path: str | Path,
        *,
        debug_dir: str | Path | None = None,
    ) -> ReadResult:
        """Read and analyze a single training script.

        Args:
            script_path: Path to the Python file to analyze.
            debug_dir: If set, malformed LLM responses are dumped here for
                inspection. Recommend passing the run's output directory.

        Returns:
            ReadResult with both human and machine outputs.
        """
        path = Path(script_path)
        source = path.read_text(encoding="utf-8")
        file_name = path.name

        facts = extract(source, file_name)

        # Sibling modules often hold the defaults the entry script relies
        # on. Feeding them in stops the LLM from guessing those fields.
        local_modules: dict[str, str] = {}
        try:
            from profine.modal.discovery import discover_local_modules
            local_modules = discover_local_modules(
                path,
                max_files=_LOCAL_MODULES_MAX_FILES,
                max_total_chars=_LOCAL_MODULES_MAX_CHARS,
            )
        except Exception:
            local_modules = {}

        record, brief = analyze(
            source, facts, provider=self.provider,
            debug_dir=debug_dir, local_modules=local_modules,
            **self._llm_kwargs,
        )

        record["script_path"] = str(path)

        # Enrich with HF Hub ground truth when a model id can be resolved.
        model_id = _extract_model_id(record)
        if model_id:
            from profine.reader.hf_config import enrich_record
            upgraded = enrich_record(record, model_id)
            if upgraded:
                print(f"  HF Hub enrichment: upgraded {', '.join(upgraded)}")

        warnings = _collect_warnings(record)

        return ReadResult(
            architecture_record=record,
            markdown_brief=brief,
            facts=facts,
            source=source,
            warnings=warnings,
        )


def _extract_model_id(record: dict[str, Any]) -> str | None:
    """Extract a HuggingFace model ID from the architecture record.

    Checks model_variable.value first, then scans evidence snippets
    for an org/model pattern (the LLM sometimes puts the Python variable
    name in value rather than the HF model ID string).
    """
    from profine.reader.hf_config import is_hf_model_id

    model_var = record.get("model_variable", {})
    if isinstance(model_var, dict):
        value = model_var.get("value", "")
        if isinstance(value, str) and is_hf_model_id(value):
            return value
        for ev in model_var.get("evidence", []):
            snippet = ev.get("snippet", "")
            for match in re.findall(r'["\']([A-Za-z0-9_-]+/[A-Za-z0-9._-]+)["\']', snippet):
                if is_hf_model_id(match):
                    return match

    for field in ("model_class", "model_family"):
        obj = record.get(field, {})
        if isinstance(obj, dict):
            for ev in obj.get("evidence", []):
                snippet = ev.get("snippet", "")
                for match in re.findall(r'["\']([A-Za-z0-9_-]+/[A-Za-z0-9._-]+)["\']', snippet):
                    if is_hf_model_id(match):
                        return match

    return None


def _collect_warnings(record: dict[str, Any]) -> list[str]:
    """Scan the record for guessed fields and build user warnings."""
    warnings: list[str] = []
    for key, val in record.items():
        if isinstance(val, dict) and val.get("confidence") == "guessed":
            notes = val.get("notes", "")
            msg = f"'{key}' is a guess"
            if notes:
                msg += f": {notes}"
            warnings.append(msg)
        elif isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, dict) and sub_val.get("confidence") == "guessed":
                    notes = sub_val.get("notes", "")
                    msg = f"'{key}.{sub_key}' is a guess"
                    if notes:
                        msg += f": {notes}"
                    warnings.append(msg)
    return warnings
