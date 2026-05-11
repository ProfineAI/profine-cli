"""Guard: every third-party import reached from the Modal-remote entry
point must be installed in the Modal image. Without this, adding a
single `import xyz` to profine.modal.* / profine.config.yaml_loader /
profine.profiler.hooks will silently break every profile / benchmark
run with a ModuleNotFoundError inside the container."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Files that get imported transitively from profine.modal.remote
# inside the Modal container. Anything reachable from here must
# either be stdlib, profine-internal, or in profiling_deps below.
REMOTE_ENTRY_POINTS = [
    "profine/modal/remote.py",
    "profine/modal/executor.py",
    "profine/modal/discovery.py",
    "profine/config/yaml_loader.py",
    "profine/profiler/hooks.py",
]

# pip-name aliases for import names that don't match.
_IMPORT_TO_PIP = {
    "yaml": "pyyaml",
    "pynvml": "nvidia-ml-py",
}

# The image_builder.profiling_deps source of truth, lowercased for matching.
INSTALLED = {"torch", "nvidia-ml-py", "pyyaml", "modal"}

_STDLIB = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()


def _walk_imports() -> set[str]:
    visited: set[Path] = set()
    imports: set[str] = set()

    def scan(path: Path) -> None:
        if path in visited or not path.exists():
            return
        visited.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imports.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])

    for rel in REMOTE_ENTRY_POINTS:
        scan(REPO / rel)
    return imports


def test_remote_runtime_imports_are_installed() -> None:
    """Every third-party import reached from the Modal-remote entry
    must be in the profiling_deps installed by image_builder."""
    imports = _walk_imports()
    third_party = {
        i for i in imports
        if i not in _STDLIB
        and i not in {"profine", "tomli", "tomllib", "typing_extensions"}
    }

    missing = []
    for imp in sorted(third_party):
        pip_name = _IMPORT_TO_PIP.get(imp, imp).lower()
        if pip_name not in INSTALLED:
            missing.append(f"{imp} (pip: {pip_name})")

    assert not missing, (
        "profine.modal.* imports the following third-party packages that "
        f"are NOT in image_builder.profiling_deps: {missing}. Either add "
        "them to profiling_deps or remove the import from the remote path."
    )
