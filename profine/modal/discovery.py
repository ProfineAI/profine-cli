"""Project and dependency discovery for Modal execution.

Finds the project root, discovers dependencies from requirements.txt
or pyproject.toml, and resolves the Python version. Falls back to
AST import scanning when no manifest file exists.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from profine.config.yaml_loader import get_import_to_pip

_PROJECT_MARKERS = (".git", "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt")

# Standard library modules — prefer sys.stdlib_module_names (3.10+)
_STDLIB_MODULES: set[str] = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()

# Import name → pip package (loaded from config/patterns.yaml)
_IMPORT_TO_PIP: dict[str, str] = get_import_to_pip()


def discover_project_root(script_path: Path) -> Path:
    """Walk up from script_path looking for a project root marker."""
    current = script_path.resolve().parent
    for _ in range(20):
        if any((current / m).exists() for m in _PROJECT_MARKERS):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return script_path.resolve().parent


def _script_belongs_to_project(script_path: Path, project_root: Path) -> bool:
    """Check if a script is part of the project rooted at project_root.

    Uses the pyproject.toml [project.name] to determine the package
    directory. If the script isn't under that package, it's likely an
    unrelated script vendored into the repo (e.g. an example training
    script that ships its own dependencies).
    """
    resolved = script_path.resolve()
    # Script directly in the project root — probably belongs
    if resolved.parent == project_root:
        return True

    pyproject = project_root / "pyproject.toml"
    if not pyproject.exists():
        return True

    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            return True

    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project_name = data.get("project", {}).get("name", "")
    if not project_name:
        return True  # No name to check against

    pkg_dir_name = project_name.replace("-", "_")
    try:
        rel = resolved.relative_to(project_root)
        return rel.parts[0] == pkg_dir_name
    except ValueError:
        return False


def discover_dependencies(script_path: Path) -> list[str]:
    """Discover Python dependencies from requirements.txt, pyproject.toml, or imports.

    Walks up from the script directory looking for dependency files,
    but never above the project root (.git boundary) to avoid picking
    up unrelated manifests in parent directories.
    """
    script_dir = script_path.resolve().parent
    project_root = discover_project_root(script_path)

    current = script_dir
    for _ in range(20):
        req_file = current / "requirements.txt"
        if req_file.exists():
            return _parse_requirements_txt(req_file)

        pyproject = current / "pyproject.toml"
        if pyproject.exists() and _script_belongs_to_project(script_path, current):
            deps = _parse_pyproject_deps(pyproject)
            if deps:
                return deps

        # Don't walk above the project root
        if current == project_root:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Fallback: scan script imports
    return discover_imports(script_path)


def _parse_requirements_txt(path: Path) -> list[str]:
    """Parse a requirements.txt file, stripping comments and blank lines."""
    deps: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        deps.append(line)
    return deps


def _parse_pyproject_deps(path: Path) -> list[str]:
    """Parse dependencies from pyproject.toml."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            return []

    text = path.read_text(encoding="utf-8")
    data = tomllib.loads(text)

    # Standard [project.dependencies]
    deps = data.get("project", {}).get("dependencies", [])
    if deps:
        return list(deps)

    # Poetry [tool.poetry.dependencies]
    poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    if poetry_deps:
        return [
            _poetry_dep_to_pip(name, spec)
            for name, spec in poetry_deps.items()
            if name.lower() != "python"
        ]

    return []


def _poetry_dep_to_pip(name: str, spec: str | dict) -> str:
    """Convert a poetry dependency spec to a pip-compatible string."""
    if isinstance(spec, str):
        if spec == "*":
            return name
        # Poetry uses ^ and ~ constraints
        spec = spec.replace("^", ">=").replace("~", "~=")
        return f"{name}{spec}"
    if isinstance(spec, dict):
        version = spec.get("version", "")
        if version and version != "*":
            version = version.replace("^", ">=").replace("~", "~=")
            return f"{name}{version}"
        return name
    return name


def discover_python_version(script_path: Path) -> str:
    """Find the Python version to use in the Modal image."""
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def discover_imports(script_path: Path) -> list[str]:
    """Scan a script's AST to discover third-party imports.

    Resolves local modules in the same directory and recursively
    collects their imports too. Returns pip package names.
    """
    root = discover_project_root(script_path)
    script_dir = script_path.resolve().parent

    visited: set[Path] = set()
    third_party: set[str] = set()

    def _scan_file(path: Path) -> None:
        if path in visited or not path.exists():
            return
        visited.add(path)
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _classify(alias.name, script_dir)
            elif isinstance(node, ast.ImportFrom) and node.module:
                _classify(node.module, script_dir)

    def _classify(module_name: str, local_dir: Path) -> None:
        top = module_name.split(".")[0]

        # Skip stdlib
        if top in _STDLIB_MODULES:
            return
        # Skip profine internals
        if top == "profine":
            return

        # Walk from local_dir up to the project root looking for a
        # local file or package matching `top`. Scripts often live in
        # nested subdirectories while their shared packages sit at the
        # project root; checking only local_dir would mistake those
        # packages for third-party deps.
        cur = local_dir
        while True:
            if (cur / f"{top}.py").exists():
                _scan_file(cur / f"{top}.py")
                return
            pkg_dir = cur / top
            if (pkg_dir / "__init__.py").exists():
                # Walk every .py in the local package — without this we
                # only see __init__.py imports and miss transitive deps
                for py in pkg_dir.rglob("*.py"):
                    _scan_file(py)
                return
            if cur == root or cur.parent == cur:
                break
            cur = cur.parent

        pip_name = _IMPORT_TO_PIP.get(top, top)
        third_party.add(pip_name)

    _scan_file(script_path.resolve())
    return sorted(third_party)


def discover_local_modules(
    script_path: Path,
    *,
    max_files: int = 40,
    max_total_chars: int = 200_000,
) -> dict[str, str]:
    """Collect local Python modules transitively imported by `script_path`.

    Returns a dict mapping each module's path (relative to the project
    root, POSIX form) to its source. The entry script itself is NOT
    included. Limits are advisory caps to keep prompt sizes sane on
    large repos — caller can tune.
    """
    script = script_path.resolve()
    root = discover_project_root(script).resolve()
    script_dir = script.parent

    visited: set[Path] = set()
    files: list[Path] = []
    pkg_visited: set[Path] = set()

    def _scan(path: Path) -> None:
        if path in visited or not path.is_file() or path.suffix != ".py":
            return
        try:
            path.relative_to(root)
        except ValueError:
            return  # Outside project root
        visited.add(path)
        if path != script:
            files.append(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            return
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules.append(node.module)
        for mod in modules:
            top = mod.split(".")[0]
            if top in _STDLIB_MODULES or top == "profine":
                continue
            cur = script_dir
            while True:
                candidate_file = cur / f"{top}.py"
                pkg_dir = cur / top
                if candidate_file.exists() and candidate_file.is_file():
                    _scan(candidate_file)
                    break
                if (pkg_dir / "__init__.py").exists():
                    if pkg_dir not in pkg_visited:
                        pkg_visited.add(pkg_dir)
                        for py in sorted(pkg_dir.rglob("*.py")):
                            _scan(py)
                    break
                if cur == root or cur.parent == cur:
                    break
                cur = cur.parent

    _scan(script)

    out: dict[str, str] = {}
    total = 0
    for path in files:
        if len(out) >= max_files:
            break
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if total + len(text) > max_total_chars:
            continue
        rel = path.relative_to(root).as_posix()
        out[rel] = text
        total += len(text)
    return out


def discover_system_packages(dependencies: list[str]) -> list[str]:
    """Detect system packages needed based on Python dependencies."""
    packages: list[str] = []
    dep_names = {d.split("[")[0].split(">=")[0].split("==")[0].lower() for d in dependencies}
    if "torchcodec" in dep_names:
        packages.append("ffmpeg")
    return packages
