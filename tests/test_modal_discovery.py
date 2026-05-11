"""Tests for project root, dep discovery, local-module discovery."""

from __future__ import annotations

from pathlib import Path

from profine.modal.discovery import (
    discover_dependencies,
    discover_local_modules,
    discover_project_root,
    discover_system_packages,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_project_root_via_pyproject(tmp_path):
    _write(tmp_path / "pyproject.toml", "[project]\nname = 'foo'\n")
    _write(tmp_path / "src" / "train.py", "print('hi')")
    assert discover_project_root(tmp_path / "src" / "train.py").resolve() == tmp_path.resolve()


def test_project_root_via_git(tmp_path):
    (tmp_path / ".git").mkdir()
    _write(tmp_path / "train.py", "x = 1")
    assert discover_project_root(tmp_path / "train.py").resolve() == tmp_path.resolve()


def test_project_root_no_marker_falls_back_to_script_dir(tmp_path):
    nested = tmp_path / "nowhere"
    _write(nested / "train.py", "x = 1")
    assert discover_project_root(nested / "train.py").resolve() == nested.resolve()


def test_dependencies_from_requirements_txt(tmp_path):
    _write(tmp_path / "requirements.txt", "torch>=2.0\n# comment\ntransformers\n\n")
    _write(tmp_path / "train.py", "x = 1")
    deps = discover_dependencies(tmp_path / "train.py")
    assert "torch>=2.0" in deps
    assert "transformers" in deps


def test_dependencies_from_pyproject_standard(tmp_path):
    _write(tmp_path / "pyproject.toml", "[project]\nname = 'foo'\ndependencies = ['numpy', 'pandas>=2']\n")
    _write(tmp_path / "foo" / "__init__.py", "")
    _write(tmp_path / "foo" / "train.py", "x = 1")
    deps = discover_dependencies(tmp_path / "foo" / "train.py")
    assert "numpy" in deps
    assert "pandas>=2" in deps


def test_dependencies_falls_back_to_imports(tmp_path):
    # No requirements.txt or pyproject — should AST-scan the script
    _write(tmp_path / ".git" / "HEAD", "x")  # mark as project root
    _write(tmp_path / "train.py", "import torch\nimport numpy\n")
    deps = discover_dependencies(tmp_path / "train.py")
    assert any("torch" in d for d in deps)


def test_discover_local_modules_picks_up_sibling_imports(tmp_path):
    _write(tmp_path / "pyproject.toml", "[project]\nname = 'demo'\n")
    _write(tmp_path / "demo" / "__init__.py", "")
    _write(tmp_path / "demo" / "model.py", "class Net: pass\n")
    _write(tmp_path / "demo" / "train.py", "from demo.model import Net\n")
    locals_ = discover_local_modules(tmp_path / "demo" / "train.py")
    # entry script itself excluded
    assert "demo/train.py" not in locals_
    assert "demo/model.py" in locals_


def test_discover_local_modules_skips_stdlib_and_third_party(tmp_path):
    _write(tmp_path / "pyproject.toml", "[project]\nname = 'demo'\n")
    _write(tmp_path / "demo" / "__init__.py", "")
    _write(tmp_path / "demo" / "train.py", "import os\nimport torch\nimport json\n")
    locals_ = discover_local_modules(tmp_path / "demo" / "train.py")
    # No stdlib / no third-party paths leak in
    assert all("torch" not in p for p in locals_)


def test_discover_local_modules_respects_max_files(tmp_path):
    _write(tmp_path / "pyproject.toml", "[project]\nname = 'd'\n")
    _write(tmp_path / "d" / "__init__.py", "")
    # Create 50 sibling modules; cap at 5
    imports = []
    for i in range(50):
        _write(tmp_path / "d" / f"mod{i}.py", "x = 1")
        imports.append(f"from d import mod{i}")
    _write(tmp_path / "d" / "train.py", "\n".join(imports))
    locals_ = discover_local_modules(tmp_path / "d" / "train.py", max_files=5)
    assert len(locals_) <= 5


def test_discover_system_packages_torchcodec_needs_ffmpeg():
    pkgs = discover_system_packages(["torchcodec>=0.1", "torch"])
    assert "ffmpeg" in pkgs


def test_discover_system_packages_default_empty():
    assert discover_system_packages(["torch", "transformers"]) == []
