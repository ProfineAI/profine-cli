"""Modal image construction.

Builds a layered Modal image with all dependencies needed to run
a profiled training script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from profine.schema.hardware import HardwareConfig, ModalRuntimeConfig

# Mount paths inside the Modal container
HF_CACHE_MOUNT = "/root/.cache/huggingface"
PIP_CACHE_MOUNT = "/root/.cache/pip"
WORKSPACE_MOUNT = "/workspace"

# Flash attention build config
FLASH_ATTENTION_VERSION = "2.7.4.post1"
CUDA_DEVEL_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"

# PyTorch index URL for CUDA wheels
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"


class ModalImageBuilder:
    """Builds a Modal image for profiler execution."""

    def __init__(self, config: ModalRuntimeConfig) -> None:
        self.config = config

    def build(
        self,
        modal_module: Any,
        *,
        project_root: Path,
        dependencies: list[str],
        system_packages: list[str] | None = None,
        python_version: str | None = None,
        hardware: HardwareConfig,
        build_flash_attention: bool = False,
        hf_token: str | None = None,
    ) -> Any:
        """Build the complete Modal image.

        Args:
            modal_module: The `modal` package (imported by caller).
            project_root: Local project root to mount.
            dependencies: pip requirements.
            system_packages: apt packages needed.
            python_version: Python version for the base image.
            hardware: GPU hardware config.
            build_flash_attention: Whether to compile flash-attn.
            hf_token: HuggingFace token for gated models.

        Returns:
            A modal.Image instance.
        """
        py_version = python_version or self.config.python_version

        # Base image selection
        if build_flash_attention:
            image = modal_module.Image.from_registry(
                CUDA_DEVEL_IMAGE,
                add_python=py_version,
            )
        else:
            image = modal_module.Image.debian_slim(python_version=py_version)

        # System packages
        sys_pkgs = list(system_packages or [])
        if sys_pkgs:
            image = image.apt_install(*sys_pkgs)

        # Separate torch from other deps (torch needs special index URL)
        torch_deps, other_deps = _split_torch_deps(dependencies)

        # Install torch first with CUDA index
        if torch_deps:
            image = image.pip_install(
                *torch_deps,
                extra_index_url=TORCH_INDEX_URL,
            )

        # profine's own remote-side runtime imports — must match the
        # third-party imports of every module reached from
        # profine/modal/remote.py (executor, discovery, yaml_loader,
        # profiler/hooks). If you add an import to those files, mirror
        # it here or the Modal container will ImportError on startup.
        profiling_deps = ["nvidia-ml-py", "torch", "torchvision", "Pillow", "PyYAML"]
        existing = {d.split(">=")[0].split("==")[0].lower() for d in dependencies}
        profiling_deps = [d for d in profiling_deps if d.lower() not in existing]
        if profiling_deps:
            image = image.pip_install(
                *profiling_deps,
                extra_index_url=TORCH_INDEX_URL,
            )

        # Other dependencies
        if other_deps:
            image = image.pip_install(*other_deps)

        # Flash attention (expensive, separate layer)
        if build_flash_attention:
            image = image.pip_install(
                f"flash-attn=={FLASH_ATTENTION_VERSION}",
                extra_options="--no-build-isolation",
            )

        # HuggingFace cache env + force UTF-8 to avoid charmap
        # encoding errors from Unicode characters in library output
        image = image.env({
            "HF_HOME": HF_CACHE_MOUNT,
            "TRANSFORMERS_CACHE": f"{HF_CACHE_MOUNT}/hub",
            "PYTHONIOENCODING": "utf-8",
        })
        if hf_token:
            image = image.env({"HF_TOKEN": hf_token})

        # Mount project source
        image = image.add_local_dir(
            str(project_root),
            remote_path=WORKSPACE_MOUNT,
        )

        # Mount profine package so hooks are available in the container
        image = image.add_local_python_source("profine")

        return image

    def build_volumes(self, modal_module: Any) -> dict[str, Any]:
        """Create shared cache volumes for HF models and pip wheels."""
        volumes = {}
        volumes[HF_CACHE_MOUNT] = modal_module.Volume.from_name(
            self.config.hf_cache_volume_name,
            create_if_missing=True,
        )
        volumes[PIP_CACHE_MOUNT] = modal_module.Volume.from_name(
            self.config.pip_cache_volume_name,
            create_if_missing=True,
        )
        return volumes

    def build_cls_kwargs(
        self,
        hardware: HardwareConfig,
        volumes: dict[str, Any] | None = None,
        secrets: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Build kwargs for modal.App.cls() decorator."""
        kwargs: dict[str, Any] = {
            "gpu": hardware.modal_gpu,
            "cpu": self.config.cpu,
            "memory": self.config.memory_mb,
            "ephemeral_disk": self.config.ephemeral_disk_mb,
            "timeout": self.config.timeout_seconds,
        }
        if volumes:
            kwargs["volumes"] = volumes
        if secrets:
            kwargs["secrets"] = secrets
        return kwargs


def _split_torch_deps(deps: list[str]) -> tuple[list[str], list[str]]:
    """Separate torch-related deps (need special index) from the rest."""
    torch_prefixes = ("torch", "torchvision", "torchaudio", "torchtext")
    torch_deps: list[str] = []
    other_deps: list[str] = []
    for dep in deps:
        name = dep.split(">=")[0].split("==")[0].split("[")[0].lower()
        if any(name == p or name.startswith(p + "=") for p in torch_prefixes):
            torch_deps.append(dep)
        else:
            other_deps.append(dep)
    return torch_deps, other_deps
