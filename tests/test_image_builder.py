"""Tests for Modal image-builder helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from profine.modal.image_builder import ModalImageBuilder, _split_torch_deps
from profine.schema.hardware import HardwareConfig, ModalRuntimeConfig, get_hardware


def test_split_torch_deps_isolates_torch_family():
    deps = ["torch>=2.0", "torchvision", "transformers", "torchaudio==2.1", "numpy"]
    torch_deps, other = _split_torch_deps(deps)
    assert "torch>=2.0" in torch_deps
    assert "torchvision" in torch_deps
    assert "torchaudio==2.1" in torch_deps
    assert "transformers" in other
    assert "numpy" in other


def test_split_torch_deps_does_not_match_torch_substring():
    # `torchao` would be unfortunately captured by a naive `startswith("torch")`,
    # but only if the prefix list includes it. Verify the current split avoids it
    # (it's not in torch_prefixes, so it should land in `other`).
    torch_deps, other = _split_torch_deps(["torchao"])
    # current implementation uses `name.startswith(p + "=")` which won't match "torchao"
    assert "torchao" not in torch_deps
    assert "torchao" in other


def test_split_torch_deps_empty():
    assert _split_torch_deps([]) == ([], [])


def test_build_cls_kwargs_contains_required_fields():
    config = ModalRuntimeConfig(timeout_seconds=600, cpu=4.0, memory_mb=16000)
    builder = ModalImageBuilder(config)
    hardware = get_hardware("1x_a100")
    kwargs = builder.build_cls_kwargs(hardware)
    assert kwargs["gpu"] == "A100-80GB"
    assert kwargs["cpu"] == 4.0
    assert kwargs["memory"] == 16000
    assert kwargs["timeout"] == 600


def test_build_cls_kwargs_includes_volumes_when_passed():
    config = ModalRuntimeConfig()
    builder = ModalImageBuilder(config)
    hardware = get_hardware("1x_t4")
    fake_volumes = {"/cache": MagicMock()}
    kwargs = builder.build_cls_kwargs(hardware, volumes=fake_volumes)
    assert kwargs["volumes"] is fake_volumes


def test_build_cls_kwargs_includes_secrets_when_passed():
    config = ModalRuntimeConfig()
    builder = ModalImageBuilder(config)
    hardware = get_hardware("1x_t4")
    secrets = [MagicMock(), MagicMock()]
    kwargs = builder.build_cls_kwargs(hardware, secrets=secrets)
    assert kwargs["secrets"] == secrets


def test_build_cls_kwargs_omits_volumes_when_not_passed():
    config = ModalRuntimeConfig()
    builder = ModalImageBuilder(config)
    hardware = get_hardware("1x_t4")
    kwargs = builder.build_cls_kwargs(hardware)
    assert "volumes" not in kwargs
    assert "secrets" not in kwargs
