"""Tests for CLI argparse, dispatch, and friendly error mapping."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from profine.cli.errors import format_user_error, is_debug_mode, print_user_error
from profine.cli.main import build_parser, main
from profine.llm.utils import LlmJsonParseError


def test_parser_help_does_not_crash():
    parser = build_parser()
    assert parser.format_help()


def test_parser_unknown_command_errors():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["nope"])


def test_parser_subcommands_present():
    parser = build_parser()
    # Each subcommand should accept its required positional args
    parser.parse_args(["read", "train.py"])
    parser.parse_args(["profile", "train.py"])
    parser.parse_args(["interpret", "--profile-dir", "x"])
    parser.parse_args(["suggest", "--interpret-dir", "x"])
    parser.parse_args(["edit", "train.py", "--suggestion-dir", "x"])
    parser.parse_args(["benchmark", "train.py", "--optimized", "y.py"])


def test_main_with_no_command_returns_1():
    # No subcommand → print help and exit non-zero
    assert main([]) == 1


def test_format_user_error_missing_openai_key():
    exc = RuntimeError("No OpenAI API key. Set OPENAI_API_KEY or pass api_key=.")
    formatted = format_user_error(exc)
    assert formatted is not None
    msg, code = formatted
    assert "OpenAI" in msg
    assert "OPENAI_API_KEY" in msg
    assert code == 2


def test_format_user_error_missing_anthropic_key():
    exc = RuntimeError("No Anthropic API key. Set ANTHROPIC_API_KEY or pass api_key=.")
    formatted = format_user_error(exc)
    assert formatted is not None
    msg, _ = formatted
    assert "Anthropic" in msg
    assert "ANTHROPIC_API_KEY" in msg


def test_format_user_error_modal_auth():
    exc = RuntimeError("modal: not authenticated, run modal token new")
    formatted = format_user_error(exc)
    assert formatted is not None
    msg, _ = formatted
    assert "modal token new" in msg


def test_format_user_error_file_not_found():
    exc = FileNotFoundError(2, "No such file", "/tmp/missing.py")
    formatted = format_user_error(exc)
    assert formatted is not None
    msg, _ = formatted
    assert "/tmp/missing.py" in msg


def test_format_user_error_permission_denied():
    exc = PermissionError(13, "Permission denied", "/root/output")
    formatted = format_user_error(exc)
    assert formatted is not None
    msg, _ = formatted
    assert "Permission" in msg
    assert "--output" in msg


def test_format_user_error_unknown_hardware():
    exc = ValueError("Unknown hardware 'gpu_xyz'. Available: 1x_a100, 1x_h100")
    formatted = format_user_error(exc)
    assert formatted is not None
    msg, _ = formatted
    assert "list-hardware" in msg


def test_format_user_error_llm_json_parse():
    exc = LlmJsonParseError("decode failed", raw="bad json")
    formatted = format_user_error(exc)
    assert formatted is not None
    msg, _ = formatted
    assert "malformed JSON" in msg


def test_format_user_error_unknown_returns_none():
    # Unrecognised exception types must signal "unknown" so the caller re-raises
    assert format_user_error(KeyError("random")) is None
    assert format_user_error(ZeroDivisionError("nope")) is None


def test_print_user_error_returns_minus_one_for_unknown(capsys):
    assert print_user_error(KeyError("?")) == -1


def test_print_user_error_writes_to_stderr_and_returns_code(capsys):
    code = print_user_error(FileNotFoundError(2, "x", "/tmp/x"))
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert code == 2


def test_is_debug_mode_default_off(monkeypatch):
    monkeypatch.delenv("PROFINE_DEBUG", raising=False)
    assert is_debug_mode() is False


def test_is_debug_mode_truthy(monkeypatch):
    monkeypatch.setenv("PROFINE_DEBUG", "1")
    assert is_debug_mode() is True


def test_is_debug_mode_falsy_strings(monkeypatch):
    for v in ("0", "false", "False", ""):
        monkeypatch.setenv("PROFINE_DEBUG", v)
        assert is_debug_mode() is False


def test_main_friendly_error_for_missing_script(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.delenv("PROFINE_DEBUG", raising=False)
    code = main(["read", str(tmp_path / "does_not_exist.py")])
    captured = capsys.readouterr()
    assert "Error: File not found" in captured.err
    assert code == 2
