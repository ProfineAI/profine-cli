"""Crash classification — raw error → one of a few bucketed classes.

We never store the raw exception text in telemetry: it routinely
contains paths, identifiers, dataset filenames, and user code
snippets. Instead the recorder calls `classify_crash(error)` and
only the resulting enum-like label leaves the process.

The classes are modeled on what real PyTorch training scripts
(minGPT/nanoGPT-style up through HuggingFace training loops) actually
crash on. Coverage matters more than precision — we'd rather have
the right label for 80% of failures and "other" for the rest than
an exhaustive taxonomy with empty buckets.

Patterns are a single ordered table at module scope. Adding a new
class = adding one row. The classifier is pattern-matching only, so
extending it is a copy-paste-and-add-test exercise rather than a
refactor.
"""

from __future__ import annotations

import re


# Public set of crash classes. classify_crash() returns one of these
# (or None for empty input). Anything outside this set is a bug.
CRASH_CLASSES: frozenset[str] = frozenset({
    # ----- memory / numeric -----
    "oom",             # CUDA OOM, host OOM, allocator failures
    "nan_loss",        # NaN/Inf in loss or gradients
    # ----- compilation / kernels -----
    "compile_fail",    # torch.compile / dynamo / inductor failures
    "kernel_error",    # CUDA kernel launch, cuBLAS, cuDNN, illegal memory access
    # ----- shapes / script bugs -----
    "shape_mismatch",  # tensor size/shape errors (model wiring vs data)
    "script_bug",      # AttributeError / TypeError / AssertionError / Index/Key in user code
    # ----- distributed -----
    "dist_init",       # torch.distributed init, NCCL setup, rendezvous
    "dist_collective", # collective op failures mid-run (NCCL timeouts, etc.)
    # ----- data / dependencies / IO -----
    "dataloader",      # dataset access, DataLoader worker crashes
    "dep_missing",     # ImportError, ModuleNotFoundError
    "auth",            # gated model access, missing HF token
    "network",         # download failures, HTTP errors, connection issues
    "disk",            # disk full, write failures, checkpoint IO
    # ----- environment -----
    "timeout",         # wall-clock, container kill by time
    "process_killed",  # SIGKILL, OOM killer, host eviction
    # ----- fallback -----
    "other",
})


# Patterns ordered most-specific-first. Each pattern is `re.search`'d
# against `str(error)`. The first match wins. To extend: add a row,
# add a test case.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # ----- memory: must come before generic kernel match because
    # OOM often surfaces as a CUDA error.
    ("oom",             re.compile(r"out of memory|outofmemory|cuda.*alloc.*fail|memory.{0,30}exhausted", re.IGNORECASE)),
    ("nan_loss",        re.compile(r"loss is nan|nan/inf in (?:loss|grad)|grad.*nan|inf gradient|loss.*became (?:nan|inf)", re.IGNORECASE)),

    # ----- compilation: explicit torch.compile keywords first so
    # they don't get caught by the generic kernel_error rule.
    ("compile_fail",    re.compile(r"torchdynamo|torch\.compile|inductor.*error|backendcompilerfailed|recompil[ae]|graph break.*error", re.IGNORECASE)),

    # ----- distributed: init/setup vs collective ops mid-run.
    ("dist_init",       re.compile(r"(?:nccl|gloo).{0,40}init|rendezvous|c10d.*init|init_process_group|distributed.*not.*initialized", re.IGNORECASE)),
    ("dist_collective", re.compile(r"nccl.*timeout|nccl.*unhandled|all[_-]?reduce.*fail|broadcast.*fail|collective.*timeout", re.IGNORECASE)),

    # ----- kernels: post-compile, post-dist so we don't swallow them.
    ("kernel_error",    re.compile(r"cuda.*illegal memory|cuda kernel|launch failed|cublas.*error|cudnn.*error|device-side assert", re.IGNORECASE)),

    # ----- shape mismatches (very common bug in custom training loops).
    ("shape_mismatch",  re.compile(r"size mismatch|shape.*mismatch|expected.*got.*size|dimension.*out of range|the size of tensor a.*must match", re.IGNORECASE)),

    # ----- dependencies before script_bug — ImportError is technically
    # a script-time error but the bucketed cause is "missing dep".
    ("dep_missing",     re.compile(r"modulenotfounderror|no module named|importerror|cannot import name", re.IGNORECASE)),

    # ----- auth / network / disk before generic FileNotFound.
    ("auth",            re.compile(r"401\b|403\b|unauthor[iz]|gated repo|huggingface.*token|access denied|forbidden", re.IGNORECASE)),
    ("network",         re.compile(r"connection.*(?:refused|reset|abort)|max retries exceeded|httperror|sslerror|name resolution|temporary failure", re.IGNORECASE)),
    ("disk",            re.compile(r"no space left|disk.*full|errno 28|cannot write.*(?:checkpoint|file)", re.IGNORECASE)),

    # ----- container-level: timeout vs process kill.
    ("timeout",         re.compile(r"timed? ?out|deadlineexceeded|wall.clock|container.*killed.*time|exceeded.*time limit", re.IGNORECASE)),
    ("process_killed",  re.compile(r"sigkill|killed.*signal|host evicted|oom-?killer|preempted", re.IGNORECASE)),

    # ----- dataloader: cover both dataset-access and worker-process modes.
    ("dataloader",      re.compile(r"dataloader|dataset|worker exited|num_workers|getitem", re.IGNORECASE)),

    # ----- generic script bugs come last so more-specific rules win.
    # Matches the common Python exception names that crash a training
    # script when the user code or data shape is off.
    ("script_bug",      re.compile(r"attributeerror|typeerror|assertionerror|indexerror|keyerror|valueerror|notimplementederror", re.IGNORECASE)),
)


def classify_crash(error: str | BaseException | None) -> str | None:
    """Return one of CRASH_CLASSES, or None for empty/None input.

    Accepts either a string or an exception instance. For exceptions
    we use `str(exc)` rather than the full traceback so customer code
    paths never leak into telemetry.
    """
    if error is None:
        return None
    text = str(error).strip()
    if not text:
        return None
    for label, pattern in _PATTERNS:
        if pattern.search(text):
            return label
    return "other"
