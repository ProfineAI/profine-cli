"""Deterministic AST-based fact extractor.

Walks a single Python source file and extracts raw observations with
file:line citations. The output is a CodeFacts dataclass that the LLM
analyzer consumes — it does NOT interpret or guess, only reports what it
can prove from the syntax tree.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Citation:
    """A located snippet of code."""
    file: str
    line: int
    snippet: str
    kind: str = "code"  # code | comment | import | config


@dataclass(slots=True)
class ImportFact:
    module: str
    names: list[str]
    line: int
    style: str  # "import" | "from"


@dataclass(slots=True)
class CallFact:
    """A function/method call observed in the source."""
    name: str
    line: int
    keywords: dict[str, str]  # kwarg_name -> repr of value
    positional_reprs: list[str]
    assigned_to: str | None = None


@dataclass(slots=True)
class AssignmentFact:
    """A variable assignment observed in the source."""
    target: str
    value_repr: str
    line: int
    value_type: str = ""  # "call" | "constant" | "expression"


@dataclass(slots=True)
class ClassFact:
    """A class definition."""
    name: str
    bases: list[str]
    line: int
    method_names: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FunctionFact:
    """A top-level or method function definition."""
    name: str
    line: int
    decorators: list[str] = field(default_factory=list)
    is_method: bool = False


@dataclass(slots=True)
class CodeFacts:
    """Everything the extractor can prove from the AST.

    This is the raw material the LLM receives alongside the source.
    """
    file_path: str = ""
    total_lines: int = 0

    imports: list[ImportFact] = field(default_factory=list)
    calls: list[CallFact] = field(default_factory=list)
    assignments: list[AssignmentFact] = field(default_factory=list)
    classes: list[ClassFact] = field(default_factory=list)
    functions: list[FunctionFact] = field(default_factory=list)

    # Pre-classified observations (deterministic pattern matches)
    model_loader_calls: list[CallFact] = field(default_factory=list)
    optimizer_calls: list[CallFact] = field(default_factory=list)
    dataloader_calls: list[CallFact] = field(default_factory=list)
    scheduler_calls: list[CallFact] = field(default_factory=list)
    loss_calls: list[CallFact] = field(default_factory=list)
    distributed_calls: list[CallFact] = field(default_factory=list)
    compile_calls: list[CallFact] = field(default_factory=list)
    autocast_calls: list[CallFact] = field(default_factory=list)
    grad_scaler_calls: list[CallFact] = field(default_factory=list)
    checkpoint_calls: list[CallFact] = field(default_factory=list)


# Pattern sets for classification

from profine.config.yaml_loader import (
    get_dataloader_patterns as _get_dataloader_patterns,
    get_distributed_patterns as _get_distributed_patterns,
    get_loss_names as _get_loss_names,
    get_model_loaders as _get_model_loaders,
    get_optimizer_names as _get_optimizer_names,
    get_scheduler_names as _get_scheduler_names,
)

# Loaded from config/patterns.yaml
_MODEL_LOADER_PATTERNS = _get_model_loaders()
_OPTIMIZER_NAMES = _get_optimizer_names()
_SCHEDULER_NAMES = _get_scheduler_names()
_LOSS_NAMES = _get_loss_names()
_DISTRIBUTED_PATTERNS = _get_distributed_patterns()
_DATALOADER_PATTERNS = _get_dataloader_patterns()


def _resolve_call_name(node: ast.expr) -> str:
    """Resolve a call target to a dotted name string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _resolve_call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):
        return _resolve_call_name(node.value)
    return ""


def _short_repr(node: ast.expr, max_len: int = 80) -> str:
    """Best-effort short repr of an AST expression."""
    try:
        text = ast.unparse(node)
    except Exception:
        text = "<?>"
    if len(text) > max_len:
        text = text[:max_len - 3] + "..."
    return text


def _extract_keywords(call: ast.Call) -> dict[str, str]:
    kws: dict[str, str] = {}
    for kw in call.keywords:
        if kw.arg:
            kws[kw.arg] = _short_repr(kw.value)
    return kws


def _extract_positionals(call: ast.Call) -> list[str]:
    return [_short_repr(a) for a in call.args]


def _leaf_name(dotted: str) -> str:
    """'torch.optim.AdamW' -> 'AdamW'"""
    return dotted.rsplit(".", 1)[-1] if dotted else dotted


def _matches_any(name: str, patterns: set[str]) -> bool:
    leaf = _leaf_name(name)
    if leaf in patterns:
        return True
    return any(p in name for p in patterns)


class _FactVisitor(ast.NodeVisitor):
    """Single-pass AST visitor that populates CodeFacts."""

    def __init__(self, file_path: str, source_lines: list[str]) -> None:
        self.facts = CodeFacts(file_path=file_path, total_lines=len(source_lines))
        self._source_lines = source_lines
        self._file = file_path

    # Imports

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.facts.imports.append(ImportFact(
                module=alias.name,
                names=[alias.asname or alias.name],
                line=node.lineno,
                style="import",
            ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        names = [a.name for a in (node.names or [])]
        self.facts.imports.append(ImportFact(
            module=module,
            names=names,
            line=node.lineno,
            style="from",
        ))
        self.generic_visit(node)

    # Classes

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = [_resolve_call_name(b) if isinstance(b, (ast.Name, ast.Attribute)) else _short_repr(b)
                 for b in node.bases]
        methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self.facts.classes.append(ClassFact(
            name=node.name,
            bases=bases,
            line=node.lineno,
            method_names=methods,
        ))
        # visit children to catch calls inside class bodies
        self.generic_visit(node)

    # Functions

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_function(node)
        self.generic_visit(node)

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        decorators = [_short_repr(d) for d in node.decorator_list]
        # is_method if parent is a ClassDef — approximate via first arg 'self'/'cls'
        args = node.args
        is_method = bool(args.args and args.args[0].arg in ("self", "cls"))
        self.facts.functions.append(FunctionFact(
            name=node.name,
            line=node.lineno,
            decorators=decorators,
            is_method=is_method,
        ))

    # Assignments

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            target_name = _resolve_call_name(target) if isinstance(target, (ast.Name, ast.Attribute)) else _short_repr(target)
            value_type = "call" if isinstance(node.value, ast.Call) else (
                "constant" if isinstance(node.value, ast.Constant) else "expression"
            )
            self.facts.assignments.append(AssignmentFact(
                target=target_name,
                value_repr=_short_repr(node.value),
                line=node.lineno,
                value_type=value_type,
            ))
        self._maybe_record_call_from_assign(node)
        self.generic_visit(node)

    def _maybe_record_call_from_assign(self, node: ast.Assign) -> None:
        if not isinstance(node.value, ast.Call):
            return
        assigned_to = None
        for t in node.targets:
            if isinstance(t, ast.Name):
                assigned_to = t.id
                break
        self._record_call(node.value, node.lineno, assigned_to)

    # Bare calls

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Call):
            self._record_call(node.value, node.lineno, None)
        self.generic_visit(node)

    # Call recording and classification

    def _record_call(self, call: ast.Call, line: int, assigned_to: str | None) -> None:
        name = _resolve_call_name(call.func)
        if not name:
            return
        cf = CallFact(
            name=name,
            line=line,
            keywords=_extract_keywords(call),
            positional_reprs=_extract_positionals(call),
            assigned_to=assigned_to,
        )
        self.facts.calls.append(cf)
        self._classify_call(cf)

    def _classify_call(self, cf: CallFact) -> None:
        name = cf.name
        leaf = _leaf_name(name)

        # Model loaders
        if leaf == "from_pretrained" or leaf.startswith("AutoModel"):
            self.facts.model_loader_calls.append(cf)

        # Optimizers
        if leaf in _OPTIMIZER_NAMES:
            self.facts.optimizer_calls.append(cf)

        # Schedulers
        if leaf in _SCHEDULER_NAMES or _matches_any(name, _SCHEDULER_NAMES):
            self.facts.scheduler_calls.append(cf)

        # Loss
        if leaf in _LOSS_NAMES or _matches_any(name, _LOSS_NAMES):
            self.facts.loss_calls.append(cf)

        # DataLoader
        if leaf in _DATALOADER_PATTERNS or _matches_any(name, _DATALOADER_PATTERNS):
            self.facts.dataloader_calls.append(cf)

        # Distributed
        if _matches_any(name, _DISTRIBUTED_PATTERNS):
            self.facts.distributed_calls.append(cf)

        # torch.compile
        if name == "torch.compile" or leaf == "compile" and "torch" in name:
            self.facts.compile_calls.append(cf)

        # Autocast
        if leaf == "autocast" or "autocast" in name.lower():
            self.facts.autocast_calls.append(cf)

        # GradScaler
        if leaf == "GradScaler" or "GradScaler" in name:
            self.facts.grad_scaler_calls.append(cf)

        # Gradient checkpointing
        if "checkpoint" in name.lower() and ("gradient" in name.lower() or "activation" in name.lower()):
            self.facts.checkpoint_calls.append(cf)
        if leaf == "gradient_checkpointing_enable":
            self.facts.checkpoint_calls.append(cf)

        # Method-style optimizer (e.g. model.configure_optimizers())
        if "configure_optimizers" in name or "configure_optimizer" in name:
            self.facts.optimizer_calls.append(cf)

        # F.cross_entropy / F.mse_loss style
        if leaf in _LOSS_NAMES:
            if cf not in self.facts.loss_calls:
                self.facts.loss_calls.append(cf)

    # With statements (context managers)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if isinstance(item.context_expr, ast.Call):
                call = item.context_expr
                name = _resolve_call_name(call.func)
                if name and "autocast" in name.lower():
                    cf = CallFact(
                        name=name,
                        line=node.lineno,
                        keywords=_extract_keywords(call),
                        positional_reprs=_extract_positionals(call),
                    )
                    self.facts.calls.append(cf)
                    self.facts.autocast_calls.append(cf)
                elif name and "GradScaler" in name:
                    cf = CallFact(
                        name=name,
                        line=node.lineno,
                        keywords=_extract_keywords(call),
                        positional_reprs=_extract_positionals(call),
                    )
                    self.facts.calls.append(cf)
                    self.facts.grad_scaler_calls.append(cf)
                elif name and "no_grad" in name:
                    pass  # not interesting for architecture
                else:
                    cf = CallFact(
                        name=name or "",
                        line=node.lineno,
                        keywords=_extract_keywords(call),
                        positional_reprs=_extract_positionals(call),
                    )
                    self.facts.calls.append(cf)
        self.generic_visit(node)


def extract(source: str, file_path: str = "<script>") -> CodeFacts:
    """Extract all deterministic facts from a Python source string."""
    tree = ast.parse(source)
    lines = source.splitlines()
    visitor = _FactVisitor(file_path, lines)
    visitor.visit(tree)
    return visitor.facts
