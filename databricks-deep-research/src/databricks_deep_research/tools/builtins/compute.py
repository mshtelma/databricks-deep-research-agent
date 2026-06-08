"""Python compute tool — sandboxed code execution for deterministic calculations.

Provides a ``PythonComputeTool`` that executes LLM-generated Python code in a
restricted sandbox.  A wide range of stdlib modules are available by default,
and third-party modules (e.g. numpy) can be added per-instance via
``extra_modules``.  Variables persist across calls within a single tool instance
(i.e., within one workflow run), enabling multi-step computation.

Security:
    - ``__import__``, ``exec``, ``eval``, ``compile``, ``open`` are blocked
    - ``getattr``/``setattr``/``delattr`` are blocked (prevents attribute escape)
    - AST validation blocks dunder attribute access (prevents ``module.__builtins__``
      and class-hierarchy escapes)
    - Per-instance import guard prevents module leakage between tool instances
    - Timeout via ``asyncio.wait_for`` + thread pool executor
    - Output truncated to ``max_output_chars``
"""

from __future__ import annotations

import ast
import asyncio
import collections as _collections_mod
import contextlib
import datetime as _datetime_mod
import decimal as _decimal_mod
import fractions as _fractions_mod
import functools as _functools_mod
import io
import itertools as _itertools_mod
import json as _json_mod
import logging
import math
import operator as _operator_mod
import re as _re_mod
import statistics
import string as _string_mod
import textwrap as _textwrap_mod
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from databricks_deep_research.tools.protocol import (
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sandbox configuration
# ---------------------------------------------------------------------------

_SAFE_BUILTINS_BASE: dict[str, Any] = {
    # Constructors / types
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "complex": complex,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    "bytes": bytes,
    "bytearray": bytearray,
    "range": range,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    # Math
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "divmod": divmod,
    "len": len,
    # Iteration
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "reversed": reversed,
    "sorted": sorted,
    "iter": iter,
    "next": next,
    "all": all,
    "any": any,
    # Formatting / inspection
    "print": print,
    "repr": repr,
    "format": format,
    "chr": chr,
    "ord": ord,
    "hash": hash,
    "id": id,
    # Number formatting
    "hex": hex,
    "bin": bin,
    "oct": oct,
    # Inspection (safe — returns bool only)
    "ascii": ascii,
    "callable": callable,
    "hasattr": hasattr,
    # Object / type
    "object": object,
    "slice": slice,
    "memoryview": memoryview,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "super": super,
    # Exception types (precise try/except)
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "ZeroDivisionError": ZeroDivisionError,
    "StopIteration": StopIteration,
    "ArithmeticError": ArithmeticError,
    "OverflowError": OverflowError,
    "RuntimeError": RuntimeError,
    "AttributeError": AttributeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "LookupError": LookupError,
    "NotImplementedError": NotImplementedError,
    "FileNotFoundError": FileNotFoundError,
    # None / True / False are always available
    "None": None,
    "True": True,
    "False": False,
}

# Allow `import math` / `from statistics import mean` in the sandbox.
# The modules are already pre-injected into the namespace, but LLMs
# sometimes generate `import` statements despite prompt instructions.
# Without this, Python raises "ImportError: __import__ not found".
_ALLOWED_IMPORT_MODULES: dict[str, Any] = {
    "math": math,
    "statistics": statistics,
    "decimal": _decimal_mod,
    "re": _re_mod,
    "fractions": _fractions_mod,
    "itertools": _itertools_mod,
    "functools": _functools_mod,
    "collections": _collections_mod,
    "copy": __import__("copy"),
    "calendar": __import__("calendar"),
    "datetime": _datetime_mod,
    "json": _json_mod,
    "operator": _operator_mod,
    "string": _string_mod,
    "textwrap": _textwrap_mod,
}


def _restricted_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Allow importing only whitelisted modules in the compute sandbox.

    Retained at module level for backward compatibility.  New instances use
    per-instance closures instead (see ``PythonComputeTool.__init__``).

    Supports submodule imports (e.g. ``from numpy.polynomial import polynomial``)
    by traversing attributes of whitelisted root modules.
    """
    del globals, locals, level
    root = name.split(".")[0]
    if root not in _ALLOWED_IMPORT_MODULES:
        available = ", ".join(sorted(_ALLOWED_IMPORT_MODULES.keys()))
        raise ImportError(
            f"Module '{name}' is not available. Available: {available}"
        )
    mod = _ALLOWED_IMPORT_MODULES[root]
    for part in name.split(".")[1:]:
        try:
            mod = getattr(mod, part)
        except AttributeError:
            raise ImportError(f"Module '{name}' has no submodule '{part}'") from None
    # CPython protocol: with fromlist → return leaf; without → return root
    if fromlist:
        return mod
    return _ALLOWED_IMPORT_MODULES[root]


# ---------------------------------------------------------------------------
# AST security validation
# ---------------------------------------------------------------------------

# Dunder attributes that enable sandbox escapes.  Blocks:
#   - module.__builtins__['__import__']('os')    → import escape
#   - ().__class__.__bases__[0].__subclasses__()  → class hierarchy escape
#   - func.__globals__                            → code introspection escape
_BLOCKED_DUNDER_ATTRS: frozenset[str] = frozenset({
    # Module / import escape
    "__builtins__", "__import__", "__loader__", "__spec__",
    # Class hierarchy escape
    "__subclasses__", "__bases__", "__mro__", "__class__",
    # Attribute / code introspection escape
    "__globals__", "__code__", "__func__", "__self__",
    "__dict__",
    # Object lifecycle (bound methods expose __globals__)
    "__init__", "__new__", "__del__",
    "__reduce__", "__reduce_ex__",
    # Descriptor protocol escape
    "__getattr__", "__setattr__", "__delattr__",
    "__set_name__", "__init_subclass__",
})


def _validate_ast(tree: ast.Module) -> None:
    """Reject code that accesses dangerous dunder attributes.

    Walks the entire AST and raises ``ValueError`` for attribute access
    (``obj.__builtins__``) or bare name references (``__import__``) that
    match the blocklist.  Legitimate dunders like ``__len__``, ``__add__``,
    ``__contains__``, ``__str__``, ``__repr__`` are NOT blocked.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _BLOCKED_DUNDER_ATTRS:
            raise ValueError(
                f"Access to '.{node.attr}' is not allowed in the compute sandbox"
            )
        if isinstance(node, ast.Name) and node.id in _BLOCKED_DUNDER_ATTRS:
            raise ValueError(
                f"Reference to '{node.id}' is not allowed in the compute sandbox"
            )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_CODE_LENGTH = 20_000

_MAX_NAMESPACE_ENTRIES = 200

_NS_MAX_VALUE_REPR = 200
_NS_MAX_TOTAL_CHARS = 2000

_DEFAULT_DESCRIPTION = (
    "Execute Python code for calculations. "
    "Available modules: math, statistics, decimal, datetime, json, re, "
    "fractions, itertools, functools, collections, operator, copy, "
    "calendar, string, textwrap. "
    "Variables persist across calls within the same session. "
    "Use print() or end with an expression to see results."
)

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="compute")


class _SandboxSyntaxError(Exception):
    """Raised inside the sandbox to signal a SyntaxError back to execute()."""


class _SandboxSecurityError(Exception):
    """Raised inside the sandbox for blocked dunder attribute access."""


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


class PythonComputeTool:
    """Sandboxed Python code execution tool.

    Implements ``ResearchTool`` protocol.  Intended for deterministic numerical
    computation within agent workflows — the LLM writes Python code and receives
    the printed output and/or the value of the last expression.

    Parameters
    ----------
    allowed_modules
        If provided, **replaces** the default module whitelist entirely.
        Only the listed modules (plus any in *extra_modules*) will be available.
    extra_modules
        Extends the module set (default or replaced) with additional modules.
        Non-installed modules are skipped with a warning.
    """

    def __init__(
        self,
        *,
        name: str = "compute",
        allowed_modules: list[str] | None = None,
        extra_modules: list[str] | None = None,
        max_execution_seconds: float = 10.0,
        max_output_chars: int = 10_000,
        max_code_length: int = _MAX_CODE_LENGTH,
        description: str = "",
    ) -> None:
        self._name = name
        self._max_execution_seconds = max_execution_seconds
        self._max_output_chars = max_output_chars
        self._max_code_length = max_code_length
        self._description = description or _DEFAULT_DESCRIPTION

        # ---- Per-instance module configuration ----
        if allowed_modules is not None:
            # Complete replacement: only listed modules are available.
            base: dict[str, Any] = {}
            for mod_name in allowed_modules:
                if mod_name in _ALLOWED_IMPORT_MODULES:
                    base[mod_name] = _ALLOWED_IMPORT_MODULES[mod_name]
                else:
                    try:
                        base[mod_name] = __import__(mod_name)
                    except ImportError:
                        logger.warning(
                            "COMPUTE_SKIP_MODULE module=%s reason=not_installed",
                            mod_name,
                        )
        else:
            base = dict(_ALLOWED_IMPORT_MODULES)

        # Extend with extra modules (third-party or additional stdlib).
        for mod_name in extra_modules or []:
            if mod_name not in base:
                try:
                    base[mod_name] = __import__(mod_name)
                    logger.info(
                        "COMPUTE_EXTRA_MODULE module=%s status=loaded", mod_name
                    )
                except ImportError:
                    logger.warning(
                        "COMPUTE_SKIP_MODULE module=%s reason=not_installed",
                        mod_name,
                    )

        self._allowed_modules: dict[str, Any] = base
        self._modules: dict[str, Any] = dict(base)

        # ---- Per-instance restricted import (closure) ----
        allowed_ref = self._allowed_modules

        def _instance_import(
            name: str,
            globals: dict[str, Any] | None = None,
            locals: dict[str, Any] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            del globals, locals, level
            root = name.split(".")[0]
            if root not in allowed_ref:
                available = ", ".join(sorted(allowed_ref.keys()))
                raise ImportError(
                    f"Module '{name}' is not available. Available: {available}"
                )
            mod = allowed_ref[root]
            for part in name.split(".")[1:]:
                try:
                    mod = getattr(mod, part)
                except AttributeError:
                    raise ImportError(
                        f"Module '{name}' has no submodule '{part}'"
                    ) from None
            # CPython protocol: with fromlist → return leaf; without → return root
            if fromlist:
                return mod
            return allowed_ref[root]

        # ---- Per-instance safe builtins ----
        self._safe_builtins: dict[str, Any] = dict(_SAFE_BUILTINS_BASE)
        self._safe_builtins["__import__"] = _instance_import

        # ---- Thread safety ----
        self._lock = threading.Lock()

        # ---- Persistent namespace for cross-call variable sharing ----
        self._namespace: dict[str, Any] = {}

        # ---- Per-execution namespace refresh hooks ----
        self._before_execute_hooks: dict[str, Callable[[PythonComputeTool], None]] = {}

    # -- ResearchTool protocol -----------------------------------------------

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code to execute. Many stdlib modules are "
                            "pre-imported (math, statistics, datetime, json, etc.). "
                            "Variables persist across calls. "
                            "Use print() or end with an expression to see results."
                        ),
                        "maxLength": self._max_code_length,
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            source_type="builtin",
            source_kind=SourceKind.builtin,
        )

    # -- Namespace inspection (framework-facing, NOT exposed to sandbox) -----

    _SAFE_NAMESPACE_TYPES = (int, float, str, bool, list, dict, tuple, type(None))

    def list_user_namespace(
        self,
        *,
        prefix: str | None = None,
        names: list[str] | None = None,
        max_items: int = 50,
        include_values: bool = True,
        max_value_repr: int = _NS_MAX_VALUE_REPR,
    ) -> list[dict[str, Any]]:
        """Return filtered, structured snapshot of user-defined variables.

        Thread-safe.  Does not execute code or modify namespace.
        """
        with self._lock:
            entries: list[dict[str, Any]] = []
            for k, v in self._namespace.items():
                if k.startswith("_"):
                    continue
                if not isinstance(v, self._SAFE_NAMESPACE_TYPES):
                    continue
                if names is not None and k not in names:
                    continue
                if prefix is not None and not k.startswith(prefix):
                    continue
                entry: dict[str, Any] = {"name": k, "type": type(v).__name__}
                if include_values:
                    val_repr = repr(v)
                    if len(val_repr) > max_value_repr:
                        val_repr = val_repr[:max_value_repr] + "..."
                    entry["value"] = val_repr
                entries.append(entry)
                if len(entries) >= max_items:
                    break
            return entries

    def namespace_snapshot(self) -> str:
        """Return prompt-safe summary of user-defined variables.

        Used by the agent harness to inject namespace state into downstream
        agent prompts, eliminating namespace-discovery tool calls.
        Delegates to ``list_user_namespace`` for filtering/repr logic.
        """
        entries = self.list_user_namespace(
            max_items=50, max_value_repr=_NS_MAX_VALUE_REPR,
        )
        if not entries:
            return "(empty — no variables stored)"

        lines = [f"  {e['name']} = {e.get('value', '?')}" for e in entries]
        result = "\n".join(lines)
        if len(result) > _NS_MAX_TOTAL_CHARS:
            truncated: list[str] = []
            chars = 0
            for line in lines:
                if chars + len(line) + 1 > _NS_MAX_TOTAL_CHARS:
                    remaining_count = len(lines) - len(truncated)
                    truncated.append(f"  ... ({remaining_count} more variables)")
                    break
                truncated.append(line)
                chars += len(line) + 1
            result = "\n".join(truncated)
        return result

    def inject_variable(self, name: str, value: Any) -> None:
        """Inject a variable into the compute namespace from an external tool.

        Used by tools like ``TableLoadTool`` to make structured data
        directly available for agent ``compute()`` calls without requiring
        the LLM to paste large strings into code.

        Thread-safe — acquires the namespace lock.
        """
        with self._lock:
            self._namespace[name] = value
            # Evict oldest if over limit (same logic as post-execute)
            while len(self._namespace) > _MAX_NAMESPACE_ENTRIES:
                oldest = next(iter(self._namespace))
                del self._namespace[oldest]

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Return one value from the persistent namespace without executing code."""
        with self._lock:
            return self._namespace.get(name, default)

    def set_before_execute_hook(
        self, name: str, hook: Callable[[PythonComputeTool], None]
    ) -> None:
        """Register or replace a hook that refreshes namespace entries.

        Hooks run immediately before every sandbox execution. They are intended
        for framework-owned variables whose values must be snapshotted at
        compute-turn entry, such as text-table bindings and per-turn budgeted
        callables.
        """
        if not name:
            raise ValueError("hook name must be non-empty")
        with self._lock:
            self._before_execute_hooks[name] = hook

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        code = arguments.get("code", "")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("'code' must be a non-empty string")
        if len(code) > self._max_code_length:
            raise ValueError(f"Code exceeds maximum length of {self._max_code_length} characters")
        return {"code": code.strip()}

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        del context
        code = arguments["code"]
        loop = asyncio.get_running_loop()

        try:
            with self._lock:
                hooks = list(self._before_execute_hooks.values())
            for hook in hooks:
                hook(self)
            result_str = await asyncio.wait_for(
                loop.run_in_executor(_EXECUTOR, self._run_sandboxed, code),
                timeout=self._max_execution_seconds,
            )
            if len(result_str) > self._max_output_chars:
                result_str = result_str[: self._max_output_chars] + "\n... (output truncated)"
            return ToolResult(
                content=result_str,
                success=True,
                data={"source_kind": SourceKind.builtin},
            )
        except TimeoutError:
            logger.warning("COMPUTE_TIMEOUT code_len=%d timeout=%.1f", len(code), self._max_execution_seconds)
            return ToolResult(
                content=f"Error: TimeoutError: Code execution exceeded {self._max_execution_seconds}s limit",
                success=False,
                error="timeout",
                data={"source_kind": SourceKind.builtin},
            )
        except Exception as exc:
            logger.warning("COMPUTE_ERROR type=%s msg=%s", type(exc).__name__, str(exc)[:200])
            return ToolResult(
                content=f"Error: {type(exc).__name__}: {exc}",
                success=False,
                error=str(exc),
                data={"source_kind": SourceKind.builtin},
            )

    # -- Internals -----------------------------------------------------------

    def _run_sandboxed(self, code: str) -> str:
        """Execute *code* in a restricted namespace.  Runs in a thread pool.

        Returns a string combining captured stdout and the value of the last
        expression (if any).
        """
        # Parse and validate AST before execution.
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            raise _SandboxSyntaxError(str(e)) from e

        # Security: block dunder attribute access (prevents sandbox escapes).
        try:
            _validate_ast(tree)
        except ValueError as e:
            raise _SandboxSecurityError(str(e)) from e

        # Detect and capture the last expression's value.
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_node = tree.body.pop()
            assert isinstance(last_node, ast.Expr)
            assign = ast.Assign(
                targets=[ast.Name(id="__result__", ctx=ast.Store())],
                value=last_node.value,
            )
            tree.body.append(assign)
            # Fix all missing locations from the root — required on Python 3.12+.
            ast.fix_missing_locations(tree)

        compiled = compile(tree, "<compute>", "exec")

        # Build execution globals: persistent namespace first, then modules
        # on top (so user variables like `math = 42` cannot shadow modules).
        with self._lock:
            exec_globals: dict[str, Any] = {
                "__builtins__": self._safe_builtins,
                **self._namespace,
                **self._modules,
            }

        # Capture stdout.  exec() runs outside the lock so the timeout
        # mechanism (asyncio.wait_for) can cancel it.
        stdout_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf):
            exec(compiled, exec_globals)  # noqa: S102 — sandboxed exec

        # Persist user-defined variables (exclude builtins and modules).
        with self._lock:
            for k, v in exec_globals.items():
                if k.startswith("__") or k in self._modules:
                    continue
                self._namespace[k] = v

            # Evict oldest entries if namespace grows too large.
            if len(self._namespace) > _MAX_NAMESPACE_ENTRIES:
                keys = list(self._namespace.keys())
                evict_count = len(keys) - _MAX_NAMESPACE_ENTRIES
                for k in keys[:evict_count]:
                    del self._namespace[k]
                logger.info(
                    "COMPUTE_NAMESPACE_EVICTION evicted=%d remaining=%d",
                    evict_count,
                    _MAX_NAMESPACE_ENTRIES,
                )

        # Build output.
        stdout_text = stdout_buf.getvalue()
        result_value = exec_globals.get("__result__")

        parts: list[str] = []
        if result_value is not None:
            parts.append(f"Result: {result_value!r}")
        if stdout_text.strip():
            parts.append(f"Output:\n{stdout_text.rstrip()}")
        if not parts:
            parts.append("(code executed successfully, no output)")

        return "\n\n".join(parts)
