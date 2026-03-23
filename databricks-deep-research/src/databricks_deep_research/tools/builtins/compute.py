"""Python compute tool — sandboxed code execution for deterministic calculations.

Provides a ``PythonComputeTool`` that executes LLM-generated Python code in a
restricted sandbox.  Only ``math`` and ``statistics`` are available.  Variables
persist across calls within a single tool instance (i.e., within one workflow
run), enabling multi-step computation.

Security:
    - ``__import__``, ``exec``, ``eval``, ``compile``, ``open`` are blocked
    - ``getattr``/``setattr``/``delattr`` are blocked (prevents attribute escape)
    - Timeout via ``asyncio.wait_for`` + thread pool executor
    - Output truncated to ``max_output_chars``
"""

from __future__ import annotations

import ast
import asyncio
import collections as _collections_mod
import contextlib
import decimal as _decimal_mod
import fractions as _fractions_mod
import functools as _functools_mod
import io
import itertools as _itertools_mod
import logging
import math
import re as _re_mod
import statistics
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

_SAFE_BUILTINS: dict[str, Any] = {
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
}


def _restricted_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Allow importing only whitelisted modules in the compute sandbox."""
    if name not in _ALLOWED_IMPORT_MODULES:
        raise ImportError(
            f"Module '{name}' is not available. "
            "math and statistics are pre-imported — use them directly."
        )
    return _ALLOWED_IMPORT_MODULES[name]


_SAFE_BUILTINS["__import__"] = _restricted_import

_MAX_CODE_LENGTH = 5_000

_DEFAULT_DESCRIPTION = (
    "Execute Python code for calculations. "
    "The `math` and `statistics` modules are pre-imported. "
    "Variables persist across calls within the same session. "
    "Use for any computation: percentages, averages, standard deviations, "
    "growth rates, regressions, geometric means, etc."
)

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="compute")


class _SandboxSyntaxError(Exception):
    """Raised inside the sandbox to signal a SyntaxError back to execute()."""


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


class PythonComputeTool:
    """Sandboxed Python code execution tool.

    Implements ``ResearchTool`` protocol.  Intended for deterministic numerical
    computation within agent workflows — the LLM writes Python code and receives
    the printed output and/or the value of the last expression.
    """

    def __init__(
        self,
        *,
        name: str = "compute",
        allowed_modules: list[str] | None = None,
        max_execution_seconds: float = 10.0,
        max_output_chars: int = 10_000,
        description: str = "",
    ) -> None:
        self._name = name
        self._max_execution_seconds = max_execution_seconds
        self._max_output_chars = max_output_chars
        self._description = description or _DEFAULT_DESCRIPTION

        # Build the pre-injected modules dict.
        self._modules: dict[str, Any] = {"math": math, "statistics": statistics}
        for mod_name in allowed_modules or []:
            if mod_name not in self._modules:
                logger.warning("COMPUTE_SKIP_MODULE module=%s reason=not_whitelisted", mod_name)

        # Persistent namespace for cross-call variable sharing.
        self._namespace: dict[str, Any] = {}

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
                            "Python code to execute. `math` and `statistics` modules "
                            "are pre-imported. Variables persist across calls. "
                            "Use print() or end with an expression to see results."
                        ),
                        "maxLength": _MAX_CODE_LENGTH,
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            source_type="builtin",
            source_kind=SourceKind.builtin,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        code = arguments.get("code", "")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("'code' must be a non-empty string")
        if len(code) > _MAX_CODE_LENGTH:
            raise ValueError(f"Code exceeds maximum length of {_MAX_CODE_LENGTH} characters")
        return {"code": code.strip()}

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        code = arguments["code"]
        loop = asyncio.get_running_loop()

        try:
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
        # Build execution globals: safe builtins + whitelisted modules + persistent ns.
        exec_globals: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            **self._modules,
            **self._namespace,
        }

        # Try to detect and capture the last expression's value.
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            raise _SandboxSyntaxError(str(e)) from e

        if tree.body and isinstance(tree.body[-1], ast.Expr):
            # Wrap the last expression: __result__ = <expr>
            last_node = tree.body.pop()
            assign = ast.Assign(
                targets=[ast.Name(id="__result__", ctx=ast.Store())],
                value=last_node.value,
            )
            tree.body.append(assign)
            # Fix all missing locations from the root — required on Python 3.12+.
            ast.fix_missing_locations(tree)

        compiled = compile(tree, "<compute>", "exec")

        # Capture stdout.
        stdout_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf):
            exec(compiled, exec_globals)  # noqa: S102 — sandboxed exec

        # Persist user-defined variables (exclude builtins and modules).
        for k, v in exec_globals.items():
            if k.startswith("__") or k in self._modules:
                continue
            self._namespace[k] = v

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
