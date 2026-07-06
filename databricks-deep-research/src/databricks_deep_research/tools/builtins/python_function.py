"""Deterministic ``python_function`` tool: fixed design-time code, sandboxed.

Unlike the ``compute`` tool (where the LLM writes code at run time), a
``python_function`` carries a FIXED snippet authored with the workflow (SMA,
reshaping, forecasting glue). It is a plain ``ResearchTool``, so the same
declaration is callable by agents mid-ReAct AND by deterministic tool nodes.

Execution backends (see :mod:`databricks_deep_research.tools.code_executor`):

* ``subprocess`` (default) — the code runs INSIDE the run's persistent
  :class:`SandboxSession` REPL, so any variable it assigns persists for later
  calls; ``bind_result`` names an extra binding for the script's ``result``;
  ``reads_namespace`` declares session variables the code expects (bridged
  from the in-process compute scratchpad when JSON-able).
* ``restricted`` — trusted-only in-process engine; ``reads_namespace`` /
  ``bind_result`` operate on the SHARED compute scratchpad directly (live
  objects allowed, no process boundary).

Script convention (same as skill scripts): declared params become globals,
the script assigns ``result``.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from typing import Any

from databricks_deep_research.tools.builtins.skill_script_executor import SandboxResult
from databricks_deep_research.tools.code_executor import (
    RestrictedCodeExecutor,
    SandboxSession,
)
from databricks_deep_research.tools.protocol import (
    SourceInfo,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)

__all__ = ["PythonFunctionTool", "compile_params_schema"]

_PARAM_TYPES: dict[str, str] = {
    "string": "string",
    "str": "string",
    "integer": "integer",
    "int": "integer",
    "number": "number",
    "float": "number",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "object": "object",
    "dict": "object",
}

_MISSING = object()


def compile_params_schema(params: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compile a declaration's ``params`` list into a JSON-schema object."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in params:
        name = str(param.get("name") or "").strip()
        if not name or not name.isidentifier():
            raise ValueError(f"python_function param has an invalid name: {param!r}")
        prop: dict[str, Any] = {
            "type": _PARAM_TYPES.get(str(param.get("type", "string")).lower(), "string")
        }
        description = param.get("description")
        if isinstance(description, str) and description:
            prop["description"] = description
        if "default" in param:
            prop["default"] = param["default"]
        properties[name] = prop
        if param.get("required"):
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


class PythonFunctionTool:
    """Fixed-code deterministic function implementing the ResearchTool protocol."""

    def __init__(
        self,
        *,
        name: str,
        code: str,
        params: Sequence[dict[str, Any]] = (),
        description: str = "",
        backend: str = "subprocess",
        session_provider: Callable[[], SandboxSession] | None = None,
        restricted_executor: RestrictedCodeExecutor | None = None,
        compute_resolver: Callable[[], Any | None] | None = None,
        extra_allowed_modules: Sequence[str] = (),
        data_lib_mode: str = "facade",
        reads_namespace: Sequence[str] = (),
        bind_result: str | None = None,
        citeable: bool = False,
        timeout_seconds: float = 10.0,
    ) -> None:
        if backend == "subprocess" and session_provider is None:
            raise ValueError("subprocess backend requires a session_provider")
        if backend == "restricted" and restricted_executor is None:
            raise ValueError("restricted backend requires a restricted_executor")
        self._name = name
        self._code = code
        self._params = [dict(p) for p in params]
        self._schema = compile_params_schema(self._params)
        self._description = description
        self._backend = backend
        self._session_provider = session_provider
        self._restricted = restricted_executor
        self._compute_resolver = compute_resolver or (lambda: None)
        self._modules = [m for m in extra_allowed_modules if m]
        self._data_lib_mode = data_lib_mode
        self._reads_namespace = [str(n) for n in reads_namespace if n]
        self._bind_result = bind_result
        self._citeable = citeable
        self._timeout_seconds = timeout_seconds
        self._code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:12]

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description
            or f"Deterministic Python function '{self._name}'",
            parameters=self._schema,
            source_type="function",
            source_kind="qa_assistant" if self._citeable else "builtin",
            metadata={
                "python_function_backend": self._backend,
                "python_function_code_hash": self._code_hash,
            },
        )

    def namespace_snapshot(self) -> str:
        """Sandbox-session scratchpad summary for ``{compute_namespace}``.

        The harness merges this with the compute tool's own snapshot, so
        agents see one combined scratchpad view. Empty string when nothing is
        bound yet (keeps the prompt variable clean).
        """
        if self._backend != "subprocess" or self._session_provider is None:
            return ""  # restricted backend uses the shared compute tool's view
        session = self._session_provider()
        shadow = session.shadow()
        described = session.described()
        if not shadow and not described:
            return ""
        lines: list[str] = []
        total = 0
        for key, value in list(shadow.items())[:50]:
            rendered = repr(value)
            if len(rendered) > 200:
                rendered = rendered[:200] + "..."
            line = f"  {key} = {rendered}"
            total += len(line)
            if total > 2000:
                lines.append(f"  ... ({len(shadow) - len(lines)} more variables)")
                break
            lines.append(line)
        for key, detail in list(described.items())[:20]:
            lines.append(f"  {key}: {detail} (live object in sandbox session)")
        return "Sandbox session variables:\n" + "\n".join(lines)

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        args = dict(arguments or {})
        missing: list[str] = []
        for param in self._params:
            name = str(param.get("name"))
            if name in args:
                continue
            if "default" in param:
                args[name] = param["default"]
            elif param.get("required"):
                missing.append(name)
        if missing:
            raise ValueError(
                f"python_function '{self._name}' missing required argument(s): {missing}"
            )
        # Only declared params reach the sandbox namespace.
        declared = {str(p.get("name")) for p in self._params}
        return {k: v for k, v in args.items() if k in declared}

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        del context  # substrate access is bound at factory create time
        if self._backend == "restricted":
            sandbox_result = await self._run_restricted(arguments)
        else:
            sandbox_result = await self._run_session(arguments)
        return self._to_tool_result(sandbox_result)

    # -- backends -------------------------------------------------------------

    async def _run_session(self, arguments: dict[str, Any]) -> SandboxResult:
        assert self._session_provider is not None  # constructor invariant
        session = self._session_provider()
        await session.ensure_policy(self._modules, self._data_lib_mode)
        for name in self._reads_namespace:
            if name in session.known_names():
                continue
            bridged = await self._bridge_from_compute(session, name)
            if not bridged:
                return SandboxResult(
                    ok=False,
                    error=(
                        f"reads_namespace: variable '{name}' is not bound in the "
                        "sandbox session and no JSON-able value with that name "
                        "exists in the compute scratchpad"
                    ),
                    error_type="KeyError",
                )
        return await session.run(
            self._code,
            arguments,
            bind_result=self._bind_result,
            timeout=self._timeout_seconds,
        )

    async def _bridge_from_compute(self, session: SandboxSession, name: str) -> bool:
        compute = self._compute_resolver()
        if compute is None:
            return False
        value = compute.get_variable(name, _MISSING)
        if value is _MISSING:
            return False
        return await session.inject(name, value)

    async def _run_restricted(self, arguments: dict[str, Any]) -> SandboxResult:
        assert self._restricted is not None  # constructor invariant
        shared_compute = self._compute_resolver()
        for name in self._reads_namespace:
            if shared_compute is None:
                return SandboxResult(
                    ok=False,
                    error=(
                        f"reads_namespace: no compute scratchpad in this run to "
                        f"read '{name}' from"
                    ),
                    error_type="KeyError",
                )
            value = shared_compute.get_variable(name, _MISSING)
            if value is _MISSING:
                return SandboxResult(
                    ok=False,
                    error=f"reads_namespace: variable '{name}' is not bound",
                    error_type="KeyError",
                )
            self._restricted.engine.inject_variable(name, value)
        result = await self._restricted.run(self._code, arguments)
        if result.ok and self._bind_result and shared_compute is not None:
            shared_compute.inject_variable(self._bind_result, result.result)
        return result

    # -- result shaping ---------------------------------------------------------

    def _to_tool_result(self, res: SandboxResult) -> ToolResult:
        stdout = res.stdout or ""
        if res.ok:
            summary = f"{self._name} completed"
            if res.result is not None:
                summary += f"; result: {_preview(res.result)}"
            content = summary + (f"\n{stdout}" if stdout.strip() else "")
        else:
            content = f"{self._name} failed: {res.error}" + (
                f"\n{stdout}" if stdout.strip() else ""
            )
        data: dict[str, Any] = {"result": res.result, "stdout": stdout}
        if res.note:
            data["note"] = res.note
        sources: list[SourceInfo] = []
        if res.ok and self._citeable:
            sources.append(
                SourceInfo(
                    url=f"function://{self._name}/{self._code_hash}",
                    title=self._description or self._name,
                    snippet=content[:800],
                    content=content,
                    source_type="function",
                    source_kind="qa_assistant",
                )
            )
        return ToolResult(
            content=content,
            success=res.ok,
            sources=sources,
            data=data,
            error=res.error or None,
        )


def _preview(value: Any, limit: int = 200) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…"
