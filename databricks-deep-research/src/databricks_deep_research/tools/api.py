"""Public Python authoring surface for tools — the ``@tool`` decorator.

Adapts a plain Python callable to the :class:`ResearchTool` protocol by
inspecting its type hints and docstring to synthesize a JSON Schema. The
resulting object is fully interoperable with the existing ToolResolver,
ReactLoop, and YAML factories — it can be passed to ``Agent(tools=[...])``
directly, or referenced from YAML via ``kind: decorated`` (see
:mod:`databricks_deep_research.tools.factories.decorated`).

Example::

    from databricks_deep_research.tools.api import tool

    @tool
    def echo(msg: str) -> str:
        '''Echo back a message.'''
        return msg

    @tool(name="search", inject={"vfs": "_framework_vfs"})
    async def search(query: str, top_k: int = 5, *, vfs: Any) -> list[dict]:
        '''Search the vector store.'''
        ...

Design notes:

- ``source_kind=SourceKind.builtin`` is the default so the ReactLoop's
  source-admission pipeline (Jaccard dedup, VS cache) does not mangle
  ``@tool`` outputs. Override via ``source_kind=SourceKind.web`` etc. when
  the tool is a true data source.
- ``inject={"name": "extras_key"}`` reads a value from
  ``ToolContext.extras[extras_key]`` at call time and binds it to the named
  parameter. User-chosen extras keys MUST NOT use the ``_framework_``
  reserved prefix.
- ``requires_confirmation=True`` flags the tool for HITL gating in Phase 2;
  it is recorded in ``ToolDefinition.metadata`` and consumed by the
  ReactLoop only when an :class:`ApprovalBroker` is attached.
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast, get_type_hints

from pydantic import BaseModel, Field, TypeAdapter

from databricks_deep_research.tools.introspect import parse_google_docstring
from databricks_deep_research.tools.protocol import (
    SourceInfo,
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

# Parameter names that are auto-detected as the ToolContext / RunContext slot
# regardless of typing. Any of these names triggers per-call context injection.
CTX_PARAM_NAMES: frozenset[str] = frozenset({"ctx", "context", "run_context"})

# Public alias for users authoring tools that accept a per-call context.
RunContext = ToolContext


@dataclass
class Cite:
    """Marker for a cited source emitted by a ``@tool`` callable.

    A tool that wants to surface evidence to the citation pipeline can
    return ``Cite(url=..., title=..., snippet=...)`` instances either as the
    sole result or alongside textual content via :class:`ToolResult`.
    """

    url: str
    title: str = ""
    snippet: str = ""
    content: str | None = None
    source_type: str = "web"
    source_kind: str | None = None
    relevance_score: float | None = None

    def to_source_info(self) -> SourceInfo:
        return SourceInfo(
            url=self.url,
            title=self.title,
            snippet=self.snippet,
            content=self.content,
            source_type=self.source_type,
            source_kind=self.source_kind,
            relevance_score=self.relevance_score,
        )


def _build_param_specs(
    fn: Callable[..., Any],  # noqa: ARG001 — kept for API symmetry/future use
    sig: inspect.Signature,
    hints: dict[str, Any],
    inject: dict[str, str],
    arg_descriptions: dict[str, str],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Collect per-parameter specs for direct schema synthesis.

    Returns the auto-detected context parameter name (or ``None``) and a list
    of ``{name, annotation, default, required, description}`` dicts ordered
    as the function signature.
    """
    specs: list[dict[str, Any]] = []
    ctx_param: str | None = None
    inject_keys = set(inject.keys())

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        annotation = hints.get(param_name, param.annotation)

        # Detect ToolContext/RunContext parameter — auto-injected at call time.
        if param_name in CTX_PARAM_NAMES or annotation is ToolContext:
            ctx_param = param_name
            continue

        # Skip injected keyword-only dependencies.
        if param_name in inject_keys:
            continue

        # Skip *args / **kwargs — the LLM never targets these.
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        if annotation is inspect.Parameter.empty:
            annotation = str
        is_required = param.default is inspect.Parameter.empty
        default = None if is_required else param.default
        description = arg_descriptions.get(param_name, "")

        specs.append({
            "name": param_name,
            "annotation": annotation,
            "default": default,
            "required": is_required,
            "description": description,
        })

    return ctx_param, specs


def _schema_from_specs(specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a JSON Schema dict for the parameter specs.

    Uses :class:`TypeAdapter` per-parameter to avoid the ``create_model``
    forward-ref pitfalls (which surface for types defined in inner scopes
    like test functions).
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    for spec in specs:
        annotation = spec["annotation"]
        sub_schema: dict[str, Any]
        try:
            sub_schema = TypeAdapter(annotation).json_schema()
        except Exception:
            # Fallback: stringify (covers exotic types like Callable).
            sub_schema = {"type": "string"}
        sub_schema = _inline_refs(sub_schema)
        sub_schema.pop("title", None)
        if spec["description"]:
            sub_schema["description"] = spec["description"]
        if not spec["required"]:
            sub_schema["default"] = spec["default"]
        properties[spec["name"]] = sub_schema
        if spec["required"]:
            required.append(spec["name"])

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _build_validator(specs: list[dict[str, Any]]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a callable that validates an args dict against the param specs."""
    adapters: dict[str, TypeAdapter[Any]] = {}
    for spec in specs:
        try:
            adapters[spec["name"]] = TypeAdapter(spec["annotation"])
        except Exception:
            # Skip — exotic types fall through unchanged.
            adapters[spec["name"]] = None  # type: ignore[assignment]

    def _validate(arguments: dict[str, Any]) -> dict[str, Any]:
        validated: dict[str, Any] = {}
        for spec in specs:
            name = spec["name"]
            if name in arguments:
                value = arguments[name]
                adapter = adapters.get(name)
                if adapter is not None:
                    with contextlib.suppress(Exception):
                        value = adapter.validate_python(value)
                validated[name] = value
            elif spec["required"]:
                raise ValueError(f"Missing required argument: {name!r}")
            else:
                validated[name] = spec["default"]
        return validated

    return _validate


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$defs`` references so OpenAI tool calling accepts the schema.

    Pydantic emits ``$ref`` / ``$defs`` for nested models; OpenAI's tool
    calling JSON Schema dialect handles ``$defs`` but some providers don't.
    We resolve a single level of ``$ref`` by replacing each ``{"$ref":
    "#/$defs/X"}`` with the inlined definition. Cyclic references are left
    alone (rare for tool args; pydantic raises on recursion in
    ``model_json_schema`` anyway).
    """
    defs = schema.pop("$defs", None) or schema.pop("definitions", None) or {}

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and len(node) == 1:
                ref_path = node["$ref"]
                # ``#/$defs/Name`` or ``#/definitions/Name``
                key = ref_path.rsplit("/", 1)[-1]
                target = defs.get(key)
                if target is None:
                    return node
                return _resolve(target)
            return {k: _resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return cast(dict[str, Any], _resolve(schema))


def _normalize_tool_output(result: Any) -> ToolResult:
    """Normalize a tool's return value into a :class:`ToolResult`.

    Supported shapes:
    - ``ToolResult`` → returned unchanged.
    - ``Cite`` / ``list[Cite]`` → empty content, sources populated.
    - ``str`` → ``content=str``.
    - ``BaseModel`` → ``content=model.model_dump_json()``, ``data=model.model_dump()``.
    - ``dict`` / ``list`` → ``content=json.dumps(...)``, ``data=value`` (for dicts).
    - Anything else → ``content=str(value)``.
    """
    import json

    if isinstance(result, ToolResult):
        return result

    if isinstance(result, Cite):
        return ToolResult(content="", sources=[result.to_source_info()])

    if isinstance(result, list) and result and all(isinstance(x, Cite) for x in result):
        return ToolResult(
            content="",
            sources=[c.to_source_info() for c in cast(list[Cite], result)],
        )

    if isinstance(result, BaseModel):
        return ToolResult(
            content=result.model_dump_json(),
            data=result.model_dump(mode="json"),
        )

    if isinstance(result, str):
        return ToolResult(content=result)

    if isinstance(result, dict):
        return ToolResult(
            content=json.dumps(result, default=str, ensure_ascii=False),
            data=result,
        )

    if isinstance(result, list):
        return ToolResult(content=json.dumps(result, default=str, ensure_ascii=False))

    return ToolResult(content=str(result))


@dataclass
class _DecoratedTool:
    """Adapts a plain Python callable to the :class:`ResearchTool` protocol.

    Constructed by :func:`tool`. Implements ``definition``,
    ``validate_arguments``, and ``execute`` per the ``ResearchTool`` protocol.
    """

    fn: Callable[..., Any]
    _definition: ToolDefinition
    _validator: Callable[[dict[str, Any]], dict[str, Any]]
    _ctx_param: str | None
    _inject: dict[str, str]
    _is_async: bool

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._validator(arguments or {})

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        # Validate arguments idempotently — safe to call even if the harness
        # already validated; ``_validate`` re-runs cheap TypeAdapters and
        # injected/context parameters never appear in the spec list.
        validated = self.validate_arguments(arguments or {})
        kwargs: dict[str, Any] = dict(validated)
        if self._ctx_param:
            kwargs[self._ctx_param] = context
        for param_name, extras_key in self._inject.items():
            kwargs[param_name] = context.extras.get(extras_key)

        if self._is_async:
            result = await self.fn(**kwargs)
        else:
            result = self.fn(**kwargs)
        return _normalize_tool_output(result)

    # -- helpers used by tests / introspection ---------------------------------

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._definition.parameters

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow the decorated tool to remain callable like the original fn."""
        return self.fn(*args, **kwargs)


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    inject: dict[str, str] | None = None,
    requires_confirmation: bool = False,
    source_kind: str = SourceKind.builtin,
    source_type: str = "user_function",
) -> Any:
    """Decorate a callable to expose it as a :class:`ResearchTool`.

    The decorator inspects the callable's signature, type hints, and Google-style
    docstring to synthesize a JSON Schema for OpenAI tool calling. The original
    function remains callable; the returned object is also a
    :class:`ResearchTool`.

    Args:
        fn: The function to decorate. When called as ``@tool`` (no parentheses)
            this is the function; when called as ``@tool(...)`` it is ``None``.
        name: Override the LLM-visible tool name (defaults to ``fn.__name__``).
        description: Override the docstring-derived description.
        inject: Map of parameter name → ``ToolContext.extras`` key. The named
            parameter is bound from ``context.extras[key]`` at execution time.
        requires_confirmation: If ``True``, sets ``metadata["requires_confirmation"]``
            on the tool definition, opting the tool into HITL gating (Phase 2).
        source_kind: Source kind for ReactLoop admission policy. Default
            ``BUILTIN`` ensures the fast path; set to ``SourceKind.web`` etc.
            when the tool is a real data source.
        source_type: ``ToolDefinition.source_type`` value (default
            ``"user_function"``).

    Returns:
        A :class:`_DecoratedTool` (also conforms to :class:`ResearchTool`).
    """

    def _wrap(target: Callable[..., Any]) -> _DecoratedTool:
        sig = inspect.signature(target)
        # Resolve type hints using the function's globals merged with the
        # caller's locals — important when types like ``Literal`` or
        # user-defined enums are imported inside the calling scope (e.g. test
        # functions). Falls back to ``param.annotation`` for any unresolved
        # entries.
        globalns = dict(getattr(target, "__globals__", {}) or {})
        localns: dict[str, Any] = {}
        # Collect every frame's locals from the call stack that lives outside
        # this module — covers types imported inside test functions, factory
        # closures, etc. We merge in walk order so outer scopes win on key
        # collisions (matches Python's normal lookup order).
        this_module = __name__
        caller_frame = inspect.currentframe()
        f = caller_frame.f_back if caller_frame is not None else None
        while f is not None:
            frame_module = f.f_globals.get("__name__")
            if frame_module != this_module:
                for k, v in f.f_locals.items():
                    localns.setdefault(k, v)
            f = f.f_back
        try:
            hints = get_type_hints(
                target, globalns=globalns, localns=localns, include_extras=True,
            )
        except Exception:
            hints = {}

        parsed = parse_google_docstring(inspect.getdoc(target))
        ctx_param, specs = _build_param_specs(
            target, sig, hints, inject or {}, parsed.args
        )
        validator = _build_validator(specs)
        params_schema = _schema_from_specs(specs)

        tool_name = name or target.__name__
        tool_description = description or parsed.summary or (inspect.getdoc(target) or "").split("\n", 1)[0]

        metadata: dict[str, Any] = {}
        # Pick up @requires_approval markers attached to the raw function
        # (the inner-then-outer decorator stacking pattern).
        wants_approval = requires_confirmation or bool(
            getattr(target, "_dr_requires_approval", False)
        )
        if wants_approval:
            metadata["requires_confirmation"] = True
        approval_reason = getattr(target, "_dr_approval_reason", None)
        if approval_reason:
            metadata["approval_reason"] = approval_reason

        definition = ToolDefinition(
            name=tool_name,
            description=tool_description,
            parameters=params_schema,
            source_type=source_type,
            source_kind=source_kind,
            metadata=metadata,
        )

        return _DecoratedTool(
            fn=target,
            _definition=definition,
            _validator=validator,
            _ctx_param=ctx_param,
            _inject=dict(inject or {}),
            _is_async=inspect.iscoroutinefunction(target),
        )

    if fn is None:
        return _wrap
    return _wrap(fn)


# Lightweight Annotated alias users can apply to expose a parameter description
# explicitly, equivalent to the ``Args:`` block in the docstring.
def Description(text: str) -> Any:  # noqa: N802 — matches PEP-style helper name
    """Return ``Field(description=text)`` for use inside ``Annotated[...]``.

    Example::

        @tool
        def add(
            a: Annotated[int, Description("first number")],
            b: Annotated[int, Description("second number")],
        ) -> int:
            return a + b
    """
    return Field(description=text)


__all__ = [
    "Cite",
    "Description",
    "RunContext",
    "_DecoratedTool",
    "tool",
]
