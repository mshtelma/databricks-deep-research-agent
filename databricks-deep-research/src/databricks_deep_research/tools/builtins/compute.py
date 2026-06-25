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
    - NO LIVE MODULE OBJECT IS REACHABLE FROM THE SANDBOX (structural invariant,
      review R6).  Every allowed module is injected as a CURATED FACADE
      (``compute_dataframe.build_stdlib_facade`` / ``build_dataframe_modules``),
      never the real module: the facade copies the module's public callables/
      classes/constants but DROPS every attribute that is itself a live module.
      This kills the whole class of attribute-reach escapes where a stdlib module
      re-exports another real module as a plain non-dunder attribute —
      ``calendar.sys.modules['os'].system('id')`` (``calendar`` does ``import
      sys``), ``fractions.operator.methodcaller('__subclasses__')(type(()))``,
      ``statistics.sys`` / ``datetime.sys`` / ``re.functools`` /
      ``collections._sys`` — each invisible to the AST attr-name guard (a
      ``Subscript`` or non-dunder ``Attribute`` it never inspects).  A build-time
      assertion (``assert_no_live_module``) runs over EVERY injected module, so a
      future stdlib version that grows a module-typed public attribute fails at
      construction, not in production.
    - No module/builtin reachable from the sandbox exposes a STRING-KEYED
      attribute/code/import fetcher (``operator.attrgetter``/``methodcaller``,
      ``string.Formatter().get_field``/``vformat``, ``importlib``, ``getattr``,
      ``types.FunctionType``/``CodeType``).  Such a primitive fetches a denylisted
      attribute by STRING at runtime and is invisible to the AST attr-name guard
      (review R5); ``operator`` and ``string`` were removed for exactly this
      reason.  Residual: ``str.format``/``format_map`` field-access
      (``"{0.__class__}".format(obj)``) can READ a class object but cannot express
      a CALL, so it cannot reach ``__subclasses__()``/``__globals__`` to escalate
      to RCE (documented + tested in ``test_compute``).
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
import re as _re_mod
import statistics
import sys
import textwrap as _textwrap_mod
import threading
import types
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from databricks_deep_research.tools.builtins.compute_dataframe import (
    FACADED_ROOTS as _FACADED_ROOTS,
)
from databricks_deep_research.tools.builtins.compute_dataframe import (
    SANDBOX_ALLOWED_DOTTED_IMPORTS as _SANDBOX_ALLOWED_DOTTED_IMPORTS,
)
from databricks_deep_research.tools.builtins.compute_dataframe import (
    assert_no_live_module as _assert_no_live_module,
)
from databricks_deep_research.tools.builtins.compute_dataframe import (
    build_stdlib_facade as _build_stdlib_facade,
)
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
#
# SECURITY (review R5 — string-keyed attribute-fetcher escape class). Every
# module here is vetted to expose NO runtime string->attribute / string->code /
# string->import primitive, because such a primitive is INVISIBLE to the AST
# attr-name guard (which only sees literal ``ast.Attribute.attr`` / ``ast.Name.id``).
# ``operator`` and ``string`` were REMOVED for exactly this reason — they exposed
# ``operator.attrgetter('__class__.__bases__')`` / ``operator.methodcaller(
# '__subclasses__')`` / ``string.Formatter().get_field('w.__init__.__globals__',
# ...)``, each a proven RCE that fetches a denylisted attribute by STRING and so
# bypasses the literal-attr guard.  The remaining modules were swept and are safe:
#   * ``functools`` — ``reduce`` (pure) and ``partial`` (binds args to an ALREADY
#     referenced callable; it is NOT a string-keyed fetcher and cannot obtain a
#     dangerous callable, since getattr/attrgetter/methodcaller/dunder reach are
#     all already blocked).
#   * ``re`` — ``re.compile`` compiles a REGEX, not Python code (it is not the
#     builtin ``compile``, which is absent from ``_SAFE_BUILTINS_BASE``).
#   * ``json`` — ``loads``/``load`` parse JSON text to plain data; no pickle, no
#     code execution.
#   * ``statistics`` re-exports ``functools.reduce`` (pure); ``math``/``decimal``/
#     ``fractions``/``itertools``/``collections``/``copy``/``calendar``/
#     ``datetime``/``textwrap`` expose only pure-compute callables/types.
# INVARIANT: do NOT add a module that exposes a string-keyed attribute/code/import
# fetcher (attrgetter/methodcaller/Formatter.get_field/vformat/import_module/
# FunctionType/CodeType/eval/exec/compile/__import__/getattr); expose a curated
# facade WITHOUT those primitives instead (cf. compute_dataframe).
#
# SECURITY (review R6 — live-module attribute-reach escape class). The values
# below are the REAL stdlib modules, but they are NEVER injected into the sandbox
# directly: each is wrapped in a curated FACADE (``_build_allowed_facades`` ->
# ``compute_dataframe.build_stdlib_facade``) that drops every module-typed
# attribute.  The real modules are retained ONLY for the import-path's dotted
# stdlib-submodule traversal (``from collections.abc import Mapping``), where
# CPython's IMPORT_FROM extracts the named non-module symbol and never binds the
# traversed submodule to the sandbox namespace.  Reaching a re-exported real
# module by ATTRIBUTE (``calendar.sys`` / ``fractions.operator`` /
# ``statistics.sys`` / ``re.functools`` / ``collections._sys``) — the proven R6
# RCE (``calendar.sys.modules['os'].system('id')``) — is therefore impossible:
# the facade simply has no such attribute.
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
    "textwrap": _textwrap_mod,
}


def _build_allowed_facades(
    real_modules: dict[str, Any],
    *,
    extra_denied: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Return ``{name: curated-facade}`` for *real_modules* (security review R6).

    Every entry is wrapped by ``build_stdlib_facade`` so NO live module handle is
    reachable from the sandbox, and the structural invariant is asserted on each
    facade at construction.  *extra_denied* additionally omits any explicitly
    denylisted symbol names (e.g. the dataframe AST method denylist).
    """
    facades: dict[str, Any] = {}
    for name, real_module in real_modules.items():
        facade = _build_stdlib_facade(name, real_module, extra_denied=extra_denied)
        _assert_no_live_module(facade)
        facades[name] = facade
    return facades


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
    "__subclasses__", "__bases__", "__base__", "__mro__", "__class__",
    "__subclasshook__",
    # Attribute / code introspection escape
    "__globals__", "__code__", "__func__", "__self__",
    "__dict__",
    # Function/closure introspection escape (code-action closures bind a
    # weakref + tool-name str only, but block these for defense-in-depth so
    # sandbox code cannot even read a closure's captured cells).
    "__closure__", "__wrapped__", "__defaults__", "__kwdefaults__",
    # Object lifecycle (bound methods expose __globals__)
    "__init__", "__new__", "__del__",
    "__reduce__", "__reduce_ex__",
    # Descriptor protocol escape. ``__getattribute__`` is the LAST string->attribute
    # reflection primitive (getattr / __getattr__ / vars / globals / locals / eval /
    # exec / compile / __import__ are already unreachable): WITHOUT it, sandbox code
    # smuggles a blocked attribute NAME as a STRING —
    # ``obj.__getattribute__("__class__")`` / ``type(object).__getattribute__(object,
    # "__subclasses__")()`` — invisible to the AST attr-name guard, re-opening the
    # ``().__class__...__subclasses__() -> _wrap_close.__init__.__globals__["system"]``
    # RCE escape (security review R4). Closing it is complete-by-construction.
    "__getattr__", "__getattribute__", "__setattr__", "__delattr__",
    "__set_name__", "__init_subclass__",
    # Dataframe-facade real-module handle (spec §5.2, R2 escape). The curated
    # facade no longer stashes the real backing module under this name (see
    # compute_dataframe._build_facade), but it is denylisted as defense-in-depth
    # so even a stray real-module handle cannot be reached by name:
    # ``np.__compute_real_module__.ctypeslib.ctypes.CDLL(None).system(...)``.
    "__compute_real_module__",
})


def _iter_bound_names(node: ast.AST) -> list[str]:
    """Return every name *node* binds, across ALL of Python's binding forms.

    Used by the reserved-name guard to detect any rebinding of a framework-
    injected name (code-action ``submit`` / tool closures). Completeness matters:
    a single missed binding form is a guard bypass (``with X() as submit:`` /
    ``except E as submit:`` / ``case submit:`` shadowing the gated ``submit``).
    Rather than enumerate every statement type, it relies on two facts:

    * Almost every binding is a Store/Del-context bare ``ast.Name`` — assignment,
      augmented/annotated, walrus, ``for``/``with ... as``/comprehension targets,
      tuple/list/star unpacking, ``del``, and (Python 3.12+) ``type X = ...``
      aliases (whose ``.name`` is a Store ``Name``). Because ``ast.walk`` visits
      every node, checking the ``Name`` itself catches them all — version-agnostic
      (no 3.12-only ``ast.TypeAlias`` class is referenced).
    * The remaining binders are NOT ``Name`` nodes but ``str`` attributes:
      ``def``/``class`` names, function/lambda parameters, ``import ... as``,
      ``except ... as``, and ``match`` capture/star/mapping-rest patterns. These
      are enumerated explicitly below.

    Attribute/subscript targets (``obj.x = ...``) are intentionally ignored: they
    cannot rebind a bare name in the sandbox namespace.
    """
    names: list[str] = []
    # (1) Any Store/Del-context bare Name (the broad majority + 3.12 type alias).
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        names.append(node.id)
    # (2) def / class names.
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.append(node.name)
    # (3) function + lambda parameters.
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = node.args
        params = [*a.posonlyargs, *a.args, *a.kwonlyargs]
        if a.vararg is not None:
            params.append(a.vararg)
        if a.kwarg is not None:
            params.append(a.kwarg)
        names.extend(arg.arg for arg in params)
    # (4) import x / import x as y / from m import n as y.
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        names.extend(alias.asname or alias.name.split(".")[0] for alias in node.names)
    # (5) except E as <name>.
    if isinstance(node, ast.ExceptHandler) and node.name is not None:
        names.append(node.name)
    # (6) match patterns: ``case ... as n`` / ``case n`` (MatchAs), ``case [*n]``
    #     (MatchStar), ``case {**n}`` (MatchMapping rest). All bind ``str`` names.
    if isinstance(node, ast.MatchAs) and node.name is not None:
        names.append(node.name)
    if isinstance(node, ast.MatchStar) and node.name is not None:
        names.append(node.name)
    if isinstance(node, ast.MatchMapping) and node.rest is not None:
        names.append(node.rest)
    # (7) PEP 695 type parameters (3.12+): ``def f[T]()`` / ``class C[T]`` /
    #     ``type X[T] = ...``. Version-safe: ``type_params`` is absent on 3.11
    #     (where the syntax is a SyntaxError, rejected before this runs), so no
    #     3.12-only ``ast.TypeVar`` class is referenced.
    for type_param in getattr(node, "type_params", ()):
        tp_name = getattr(type_param, "name", None)
        if isinstance(tp_name, str):
            names.append(tp_name)
    return names


def _check_facaded_submodule_import(node: ast.AST) -> None:
    """Reject a sandbox-authored import that reaches a facaded library's REAL surface.

    Security (spec §5.2, sandbox-escape fix): the curated pandas/numpy facades
    expose ONLY a safe top-level surface, but two import forms reach a REAL
    backing module and defeat the facade — both proven RCE vectors:

      * A DOTTED ``import`` (``import pandas.io as pio`` → ``pio.pickle.
        read_pickle`` unpickle-RCE; ``import numpy.ctypeslib`` → native-libc RCE).
      * ANY ``from <facaded-root> import <name>`` (R2 escape).  ``from numpy
        import ctypeslib`` does ``__import__('numpy', fromlist=['ctypeslib'])``
        → the facade (whose ``__name__`` is ``"numpy"``), then CPython's
        IMPORT_FROM ``getattr(facade, 'ctypeslib')`` raises AttributeError and
        FALLS BACK to ``sys.modules['numpy.ctypeslib']`` — resolving the REAL
        ``numpy.ctypeslib`` → ``ctypeslib.ctypes.CDLL(None).system(...)``.
        ``from pandas import io`` likewise re-opens the ``pandas.io`` family.

    This parse-time check is the airtight enforcement point: it blocks EVERY
    sandbox import form that could reach a facaded lib's real surface, regardless
    of ``sys.modules`` state or the facade's ``__name__``, and cannot be bypassed
    by reflection (``__import__``/``importlib``/``getattr`` are all unreachable).

    Allowed:
      * ``import numpy`` / ``import pandas`` (bare root → facade).  The facade's
        curated top-level symbols are then reached as attributes
        (``np.array(...)`` / ``pd.DataFrame(...)``), gated by the facade
        allowlist — NOT via ``from`` (which is rejected, killing the IMPORT_FROM
        ``sys.modules`` submodule fallback).

    There are NO dotted-import exceptions (:data:`SANDBOX_ALLOWED_DOTTED_IMPORTS`
    is EMPTY in round 4 — ``matplotlib.pyplot`` was the only entry and matplotlib
    is dropped from the sandbox).  The exception check is retained (against the
    now-empty set) so re-adding a vetted dotted import later needs no code change
    here, only a facade + a set entry.

    A facaded library's OWN internal lazy imports (numpy's ``from numpy._core
    import _methods``) execute in the library's compiled bytecode and are NEVER
    part of this parsed sandbox AST, so they are unaffected.
    """
    if isinstance(node, ast.Import):
        for alias in node.names:
            if "." not in alias.name:
                continue
            if alias.name.split(".", 1)[0] not in _FACADED_ROOTS:
                continue
            if alias.name in _SANDBOX_ALLOWED_DOTTED_IMPORTS:
                continue
            raise ValueError(
                f"submodule import '{alias.name}' is not allowed in the "
                "compute sandbox"
            )
    elif isinstance(node, ast.ImportFrom):
        # Relative imports (level > 0) cannot name a facaded root and have no
        # legitimate use in the sandbox; only guard absolute modules.
        module = node.module
        if (node.level or 0) != 0 or not module:
            return
        # Reject EVERY ``from <facaded-root> import <name>`` — dotted (reaches a
        # real submodule directly) AND bare-root (the IMPORT_FROM ``sys.modules``
        # submodule fallback resolves the REAL ``<root>.<name>`` because the
        # facade's ``__name__`` is ``<root>``).  Use ``import numpy as np;
        # np.array(...)`` instead; the facade is reached via the bare root import.
        if module.split(".", 1)[0] not in _FACADED_ROOTS:
            return
        raise ValueError(
            f"'from {module} import ...' is not allowed in the compute sandbox; "
            "use 'import numpy as np' / 'import pandas as pd' and access symbols "
            "as attributes (e.g. np.array(...))"
        )


def _validate_ast(
    tree: ast.Module,
    reserved_names: frozenset[str] = frozenset(),
    blocked_attrs: frozenset[str] = frozenset(),
    block_facaded_submodules: bool = False,
) -> None:
    """Reject code that accesses dangerous dunders or rebinds reserved names.

    Walks the entire AST and raises ``ValueError`` for attribute access
    (``obj.__builtins__``) or bare name references (``__import__``) that
    match the blocklist.  Legitimate dunders like ``__len__``, ``__add__``,
    ``__contains__``, ``__str__``, ``__repr__`` are NOT blocked.

    When ``reserved_names`` is non-empty (code-action mode, spec §1.4), the
    guard ALSO rejects any rebinding of those names via ANY Python binding form
    (see :func:`_iter_bound_names`: assignment/augmented/annotated/walrus,
    ``for``/``with ... as``/comprehension/star targets, ``del``, ``def``/
    ``class``, function & lambda parameters, ``import ... as``, ``except ... as``,
    ``match`` capture/star/mapping-rest, and 3.12 ``type`` aliases) — so sandbox
    code cannot shadow ``submit`` or a tool closure to subvert the gated bridge.

    When ``blocked_attrs`` is non-empty (dataframe mode, spec §5.2), the guard
    ALSO rejects any attribute access whose (non-dunder) name is in the set —
    e.g. ``df.to_pickle`` / ``arr.tofile`` / ``df.eval``.  These are pandas/numpy
    instance-method reaches to the filesystem / network / pickle / arbitrary
    expression evaluation that bypass the module-level facade allowlist (the
    user constructs the object, so the method is on it regardless of the
    facade).  Combined with the existing getattr/eval/exec blocks, there is no
    dynamic route to reach a blocked name by string, so the parse-time check is
    decisive.  Empty by default => byte-identical to the legacy behaviour.

    When ``block_facaded_submodules`` is set (dataframe mode, spec §5.2), the
    guard ALSO rejects every sandbox-authored DOTTED import rooted at a facaded
    library (see :func:`_check_facaded_submodule_import`) — the airtight fix for
    the dotted-submodule sandbox escape.  ``False`` by default => byte-identical.
    """
    for node in ast.walk(tree):
        if block_facaded_submodules:
            _check_facaded_submodule_import(node)
        if isinstance(node, ast.Attribute) and node.attr in _BLOCKED_DUNDER_ATTRS:
            raise ValueError(
                f"Access to '.{node.attr}' is not allowed in the compute sandbox"
            )
        if isinstance(node, ast.Name) and node.id in _BLOCKED_DUNDER_ATTRS:
            raise ValueError(
                f"Reference to '{node.id}' is not allowed in the compute sandbox"
            )
        if blocked_attrs:
            # Dataframe-mode method/attr denylist (spec §5.2). Reject both
            # ``obj.<name>`` attribute access AND a bare ``<name>`` reference
            # (so a blocked module-level numpy func cannot be aliased either).
            if isinstance(node, ast.Attribute) and node.attr in blocked_attrs:
                raise ValueError(
                    f"Access to '.{node.attr}' is not allowed in the compute "
                    "sandbox (file/network/pickle/eval reach)"
                )
            if isinstance(node, ast.Name) and node.id in blocked_attrs:
                raise ValueError(
                    f"Reference to '{node.id}' is not allowed in the compute "
                    "sandbox (file/network/pickle/eval reach)"
                )
        if not reserved_names:
            continue
        # Reserved-name rebinding guard (code-action only). ``_iter_bound_names``
        # covers EVERY binding form (Store/Del Name + def/class/param/import/
        # except/match), so no per-statement enumeration is needed here.
        for bound in _iter_bound_names(node):
            if bound in reserved_names:
                raise ValueError(
                    f"Rebinding of reserved name '{bound}' is not allowed "
                    "in the compute sandbox"
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
    "fractions, itertools, functools, collections, copy, "
    "calendar, textwrap. "
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
        Every name is gated against a VETTED allowlist (the dataframe roots
        pandas/numpy + the stdlib whitelist :data:`_ALLOWED_IMPORT_MODULES`);
        a name outside it raises ``ValueError`` at construction (security review
        R7 — closes ``allowed_modules=["os"]``-style arbitrary-module injection).
    extra_modules
        Extends the module set (default or replaced) with additional modules.
        Subject to the SAME vetted allowlist as *allowed_modules* — only
        pandas/numpy and the stdlib whitelist are accepted; anything else raises
        ``ValueError`` at construction.  A facaded root (pandas/numpy) requested
        here is ALWAYS routed through the SAFE dataframe facade (as if
        *enable_dataframes* were set), never the generic stdlib facade.  An
        allowlisted-but-not-installed optional dep (pandas/numpy) is skipped with
        a warning.
    """

    def __init__(
        self,
        *,
        name: str = "compute",
        allowed_modules: list[str] | None = None,
        extra_modules: list[str] | None = None,
        enable_dataframes: bool = False,
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

        # ---- Module-name allowlist (security review R7 — MEDIUM) ----
        # ``extra_modules``/``allowed_modules`` are author-supplied (Designer
        # registry, custom-agent YAML). Free-form names were an arbitrary-module
        # injection: ``extra_modules=["os"]`` yielded a working in-sandbox
        # ``os.system`` (the generic facade strips ``os.sys``/``os.path`` module
        # handles but COPIES the ``system`` callable). Gate EVERY requested name
        # against a VETTED allowlist at construction: the curated dataframe roots
        # (``_FACADED_ROOTS`` — pandas/numpy, always routed through the SAFE
        # facade below) plus the stdlib whitelist (``_ALLOWED_IMPORT_MODULES``).
        # Anything else (os/subprocess/socket/requests/uuid/…) is rejected with a
        # clear error — uuid alone re-exports ``os``/``sys`` as public attrs, the
        # exact reach class this closes.
        _vetted_modules = _FACADED_ROOTS | _ALLOWED_IMPORT_MODULES.keys()
        _requested_modules: list[str] = [
            *(allowed_modules or []),
            *(extra_modules or []),
        ]
        _rejected = sorted(
            {m for m in _requested_modules if m not in _vetted_modules}
        )
        if _rejected:
            raise ValueError(
                "Module(s) "
                + ", ".join(repr(m) for m in _rejected)
                + " are not permitted in the compute sandbox. Allowed: "
                + ", ".join(sorted(_vetted_modules))
            )

        # A FACADED root (pandas/numpy) requested via ANY entry path —
        # ``extra_modules`` OR ``allowed_modules`` — MUST receive the SAFE
        # dataframe treatment (sub-facades + ``BLOCKED_DATAFRAME_METHODS`` +
        # ``.ctypes``/dotted-import blocks), NEVER the generic ``build_stdlib_facade``
        # (which copies ``read_pickle``/``np.load``/``.ctypes`` — confirmed
        # unpickle/native-libc RCE, security review R7 CRITICAL). Requesting a
        # faceted root therefore implies the same hardening ``enable_dataframes``
        # provides. ``_dataframe_safe`` is the single effective flag used below.
        _facaded_requested = bool(_FACADED_ROOTS & set(_requested_modules))
        _dataframe_safe = enable_dataframes or _facaded_requested

        # ---- Per-instance module configuration ----
        # ``real_base`` maps name -> REAL module (retained ONLY for the import
        # path's dotted stdlib-submodule traversal). The sandbox is NEVER given a
        # real module: ``base`` below is the curated-facade map that is actually
        # injected and gated on (security review R6 — no live module reachable).
        if allowed_modules is not None:
            # Complete replacement: only listed modules are available.
            real_base: dict[str, Any] = {}
            for mod_name in allowed_modules:
                if mod_name in _ALLOWED_IMPORT_MODULES:
                    real_base[mod_name] = _ALLOWED_IMPORT_MODULES[mod_name]
                else:
                    try:
                        real_base[mod_name] = __import__(mod_name)
                    except ImportError:
                        logger.warning(
                            "COMPUTE_SKIP_MODULE module=%s reason=not_installed",
                            mod_name,
                        )
        else:
            real_base = dict(_ALLOWED_IMPORT_MODULES)

        # Extend with extra modules (third-party or additional stdlib).
        for mod_name in extra_modules or []:
            if mod_name not in real_base:
                try:
                    real_base[mod_name] = __import__(mod_name)
                    logger.info(
                        "COMPUTE_EXTRA_MODULE module=%s status=loaded", mod_name
                    )
                except ImportError:
                    logger.warning(
                        "COMPUTE_SKIP_MODULE module=%s reason=not_installed",
                        mod_name,
                    )

        # ---- Safe pandas/numpy dataframe facades (spec §5.2) ----
        # When enabled, ``import pandas`` / ``import numpy`` in the sandbox
        # resolve to CURATED FACADES (no read_*/load/save/pickle/eval), and the
        # AST guard activates the instance-method denylist. Optional deps degrade
        # gracefully: a missing lib is simply absent from the facade map. The
        # default (disabled) path adds nothing and stays byte-identical.
        self._blocked_attrs: frozenset[str] = frozenset()
        # Activates the AST dotted-submodule-import block (spec §5.2 escape fix)
        # exactly when the pandas/numpy facades are wired in. (matplotlib was
        # dropped from the sandbox in round 4 — in-sandbox chart rendering is
        # deferred pending a safe matplotlib strategy.)
        self._block_facaded_submodules: bool = _dataframe_safe
        if _dataframe_safe:
            from databricks_deep_research.tools.builtins.compute_dataframe import (
                BLOCKED_DATAFRAME_METHODS,
            )

            self._blocked_attrs = BLOCKED_DATAFRAME_METHODS

        # ---- Curate every injected module into a live-module-free facade ----
        # SECURITY (review R6): build a facade for EVERY allowed module and inject
        # ONLY the facade. ``build_stdlib_facade`` drops every module-typed public
        # attribute (``calendar.sys`` / ``fractions.operator`` / ``statistics.sys``
        # / ``re.functools`` / ``collections._sys`` …), and ``_build_allowed_facades``
        # asserts the no-live-module invariant on each at construction. Any symbol
        # in the AST method denylist (dataframe mode) is omitted from the facade
        # too, so a denied name is not even present.
        #
        # Whenever the dataframe-safe path is active (``enable_dataframes`` OR a
        # facaded root requested via ``extra_modules``/``allowed_modules`` —
        # security review R7 CRITICAL), pandas/numpy use their OWN richer curated
        # facades (sub-facades + ndarray-internal handling) from
        # ``build_dataframe_modules``, so they are excluded from the generic
        # stdlib pass and injected from there.  This is what makes
        # ``extra_modules=["numpy","pandas"]`` as safe as ``enable_dataframes=True``:
        # a facaded root NEVER reaches the generic ``build_stdlib_facade`` (which
        # would copy ``read_pickle``/``np.load``/``.ctypes`` — the proven RCE).
        # The old code routed an ``extra_modules`` numpy through the generic facade
        # (copying its IO/pickle/ctypes reach), which was exactly the escape.
        from databricks_deep_research.tools.builtins.compute_dataframe import (
            build_dataframe_modules,
        )

        dataframe_handled = _FACADED_ROOTS if _dataframe_safe else frozenset()
        stdlib_real = {
            name: mod
            for name, mod in real_base.items()
            if name not in dataframe_handled
        }
        base: dict[str, Any] = _build_allowed_facades(
            stdlib_real, extra_denied=self._blocked_attrs
        )
        if _dataframe_safe:
            # Which dataframe roots to inject as curated facades:
            #   * ``enable_dataframes=True`` — the explicit capability switch
            #     exposes BOTH pandas+numpy facades (legacy, byte-identical).
            #   * implied via ``extra_modules``/``allowed_modules`` — inject ONLY
            #     the facaded roots the caller actually requested (so
            #     ``extra_modules=["numpy"]`` does not silently add pandas).
            # ``build_dataframe_modules`` returns curated facades for every
            # INSTALLED root, so an uninstalled optional dep degrades gracefully.
            _wanted_facaded = (
                _FACADED_ROOTS if enable_dataframes else _FACADED_ROOTS & set(real_base)
            )
            for mod_name, facade in build_dataframe_modules().items():
                if mod_name not in _wanted_facaded:
                    continue
                base[mod_name] = facade
                logger.info(
                    "COMPUTE_DATAFRAME module=%s status=facade_loaded", mod_name
                )

        # Facades are what the sandbox sees + what gating keys on. Real modules
        # are retained ONLY for the import path's dotted stdlib-submodule traversal
        # (``from collections.abc import Mapping``), where CPython's IMPORT_FROM
        # extracts the named non-module symbol and never binds the live submodule.
        self._allowed_modules: dict[str, Any] = base
        self._real_modules: dict[str, Any] = stdlib_real
        self._modules: dict[str, Any] = dict(base)

        # ---- Per-instance restricted import (closure) ----
        allowed_ref = self._allowed_modules
        real_ref = self._real_modules
        facaded_roots = _FACADED_ROOTS if _dataframe_safe else frozenset()

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
            root_mod = allowed_ref[root]
            parts = name.split(".")[1:]
            # SECURITY: the curated FACADE is the ONLY surface user code may reach
            # by NAME — for the dataframe roots (numpy/pandas) AND, since review
            # R6, for every stdlib root.  A bare-root import (``import numpy`` /
            # ``import collections``) resolves to the FACADE so omitted top-level
            # symbols (``read_*``/``load`` on numpy; the re-exported live modules
            # ``calendar.sys``/``fractions.operator``/… on stdlib) stay
            # unreachable.  For a FACADED root, ``from numpy import X`` is rejected
            # at PARSE time by ``_check_facaded_submodule_import`` (R2 escape fix —
            # the bare ``__import__('numpy', fromlist=['X'])`` would let CPython's
            # IMPORT_FROM fall back to the real ``sys.modules['numpy.X']``
            # submodule), so that branch is reached for a facaded root ONLY by the
            # library's own ``import numpy`` / by the runtime internal-submodule
            # path below — never by user ``from``.
            if not parts:
                # No submodule path: return the curated facade (root).  With a
                # fromlist, CPython getattrs each requested name off THIS facade;
                # the facade allowlist is the gate.
                #
                # SECURITY (review R6 — IMPORT_FROM ``sys.modules`` fallback): for
                # ``from collections import abc`` CPython does ``getattr(facade,
                # 'abc')`` and, on AttributeError, FALLS BACK to
                # ``sys.modules['collections.abc']`` — binding the REAL submodule
                # (the facade's ``__name__`` is ``'collections'``).  We pre-empt
                # that fallback HERE: if a requested name is absent from the
                # facade, raise ImportError so CPython never reaches the
                # ``sys.modules`` lookup.  Curated symbols (``from collections
                # import OrderedDict``) are present on the facade and resolve
                # normally.  Use ``from collections.abc import Mapping`` (a dotted
                # submodule import, served by real traversal below) to reach a
                # submodule's safe symbols.
                for requested in fromlist or ():
                    if requested == "*":
                        continue
                    if not hasattr(root_mod, requested):
                        raise ImportError(
                            f"cannot import name '{requested}' from '{root}' in "
                            "the compute sandbox"
                        )
                return root_mod

            # ---- DOTTED import path ----
            if root in facaded_roots:
                # SECURITY (spec §5.2, sandbox-escape fix): NEVER traverse the
                # REAL backing module for a facaded root — that defeated the
                # facade and was the proven escape (``import pandas.io as pio`` →
                # ``pio.pickle.read_pickle``). Every sandbox-AUTHORED dotted
                # import rooted at a facaded lib is already rejected at parse
                # time by ``_check_facaded_submodule_import``; this branch is
                # therefore reached ONLY by the library's OWN bytecode
                # re-requesting an already-loaded internal submodule at runtime —
                # numpy's ``__import__('numpy._core._methods', fromlist=())``
                # fired by ``ndarray.mean()``. The submodule is in ``sys.modules``
                # (pre-warmed at facade build); numpy reads it from ``sys.modules``
                # itself, so we satisfy CPython's contract by returning the
                # FACADE — user code never receives the internal module object.
                # (The ``_SANDBOX_ALLOWED_DOTTED_IMPORTS`` exception is retained
                # but EMPTY in round 4 — matplotlib.pyplot was its only entry and
                # matplotlib is dropped.)  Anything else (a fresh / non-cached
                # dotted facade import that somehow reaches here) is rejected.
                if name in _SANDBOX_ALLOWED_DOTTED_IMPORTS and name in sys.modules:
                    return root_mod
                if name in sys.modules and not fromlist:
                    return root_mod
                raise ImportError(
                    f"submodule import '{name}' is not allowed in the "
                    "compute sandbox"
                )

            # Non-facaded allowed root (stdlib whitelist): a DOTTED submodule
            # import such as ``from collections.abc import Mapping`` /
            # ``from numpy.linalg import norm`` (numpy via ``extra_modules``, not a
            # facaded root).  Traverse the REAL module to resolve the submodule.
            #
            # SECURITY (review R6): this branch hands the resolved leaf to CPython
            # ONLY when a ``fromlist`` is present, and CPython's IMPORT_FROM then
            # binds ONLY the explicitly named symbols off that leaf — it does NOT
            # bind the traversed submodule itself to the sandbox namespace
            # (verified: ``from collections.abc import Mapping`` binds ``Mapping``,
            # never ``abc``).  A name whose resolved value is itself a live module
            # is refused, so ``from collections.abc import <a-submodule>`` cannot
            # smuggle a module handle.  Without a fromlist (``import
            # collections.abc``) we return the FACADE root per CPython protocol —
            # the live leaf is never exposed.
            real_root = real_ref.get(root)
            if real_root is None:
                raise ImportError(
                    f"Module '{name}' is not available. Available: "
                    f"{', '.join(sorted(allowed_ref.keys()))}"
                )
            mod = real_root
            for part in parts:
                try:
                    mod = getattr(mod, part)
                except AttributeError:
                    raise ImportError(
                        f"Module '{name}' has no submodule '{part}'"
                    ) from None
            # CPython protocol: with fromlist → return leaf; without → return root
            if fromlist:
                # Refuse to hand back a leaf from which a named import would bind a
                # live MODULE (defense-in-depth for the R6 invariant). IMPORT_FROM
                # only extracts ``fromlist`` names, so reject only when a requested
                # name resolves to a module (``from collections.abc import
                # warnings`` would otherwise bind the real ``warnings`` module).
                for requested in fromlist:
                    if requested == "*":
                        continue
                    if isinstance(getattr(mod, requested, None), types.ModuleType):
                        raise ImportError(
                            f"cannot import name '{requested}' from '{name}' in "
                            "the compute sandbox (live module handle)"
                        )
                return mod
            return root_mod

        # ---- Per-instance safe builtins ----
        self._safe_builtins: dict[str, Any] = dict(_SAFE_BUILTINS_BASE)
        self._safe_builtins["__import__"] = _instance_import

        # ---- Thread safety ----
        self._lock = threading.Lock()

        # ---- Persistent namespace for cross-call variable sharing ----
        self._namespace: dict[str, Any] = {}

        # ---- Per-execution namespace refresh hooks ----
        self._before_execute_hooks: dict[str, Callable[[PythonComputeTool], None]] = {}

        # ---- Reserved sandbox names (code-action bridge, spec §1.4) ----
        # Names that sandbox code may CALL but MUST NOT rebind/shadow (the
        # injected ``submit`` + tool closures). Empty by default => the AST
        # reserved-name guard is inert and behaviour is byte-identical.
        self._reserved_names: frozenset[str] = frozenset()

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

    def reserve_sandbox_names(self, names: frozenset[str]) -> None:
        """Mark ``names`` as reserved — callable but not rebindable in sandbox.

        Used by the code-action bridge (spec §1.4) to protect the injected
        ``submit`` + tool closures from being shadowed by sandbox code. The AST
        guard rejects any assignment/del/def/class/import-as/parameter that
        would rebind a reserved name. Passing an empty set clears the guard.
        """
        with self._lock:
            self._reserved_names = frozenset(names)

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

        # Security: block dunder attribute access (prevents sandbox escapes),
        # rebinding of reserved code-action names (submit / tool closures), and
        # — in dataframe mode — pandas/numpy file/network/pickle/eval methods AND
        # dotted-submodule imports rooted at a facaded lib (spec §5.2 escape fix).
        with self._lock:
            reserved = self._reserved_names
            blocked_attrs = self._blocked_attrs
            block_submodules = self._block_facaded_submodules
        try:
            _validate_ast(tree, reserved, blocked_attrs, block_submodules)
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
                # SECURITY (review R6, defense-in-depth): never persist a LIVE
                # module object across calls. The import path already refuses to
                # bind a live module, but this structural backstop guarantees the
                # cross-call namespace can never carry a ``types.ModuleType``
                # handle — so even a future binding path cannot smuggle one in.
                if isinstance(v, types.ModuleType):
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
