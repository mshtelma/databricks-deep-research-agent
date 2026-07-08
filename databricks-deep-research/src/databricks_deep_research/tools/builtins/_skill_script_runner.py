"""Self-contained sandboxed runner for skill scripts (the in-process boundary).

This module has **two roles**, both stdlib-only by design:

1. As an importable library, it exposes :func:`validate_script_source` (the AST
   policy) and the safe-execution primitives. The PARENT process
   (:mod:`skill_script_executor`) imports :func:`validate_script_source` for a
   fail-fast pre-check before it ever spawns a subprocess.

2. As ``python -I _skill_script_runner.py`` (a subprocess ``__main__``), it reads
   ``{code, args, max_output_bytes}`` as JSON from stdin, applies the SAME AST
   policy, executes the code in a restricted namespace, and writes a single JSON
   result object to stdout. User ``print`` output is captured (redirected) so it
   never collides with the result envelope on stdout.

Why a SEPARATE, self-contained runner (not a reuse of ``compute.py``'s in-process
threadpool sandbox)? A Codex security review established that the compute path is
NOT a hard boundary: its ``ThreadPoolExecutor`` orphans the worker thread on
timeout (the code keeps running), it has no memory cap, and it shares the parent
process's secrets (``os.environ``) and file descriptors. Skill scripts are
USER-AUTHORED, so they need a real OS boundary. The boundary is provided by the
parent (subprocess + ``setrlimit`` + scrubbed env + ``SIGKILL`` on timeout); this
module keeps the in-subprocess trusted computing base MINIMAL — pure stdlib, no
framework imports — so it is small enough to audit and cannot itself reach the
framework's secrets even if a policy gap were found.

Defense in depth (each layer independently blocks an escape class):

* **AST policy** (``validate_script_source``): rejects ``import`` of any module
  outside :data:`ALLOWED_MODULES`, every dangerous dunder
  (:data:`_BLOCKED_DUNDER_ATTRS`), and bare references to reflective/eval builtins
  (:data:`_BLOCKED_BUILTIN_NAMES`). Applied in BOTH the parent (fail-fast) and the
  subprocess (authoritative).
* **Restricted ``__builtins__``**: the exec namespace exposes only
  :data:`_SAFE_BUILTIN_NAMES` — no ``eval``/``exec``/``compile``/``open``/
  ``__import__``/``getattr``/``input``. So even if the AST guard were bypassed, an
  ``import`` statement (which calls ``__builtins__.__import__``) routes through the
  allowlisting :func:`_guarded_import` and a non-allowlisted module raises.
* **OS isolation** (parent): rlimits (CPU/AS/FSIZE/NOFILE), a scrubbed env with no
  secrets, and ``SIGKILL`` of the whole process group on the wall-clock timeout.

INVARIANT: keep this module import-free of the framework and of any stdlib module
that exposes a string-keyed attribute/code/import primitive
(``operator.attrgetter`` / ``string.Formatter().get_field`` / ``importlib`` in the
*allowlist*); such a primitive is invisible to the literal-name AST guard. The
:data:`ALLOWED_MODULES` set mirrors ``compute._ALLOWED_IMPORT_MODULES`` and was
vetted on the same grounds (see the security note there); a parity test
(``test_runner_allowlist_matches_compute``) keeps the two in lockstep.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import sys
import types
from typing import Any

__all__ = [
    "ALLOWED_MODULES",
    "SkillScriptPolicyError",
    "run_script",
    "validate_script_source",
]


class SkillScriptPolicyError(Exception):
    """Raised when a skill script violates the static AST policy."""


# ---------------------------------------------------------------------------
# Policy data (mirrors compute.py; kept self-contained — see module docstring)
# ---------------------------------------------------------------------------

# Stdlib modules a skill script may import. Mirrors
# ``compute._ALLOWED_IMPORT_MODULES`` (keys); each was vetted to expose NO
# string-keyed attribute/code/import primitive (which the literal-name AST guard
# could not see). A parity test asserts this stays equal to compute's set.
ALLOWED_MODULES: frozenset[str] = frozenset({
    "math",
    "statistics",
    "decimal",
    "re",
    "fractions",
    "itertools",
    "functools",
    "collections",
    "copy",
    "calendar",
    "datetime",
    "json",
    "textwrap",
})

# Dunder attributes/names that enable sandbox escapes. Copied verbatim from
# ``compute._BLOCKED_DUNDER_ATTRS`` (a parity test asserts this is a superset);
# kept inline so the subprocess never imports the framework. Each blocks a proven
# escape class — module/import (``__builtins__``/``__import__``), class-hierarchy
# (``__class__``/``__subclasses__``/``__bases__``/``__mro__``), code introspection
# (``__globals__``/``__code__``/``__closure__``), and the string->attribute
# reflection primitives (``__getattribute__`` et al.).
_BLOCKED_DUNDER_ATTRS: frozenset[str] = frozenset({
    "__builtins__", "__import__", "__loader__", "__spec__",
    "__subclasses__", "__bases__", "__base__", "__mro__", "__class__",
    "__subclasshook__",
    "__globals__", "__code__", "__func__", "__self__", "__dict__",
    "__closure__", "__wrapped__", "__defaults__", "__kwdefaults__",
    "__init__", "__new__", "__del__", "__reduce__", "__reduce_ex__",
    "__getattr__", "__getattribute__", "__setattr__", "__delattr__",
    "__set_name__", "__init_subclass__",
    "__compute_real_module__",
})

# Builtin NAMES that are never exposed in the exec namespace and are additionally
# rejected by the AST guard (so an attempt reads as a clear policy error rather
# than a ``NameError`` at runtime). These are the reflective / eval / IO / import
# primitives that would defeat the restricted ``__builtins__`` if reachable.
_BLOCKED_BUILTIN_NAMES: frozenset[str] = frozenset({
    "eval", "exec", "compile", "open", "input", "breakpoint",
    "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "dir", "help", "exit", "quit", "copyright", "credits",
    "license", "memoryview",
})

# Safe builtins exposed to skill scripts. Mirrors the keys of
# ``compute._SAFE_BUILTINS_BASE`` (constructors, math, iteration, formatting,
# precise exception types) — deliberately EXCLUDES every name in
# :data:`_BLOCKED_BUILTIN_NAMES`. ``__import__`` is supplied separately as the
# allowlisting :func:`_guarded_import`.
_SAFE_BUILTIN_NAMES: frozenset[str] = frozenset({
    # Constructors / types
    "int", "float", "str", "bool", "complex", "list", "dict", "tuple",
    "set", "frozenset", "bytes", "bytearray", "range", "type",
    "isinstance", "issubclass",
    # Math
    "abs", "round", "min", "max", "sum", "pow", "divmod", "len",
    # Iteration
    "enumerate", "zip", "map", "filter", "reversed", "sorted", "iter",
    "next", "all", "any",
    # Formatting / inspection (return primitives only)
    "print", "repr", "format", "chr", "ord", "hash", "id", "hex", "bin",
    "oct", "ascii", "callable", "hasattr",
    # Object / type machinery (safe; reflective dunders are blocked separately)
    "object", "slice", "property", "staticmethod", "classmethod", "super",
    # Exception types (precise try/except)
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "ZeroDivisionError", "StopIteration", "ArithmeticError", "OverflowError",
    "RuntimeError", "AttributeError", "NameError", "ImportError",
    "LookupError", "NotImplementedError", "FileNotFoundError",
})

# Result envelope size cap (bytes) — a backstop independent of the parent's
# output cap, so a runaway ``result`` cannot produce an unbounded stdout write.
_MAX_RESULT_BYTES_DEFAULT = 64 * 1024
_MAX_SOURCE_LENGTH = 20_000


# ---------------------------------------------------------------------------
# AST policy
# ---------------------------------------------------------------------------

def validate_script_source(
    source: str,
    *,
    allowed_modules: frozenset[str] = ALLOWED_MODULES,
    blocked_attrs: frozenset[str] = frozenset(),
) -> None:
    """Raise :class:`SkillScriptPolicyError` if *source* violates the policy.

    Walks the entire AST and rejects:

    * any ``import``/``from import`` whose ROOT module is not in
      *allowed_modules* (default :data:`ALLOWED_MODULES`);
    * any attribute access or bare name in :data:`_BLOCKED_DUNDER_ATTRS`,
      plus any attribute in *blocked_attrs* (session mode adds data-lib IO
      method names here);
    * any bare reference to a name in :data:`_BLOCKED_BUILTIN_NAMES`.

    This is the SAME check the subprocess applies; the parent calls it first as a
    fail-fast gate so obviously-malicious code never spawns a process.
    """
    if not isinstance(source, str) or not source.strip():
        raise SkillScriptPolicyError("skill script source must be a non-empty string")
    if len(source) > _MAX_SOURCE_LENGTH:
        raise SkillScriptPolicyError(
            f"skill script exceeds maximum length of {_MAX_SOURCE_LENGTH} characters"
        )
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SkillScriptPolicyError(f"skill script has a syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in allowed_modules:
                    raise SkillScriptPolicyError(
                        f"import of '{alias.name}' is not allowed in a skill script"
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if node.level != 0 or root not in allowed_modules:
                raise SkillScriptPolicyError(
                    f"import from '{node.module}' is not allowed in a skill script"
                )
        elif isinstance(node, ast.Attribute) and (
            node.attr in _BLOCKED_DUNDER_ATTRS or node.attr in blocked_attrs
        ):
            raise SkillScriptPolicyError(
                f"access to '.{node.attr}' is not allowed in a skill script"
            )
        elif isinstance(node, ast.Name) and (
            node.id in _BLOCKED_DUNDER_ATTRS or node.id in _BLOCKED_BUILTIN_NAMES
        ):
            raise SkillScriptPolicyError(
                f"reference to '{node.id}' is not allowed in a skill script"
            )
        # MED-1: a string literal can smuggle a blocked dunder past the
        # literal-attr guard via the format mini-language —
        # ``"{0.__class__.__bases__}".format(x)`` reaches the class hierarchy
        # without ever emitting an ``ast.Attribute`` node. Reject any string
        # literal containing a blocked dunder name (also covers a stray
        # ``getattr``-style string, though getattr itself is already blocked).
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and any(blocked in node.value for blocked in _BLOCKED_DUNDER_ATTRS)
        ):
            raise SkillScriptPolicyError(
                "string literal references a blocked attribute name "
                "(dunder reflection is not allowed in a skill script)"
            )


# ---------------------------------------------------------------------------
# Restricted execution
# ---------------------------------------------------------------------------

def _facade_module(module: Any, extra_drops: frozenset[str] = frozenset()) -> Any:
    """Return a safe facade over *module* exposing only non-module public attrs.

    SECURITY (CRITICAL — review CRIT-1, the R6 live-module-reach escape class).
    The real allowlisted stdlib modules re-export OTHER modules as attributes
    (``statistics.sys`` / ``calendar.sys`` / ``fractions.sys`` / ``re.functools``
    …). Returning the live module would let sandbox code reach
    ``statistics.sys.modules['os'].popen('id')`` — full RCE — invisible to the
    literal-attr AST guard. The facade DROPS every module-typed attribute and
    every dunder/private name, so no live module handle is reachable. Mirrors
    ``compute._build_stdlib_facade`` but is kept self-contained (the subprocess
    never imports the framework). Compute's allowlist parity is asserted by a
    test, so no facade'd module exposes a string-keyed attribute/code/import
    primitive either.
    """
    facade = types.SimpleNamespace()
    for attr in dir(module):
        if attr.startswith("_") or attr in _BLOCKED_DUNDER_ATTRS or attr in extra_drops:
            continue
        try:
            value = getattr(module, attr)
        except AttributeError:  # pragma: no cover - defensive
            continue
        if isinstance(value, types.ModuleType):
            continue  # drop re-exported modules — the R6 escape vector
        setattr(facade, attr, value)
    return facade


def _guarded_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Allowlisting ``__import__`` returning a module FACADE (not the live module).

    Runtime backstop to the AST guard: a non-allowlisted module raises here, and
    every returned module is wrapped in :func:`_facade_module` so no live
    re-exported module (``statistics.sys`` …) is reachable. Relative imports
    (``level > 0``) are always rejected.
    """
    del globals, locals
    if level != 0:
        raise ImportError("relative imports are not allowed in a skill script")
    root = name.split(".", 1)[0]
    if root not in ALLOWED_MODULES:
        available = ", ".join(sorted(ALLOWED_MODULES))
        raise ImportError(
            f"module '{name}' is not available in a skill script. Available: {available}"
        )
    module = __import__(name, fromlist=fromlist)
    return _facade_module(module)


def _safe_builtins() -> dict[str, Any]:
    """Build the restricted ``__builtins__`` mapping for the exec namespace."""
    safe: dict[str, Any] = {
        name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES
    }
    safe["__import__"] = _guarded_import
    return safe


def _coerce_result(value: Any, *, max_bytes: int) -> tuple[Any, str | None]:
    """Return a JSON-serialisable view of *value* plus an optional note.

    Skill-script results cross a process boundary as JSON. A non-serialisable
    value is coerced to its ``repr`` (capped) with a note, rather than failing the
    whole run.
    """
    if value is None:
        return None, None
    try:
        encoded = json.dumps(value)
    except (TypeError, ValueError):
        text = repr(value)
        if len(text) > max_bytes:
            text = text[:max_bytes] + "... (truncated)"
        return text, "result was not JSON-serialisable; coerced to repr()"
    if len(encoded) > max_bytes:
        return (
            encoded[:max_bytes] + "... (truncated)",
            "result exceeded the size cap; truncated",
        )
    return value, None


def run_script(
    source: str,
    args: dict[str, Any],
    *,
    max_output_bytes: int = _MAX_RESULT_BYTES_DEFAULT,
) -> dict[str, Any]:
    """Validate, execute *source* in a restricted namespace, and capture output.

    The script may read injected ``args`` (each key becomes a global) and assign a
    ``result`` variable to return a value; anything printed is captured. Returns a
    result envelope: ``{"ok": True, "result": ..., "stdout": ..., "note": ...}`` or
    ``{"ok": False, "error": ..., "error_type": ...}``.
    """
    try:
        validate_script_source(source)
    except SkillScriptPolicyError as exc:
        return {"ok": False, "error": str(exc), "error_type": "SkillScriptPolicyError"}

    namespace: dict[str, Any] = {"__builtins__": _safe_builtins()}
    if isinstance(args, dict):
        for key, value in args.items():
            if isinstance(key, str) and key.isidentifier() and not key.startswith("__"):
                namespace[key] = value

    buffer = io.StringIO()
    try:
        compiled = compile(source, "<skill_script>", "exec")
        with contextlib.redirect_stdout(buffer):
            exec(compiled, namespace)  # noqa: S102 — restricted namespace is the boundary
    except SkillScriptPolicyError as exc:  # pragma: no cover - validated above
        return {"ok": False, "error": str(exc), "error_type": "SkillScriptPolicyError"}
    except BaseException as exc:  # noqa: BLE001 — report ANY failure to the caller
        captured = buffer.getvalue()
        if len(captured) > max_output_bytes:
            captured = captured[:max_output_bytes] + "\n... (output truncated)"
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "error_type": type(exc).__name__,
            "stdout": captured,
        }

    stdout = buffer.getvalue()
    if len(stdout) > max_output_bytes:
        stdout = stdout[:max_output_bytes] + "\n... (output truncated)"
    result_value, note = _coerce_result(
        namespace.get("result"), max_bytes=max_output_bytes
    )
    envelope: dict[str, Any] = {"ok": True, "result": result_value, "stdout": stdout}
    if note is not None:
        envelope["note"] = note
    return envelope


# ---------------------------------------------------------------------------
# Session mode — persistent REPL for python_function tools
# ---------------------------------------------------------------------------
#
# ``python -I -B _skill_script_runner.py --session`` runs a LINE-DELIMITED JSON
# command loop over stdin/stdout instead of the one-shot envelope. The exec
# namespace PERSISTS across commands (the run-scoped MemEx scratchpad lives in
# this process); the parent keeps a JSON-able "shadow" fed by per-exec deltas.
# Same trusted-computing-base rules as one-shot mode: pure stdlib, AST policy +
# restricted builtins in-process, OS boundary (rlimits/env-scrub/SIGKILL) in
# the parent. Exits on stdin EOF so an orphaned child never outlives its parent.

# Data libraries importable in a session when the parent's init allows them.
# Both are exposed as facades by default (module-typed attrs dropped => no
# ``np.linalg``/``pd.api`` submodules; top-level API only) with the IO/eval
# primitives below removed; ``data_lib_mode == "live"`` (trusted configs only)
# returns the real modules.
DATA_LIBS: frozenset[str] = frozenset({"pandas", "numpy"})

# Module-level attrs dropped from data-lib facades: file/DB/pickle IO and
# string-eval primitives that would bypass the no-open/no-eval posture.
_DATA_LIB_FACADE_DROPS: dict[str, frozenset[str]] = {
    "pandas": frozenset({
        "read_pickle", "read_csv", "read_table", "read_fwf", "read_clipboard",
        "read_excel", "read_json", "read_html", "read_xml", "read_hdf",
        "read_feather", "read_parquet", "read_orc", "read_sas", "read_spss",
        "read_sql", "read_sql_query", "read_sql_table", "read_gbq",
        "read_stata", "to_pickle", "eval",
    }),
    "numpy": frozenset({
        "load", "save", "savez", "savez_compressed", "loadtxt", "savetxt",
        "genfromtxt", "fromfile", "memmap", "frombuffer",
    }),
}

# Instance-method names rejected by the AST policy when data libs are active
# (module facades can't see methods; serialization/DB sinks are blocked here,
# residual plain-file writes are bounded by RLIMIT_FSIZE + the tempdir cwd).
_DATA_LIB_BLOCKED_METHOD_ATTRS: frozenset[str] = frozenset({
    "to_pickle", "read_pickle", "to_sql", "to_hdf", "tofile",
})

_SHADOW_VALUE_MAX_BYTES = 8 * 1024
_SHADOW_TOTAL_MAX_BYTES = 256 * 1024


class _Session:
    """Persistent exec namespace + policy state for one session process."""

    def __init__(self) -> None:
        self.allowed: set[str] = set(ALLOWED_MODULES)
        self.data_lib_mode = "facade"
        self.max_output_bytes = _MAX_RESULT_BYTES_DEFAULT
        self.namespace: dict[str, Any] = {"__builtins__": self._build_builtins()}
        # name -> serialized JSON of the last shadow value (delta detection)
        self._shadow_cache: dict[str, str] = {}

    # -- imports ------------------------------------------------------------

    def _guarded_session_import(
        self,
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        del globals, locals
        if level != 0:
            raise ImportError("relative imports are not allowed in a skill script")
        root = name.split(".", 1)[0]
        if root not in self.allowed:
            available = ", ".join(sorted(self.allowed))
            raise ImportError(
                f"module '{name}' is not available in this session. Available: {available}"
            )
        module = __import__(name, fromlist=fromlist)
        if root in DATA_LIBS:
            if self.data_lib_mode == "live":
                return module
            return _facade_module(module, _DATA_LIB_FACADE_DROPS.get(root, frozenset()))
        return _facade_module(module)

    def _build_builtins(self) -> dict[str, Any]:
        safe: dict[str, Any] = {
            name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES
        }
        safe["__import__"] = self._guarded_session_import
        return safe

    # -- shadow -------------------------------------------------------------

    def _shadow_and_described(self) -> tuple[dict[str, Any], dict[str, str]]:
        """JSON-able view of user bindings + type descriptors for the rest."""
        shadow: dict[str, Any] = {}
        described: dict[str, str] = {}
        total = 0
        for key, value in self.namespace.items():
            if key.startswith("_") or key == "__builtins__":
                continue
            try:
                encoded = json.dumps(value)
            except (TypeError, ValueError):
                encoded = None
            if encoded is not None and len(encoded) <= _SHADOW_VALUE_MAX_BYTES:
                if total + len(encoded) > _SHADOW_TOTAL_MAX_BYTES:
                    described[key] = f"{type(value).__name__} (shadow budget exceeded)"
                    continue
                shadow[key] = value
                total += len(encoded)
            else:
                detail = type(value).__name__
                with contextlib.suppress(TypeError):
                    detail += f" len={len(value)}"
                described[key] = detail
        return shadow, described

    def _shadow_delta(self) -> tuple[dict[str, Any], dict[str, str]]:
        shadow, described = self._shadow_and_described()
        delta: dict[str, Any] = {}
        for key, value in shadow.items():
            encoded = json.dumps(value)
            if self._shadow_cache.get(key) != encoded:
                delta[key] = value
                self._shadow_cache[key] = encoded
        for gone in set(self._shadow_cache) - set(shadow):
            del self._shadow_cache[gone]
        return delta, described

    # -- command handlers -----------------------------------------------------

    def handle(self, command: dict[str, Any]) -> dict[str, Any]:
        op = command.get("op")
        if op == "init":
            requested = command.get("extra_allowed_modules") or []
            if isinstance(requested, list):
                self.allowed |= {
                    m for m in requested if isinstance(m, str) and m in DATA_LIBS
                }
            if command.get("data_lib_mode") == "live":
                self.data_lib_mode = "live"
            max_out = command.get("max_output_bytes")
            if isinstance(max_out, int) and max_out > 0:
                self.max_output_bytes = max_out
            return {"ok": True, "op": "init"}
        if op == "exec":
            return self._exec(
                str(command.get("code", "")),
                command.get("args") or {},
                command.get("bind_result"),
            )
        if op == "inject":
            name = command.get("name")
            if not (isinstance(name, str) and name.isidentifier() and not name.startswith("__")):
                return {"ok": False, "error": f"invalid variable name: {name!r}"}
            self.namespace[name] = command.get("value")
            return {"ok": True, "op": "inject"}
        if op == "snapshot":
            shadow, described = self._shadow_and_described()
            return {"ok": True, "op": "snapshot", "shadow": shadow, "described": described}
        if op == "ping":
            return {"ok": True, "op": "ping"}
        if op == "shutdown":
            return {"ok": True, "op": "shutdown"}
        return {"ok": False, "error": f"unknown op: {op!r}"}

    def _exec(
        self, source: str, args: dict[str, Any], bind_result: Any
    ) -> dict[str, Any]:
        data_libs_active = bool(self.allowed & DATA_LIBS)
        try:
            validate_script_source(
                source,
                allowed_modules=frozenset(self.allowed),
                blocked_attrs=(
                    _DATA_LIB_BLOCKED_METHOD_ATTRS if data_libs_active else frozenset()
                ),
            )
        except SkillScriptPolicyError as exc:
            return {"ok": False, "error": str(exc), "error_type": "SkillScriptPolicyError"}

        if isinstance(args, dict):
            for key, value in args.items():
                if isinstance(key, str) and key.isidentifier() and not key.startswith("__"):
                    self.namespace[key] = value
        self.namespace.pop("result", None)

        buffer = io.StringIO()
        try:
            compiled = compile(source, "<python_function>", "exec")
            with contextlib.redirect_stdout(buffer):
                exec(compiled, self.namespace)  # noqa: S102 — restricted namespace is the boundary
        except BaseException as exc:  # noqa: BLE001 — report ANY failure to the caller
            captured = buffer.getvalue()
            if len(captured) > self.max_output_bytes:
                captured = captured[: self.max_output_bytes] + "\n... (output truncated)"
            delta, described = self._shadow_delta()
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
                "stdout": captured,
                "shadow_delta": delta,
                "described": described,
            }

        if isinstance(bind_result, str) and bind_result.isidentifier():
            self.namespace[bind_result] = self.namespace.get("result")

        stdout = buffer.getvalue()
        if len(stdout) > self.max_output_bytes:
            stdout = stdout[: self.max_output_bytes] + "\n... (output truncated)"
        result_value, note = _coerce_result(
            self.namespace.get("result"), max_bytes=self.max_output_bytes
        )
        delta, described = self._shadow_delta()
        envelope: dict[str, Any] = {
            "ok": True,
            "result": result_value,
            "stdout": stdout,
            "shadow_delta": delta,
            "described": described,
        }
        if note is not None:
            envelope["note"] = note
        return envelope


def _session_main() -> int:
    """Session entry point: line-delimited JSON commands until EOF/shutdown."""
    session = _Session()
    while True:
        line = sys.stdin.readline()
        if not line:  # EOF — parent died or closed stdin; never outlive it.
            return 0
        line = line.strip()
        if not line:
            continue
        try:
            command = json.loads(line)
            reply = session.handle(command)
        except Exception as exc:  # noqa: BLE001 — the loop must never die mid-session
            reply = {"ok": False, "error": f"bad command: {exc}", "error_type": type(exc).__name__}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()
        if reply.get("op") == "shutdown":
            return 0


def _main() -> int:
    """Subprocess entry point: stdin JSON -> run -> stdout JSON envelope."""
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except (ValueError, OSError) as exc:
        sys.stdout.write(
            json.dumps({"ok": False, "error": f"bad payload: {exc}", "error_type": "ValueError"})
        )
        return 0
    code = payload.get("code", "")
    args = payload.get("args", {}) or {}
    max_output = int(payload.get("max_output_bytes", _MAX_RESULT_BYTES_DEFAULT))
    envelope = run_script(code, args, max_output_bytes=max_output)
    # The result envelope is the ONLY thing written to the real stdout (user
    # ``print`` output was redirected into the envelope's ``stdout`` field).
    sys.stdout.write(json.dumps(envelope))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    if "--session" in sys.argv[1:]:
        raise SystemExit(_session_main())
    raise SystemExit(_main())
