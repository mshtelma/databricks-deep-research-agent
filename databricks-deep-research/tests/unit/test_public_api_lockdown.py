"""Public-API lockdown lint tests (PR3c — M4).

Three tests guard the framework's public-API contract:

1. Non-framework callers must not reach into private (``_``-prefixed)
   submodules of ``databricks_deep_research``.
2. Every non-private module under ``databricks_deep_research/api/``
   declares ``__all__`` at module level.
3. The reserved ``_framework_*`` extras-key prefix is written only by
   canonical writers; non-framework, non-test source files must not
   construct strings starting with ``_framework_`` for use as keys.

The tests scan the workspace at runtime; they run in well under 5
seconds with no network or LLM access.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# Compute repo paths from this file's location:
# repo/databricks-deep-research/tests/unit/test_public_api_lockdown.py
_THIS_FILE = Path(__file__).resolve()
_FRAMEWORK_PKG = _THIS_FILE.parent.parent.parent  # databricks-deep-research/
_REPO_ROOT = _FRAMEWORK_PKG.parent
_FRAMEWORK_SRC = _FRAMEWORK_PKG / "src" / "databricks_deep_research"
_FRAMEWORK_TESTS = _FRAMEWORK_PKG / "tests"
_APP_PKG = _REPO_ROOT / "databricks-deep-research-app"
_APP_SRC = _APP_PKG / "src"
_API_DIR = _FRAMEWORK_SRC / "api"

# Canonical writers permitted to assign ``_framework_*`` extras keys.
_FRAMEWORK_PREFIX_ALLOWLIST = {
    _FRAMEWORK_SRC / "api" / "agent.py",
    _APP_SRC / "deep_research" / "agent" / "framework_orchestrator.py",
}

# Pattern for ``from databricks_deep_research[.x.y]._private import ...``.
_PRIVATE_IMPORT_RE = re.compile(
    r"from\s+databricks_deep_research(?:\.[a-z_]+)*\._\w+\s+import"
)


def _iter_py_files(root: Path) -> list[Path]:
    """Yield all ``.py`` files under ``root`` (skipping caches)."""
    if not root.exists():
        return []
    return [
        p
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and ".venv" not in p.parts
    ]


# --- Test 1: no private imports outside the framework -------------------


def test_no_private_imports_outside_framework() -> None:
    """App code must not import from underscore-prefixed framework modules."""
    if not _APP_SRC.exists():
        pytest.skip("app source tree not present")

    violations: list[tuple[Path, int, str]] = []
    for path in _iter_py_files(_APP_SRC):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _PRIVATE_IMPORT_RE.search(line):
                violations.append((path, lineno, line.strip()))

    if violations:
        formatted = "\n".join(
            f"  {p.relative_to(_REPO_ROOT)}:{ln}: {src}" for p, ln, src in violations
        )
        pytest.fail(
            "App source files import private framework modules; only public "
            "(non-`_`-prefixed) modules may be imported. Violations:\n"
            + formatted
        )


# --- Test 2: every public api module declares __all__ -------------------


def _module_has_all(path: Path) -> bool:
    """Return True iff the module has a top-level ``__all__`` assignment."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return True
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == "__all__":
                return True
    return False


def test_api_modules_have_all() -> None:
    """Every non-private ``api/*.py`` module must declare ``__all__``."""
    if not _API_DIR.exists():
        pytest.skip("api/ dir not present")

    missing: list[Path] = []
    for path in _API_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        # Skip private submodules (allowed to omit __all__).
        if path.name.startswith("_") and path.name != "__init__.py":
            continue
        # __init__.py is the package surface — required.
        # Other public modules — required.
        if not _module_has_all(path):
            missing.append(path)

    if missing:
        formatted = "\n".join(
            f"  {p.relative_to(_REPO_ROOT)}" for p in missing
        )
        pytest.fail(
            "Public API modules under api/ must declare `__all__`. Missing:\n"
            + formatted
        )


# --- Test 3: only canonical writers use _framework_* extras keys --------


class _FrameworkPrefixVisitor(ast.NodeVisitor):
    """Collect string-literal usages of ``_framework_*`` keys in key-like positions."""

    def __init__(self) -> None:
        self.hits: list[tuple[int, str]] = []

    def _is_framework_str(self, node: ast.expr) -> str | None:
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("_framework_")
        ):
            return node.value
        return None

    # Subscript: d["_framework_..."] = ...
    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        slice_node = node.slice
        s = self._is_framework_str(slice_node)
        if s is not None:
            self.hits.append((node.lineno, s))
        self.generic_visit(node)

    # Dict literal keys: {"_framework_..." : value}
    def visit_Dict(self, node: ast.Dict) -> None:  # noqa: N802
        for key in node.keys:
            if key is None:
                continue
            s = self._is_framework_str(key)
            if s is not None:
                self.hits.append((node.lineno, s))
        self.generic_visit(node)

    # Call kwargs: foo(**{"_framework_...": v}) — covered by visit_Dict.
    # Direct call keyword= is impossible because keyword names are
    # identifiers, not strings.


def test_no_framework_prefix_extras_outside_framework() -> None:
    """Reserved ``_framework_*`` extras keys are written only by canonical writers."""
    targets: list[Path] = []

    # Non-framework production source files (app side).
    if _APP_SRC.exists():
        for path in _iter_py_files(_APP_SRC):
            if path in _FRAMEWORK_PREFIX_ALLOWLIST:
                continue
            if "/tests/" in str(path) or path.parts and "tests" in path.parts:
                continue
            targets.append(path)

    # Non-test framework source files. (Framework production code may
    # use these keys, but the canonical writer is api/agent.py — anyone
    # else is suspicious. We allowlist by full path above.)
    # NOTE: This test focuses on NON-FRAMEWORK callers per US-006 spec
    # ("walk non-framework, non-test .py files"). The framework itself
    # is not scanned to avoid false-positives in code that legitimately
    # consumes (reads) the keys; only canonical *writers* outside the
    # framework are the lockdown target.

    violations: list[tuple[Path, int, str]] = []
    for path in targets:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        visitor = _FrameworkPrefixVisitor()
        visitor.visit(tree)
        for lineno, key in visitor.hits:
            violations.append((path, lineno, key))

    if violations:
        formatted = "\n".join(
            f"  {p.relative_to(_REPO_ROOT)}:{ln}: {key!r}"
            for p, ln, key in violations
        )
        pytest.fail(
            "Reserved `_framework_*` extras keys may only be written by "
            "canonical writers (api/agent.py and framework_orchestrator.py). "
            "Violations:\n" + formatted
        )
