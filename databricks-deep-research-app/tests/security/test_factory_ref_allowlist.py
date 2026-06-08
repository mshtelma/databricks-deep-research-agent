"""Security tests for factory_ref allow-list enforcement.

These tests verify that:
1. Arbitrary factory_ref strings (e.g. "os.system") are rejected with 400.
2. Dotted module paths are rejected (no dynamic import possible).
3. Unicode injection attempts are rejected.
4. The framework's factory-resolution code contains zero calls to
   importlib.import_module (grep-based static analysis).

These tests do NOT require a database connection and always run.

Run:
    uv run pytest tests/security/test_factory_ref_allowlist.py -v
"""

from __future__ import annotations

import contextlib
import os
import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Must be set before importing `app`
os.environ.setdefault("STORAGE_SERVICE_IMPL", "sqlalchemy_legacy")

from fastapi.testclient import TestClient

from deep_research.core.auth import UserIdentity
from deep_research.db.session import get_db
from deep_research.main import app
from deep_research.middleware.auth import get_current_user_identity

# ---------------------------------------------------------------------------
# Helper: test client with mock DB (no real DB required)
# ---------------------------------------------------------------------------

_TEST_USER = UserIdentity(
    user_id="security-test-user",
    email="security@test.example",
    display_name="Security Test User",
)


@contextlib.contextmanager
def _mock_client() -> Generator[TestClient, None, None]:
    """TestClient with a mock DB session -- factory_ref is rejected before DB."""

    async def _override_db() -> Any:
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result
        yield mock_session

    async def _override_user() -> UserIdentity:
        return _TEST_USER

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user_identity] = _override_user
    try:
        client = TestClient(app, raise_server_exceptions=True)
        yield client
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user_identity, None)


def _post_tool(client: TestClient, factory_ref: str) -> Any:
    return client.post(
        "/api/v1/agent-designer/custom-tools",
        json={
            "name": "sec_test_tool",
            "config_schema": {},
            "factory_ref": factory_ref,
            "visibility": "private",
        },
    )


def _error_payload(resp: Any) -> dict[str, Any]:
    """Extract error payload regardless of middleware wrapping.

    FastAPI returns ``{"detail": {...}}`` but app middleware may transform this
    to ``{"code": "HTTP_ERROR", "message": {...}}``.  Support both shapes.
    """
    body = resp.json()
    return body.get("detail") or body.get("message") or {}


# ---------------------------------------------------------------------------
# 1. Arbitrary string -> 400 factory_ref_not_in_allowlist
# ---------------------------------------------------------------------------


def test_create_with_arbitrary_factory_ref_rejected() -> None:
    """factory_ref='os.system' must be rejected with 400 and correct error_kind."""
    with _mock_client() as client:
        resp = _post_tool(client, "os.system")
    assert resp.status_code == 400, resp.text
    err = _error_payload(resp)
    assert err["error_kind"] == "factory_ref_not_in_allowlist"
    assert err["received"] == "os.system"


# ---------------------------------------------------------------------------
# 2. Dotted path -> 400
# ---------------------------------------------------------------------------


def test_create_with_dotted_path_rejected() -> None:
    """Dotted module paths like 'some.module.factory' must be rejected."""
    with _mock_client() as client:
        resp = _post_tool(client, "some.module.factory")
    assert resp.status_code == 400, resp.text
    assert _error_payload(resp)["error_kind"] == "factory_ref_not_in_allowlist"


# ---------------------------------------------------------------------------
# 3. Unicode injection -> 400
# Use ‮ (RTL override) embedded in the factory_ref string to verify
# the allow-list check rejects it even with homoglyph/direction attacks.
# ---------------------------------------------------------------------------


def test_create_with_unicode_injection_rejected() -> None:
    """Unicode sequences embedded in factory_ref must be rejected."""
    # U+202E RIGHT-TO-LEFT OVERRIDE embedded to test homoglyph attacks
    unicode_factory_ref = "web‮search_v1"
    with _mock_client() as client:
        resp = _post_tool(client, unicode_factory_ref)
    assert resp.status_code == 400, resp.text
    assert _error_payload(resp)["error_kind"] == "factory_ref_not_in_allowlist"


# ---------------------------------------------------------------------------
# 4. resolve_factory() unit tests
# ---------------------------------------------------------------------------


def test_resolve_factory_raises_for_unknown_ref() -> None:
    """resolve_factory() must raise ValueError for any string not in allow-list."""
    from databricks_deep_research.tools.factories import resolve_factory

    with pytest.raises(ValueError, match="factory_ref_not_in_allowlist"):
        resolve_factory("os.system")

    with pytest.raises(ValueError, match="factory_ref_not_in_allowlist"):
        resolve_factory("")

    with pytest.raises(ValueError, match="factory_ref_not_in_allowlist"):
        resolve_factory("importlib.import_module")


def test_resolve_factory_succeeds_for_allowlisted_ref() -> None:
    """resolve_factory() must succeed for all keys in BUILTIN_FACTORIES."""
    from databricks_deep_research.tools.factories import BUILTIN_FACTORIES, resolve_factory

    for key in BUILTIN_FACTORIES:
        factory_cls = resolve_factory(key)
        assert factory_cls is not None


# ---------------------------------------------------------------------------
# 5. Static analysis: no live importlib.import_module in factory __init__.py
# ---------------------------------------------------------------------------


def test_grep_no_importlib_import_module_on_user_input() -> None:
    """The factories __init__.py must not contain live importlib.import_module calls.

    This is a security-critical invariant: factory_ref resolution MUST be a
    pure dict lookup.  Any live (non-comment) invocation of
    importlib.import_module in the resolution path would allow arbitrary code
    execution via a crafted factory_ref value.

    Comment lines (# ...) and docstring text are excluded from the check
    because they are documentation, not executable code.
    """
    framework_factories_dir = Path(__file__).parents[3] / (
        "databricks-deep-research/src/databricks_deep_research/tools/factories"
    )
    assert framework_factories_dir.exists(), (
        f"factories directory not found at {framework_factories_dir}"
    )

    factories_init = framework_factories_dir / "__init__.py"
    assert factories_init.exists(), f"factories __init__.py not found at {factories_init}"

    result = subprocess.run(
        ["grep", "-n", "importlib.import_module", str(factories_init)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # grep exit 1 means no matches -- pass
        return

    # Filter out non-executable lines.  Only lines that contain a live
    # importlib.import_module(...) call in executable code are a violation.
    # grep -n output format: "42:    some code here"
    live_lines = []
    for line in result.stdout.splitlines():
        # Strip leading line-number prefix from grep -n output
        code_part = line.split(":", 1)[1] if ":" in line else line
        stripped = code_part.strip()
        # Skip pure comment lines (first non-whitespace char is #)
        if stripped.startswith("#"):
            continue
        # Skip docstring delimiter / body lines (start with quote chars)
        if stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        if stripped.startswith('"') or stripped.startswith("'"):
            continue
        # Skip lines that contain "import_module" only inside a string
        # (they appear in MUST NOT / docstring prose, not as a call).
        # A real call looks like: importlib.import_module(
        if "importlib.import_module(" not in stripped:
            continue
        live_lines.append(line)

    assert not live_lines, (
        f"Live importlib.import_module call detected in {factories_init} "
        "(security violation -- must use BUILTIN_FACTORIES dict lookup only):\n"
        + "\n".join(live_lines)
    )
