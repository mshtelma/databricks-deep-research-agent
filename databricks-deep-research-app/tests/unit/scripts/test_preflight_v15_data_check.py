"""Unit tests for scripts/preflight_v15_data_check.py.

All database interactions are mocked — no live DB required.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helper: import the script as a module regardless of working directory
# ---------------------------------------------------------------------------
SCRIPT_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "scripts"
    / "preflight_v15_data_check.py"
)


def _import_script():  # type: ignore[return]
    """Dynamically import the preflight V1.5 script module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "preflight_v15_data_check", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_mod = _import_script()
run = _mod.run
resolve_connection_string = _mod.resolve_connection_string
count_custom_agent_id_columns = _mod.count_custom_agent_id_columns
get_alembic_version = _mod.get_alembic_version
ACCEPTABLE_HEAD_VERSIONS: list[str] = _mod._ACCEPTABLE_HEAD_VERSIONS

# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------

FAKE_DSN = "postgresql://user:pass@localhost:5432/testdb"
GOOD_VERSION = "024_drop_custom_agents"


def _make_mock_conn(col_count: int = 0, version: str | None = GOOD_VERSION) -> AsyncMock:
    """Return an async mock behaving like an asyncpg connection.

    ``fetchval`` is called twice in sequence:
      first  → col_count (information_schema check)
      second → version   (alembic_version check)
    """
    conn = AsyncMock()
    conn.fetchval = AsyncMock(side_effect=[col_count, version])
    conn.close = AsyncMock()
    return conn


# ---------------------------------------------------------------------------
# Test 1: zero refs AND 024 at head → exit 0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_refs_and_024_head_returns_0(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both checks pass: exits 0 with an OK message."""
    conn = _make_mock_conn(col_count=0, version=GOOD_VERSION)
    with patch("asyncpg.connect", AsyncMock(return_value=conn)):
        code = await run(connection_string=FAKE_DSN, check_only=False)

    assert code == 0
    captured = capsys.readouterr()
    assert "OK" in captured.out
    assert "safe to apply" in captured.out
    assert captured.err == ""


# ---------------------------------------------------------------------------
# Test 2: column still exists → exit 2
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nonzero_refs_returns_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """custom_agent_id column still present: exits 2 with a block message."""
    # Only the first fetchval (col_count) is called before early return
    conn = AsyncMock()
    conn.fetchval = AsyncMock(side_effect=[3])  # col_count = 3
    conn.close = AsyncMock()

    with patch("asyncpg.connect", AsyncMock(return_value=conn)):
        code = await run(connection_string=FAKE_DSN, check_only=False)

    assert code == 2
    captured = capsys.readouterr()
    assert "ERROR" in captured.err
    assert "3" in captured.err
    assert "custom_agent_id" in captured.err
    assert captured.out == ""


# ---------------------------------------------------------------------------
# Test 3: 024 NOT at head → exit 2
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_024_not_at_head_returns_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """alembic_version is older than 024: exits 2 with a block message."""
    conn = _make_mock_conn(col_count=0, version="023_add_indexes")

    with patch("asyncpg.connect", AsyncMock(return_value=conn)):
        code = await run(connection_string=FAKE_DSN, check_only=False)

    assert code == 2
    captured = capsys.readouterr()
    assert "ERROR" in captured.err
    assert "023_add_indexes" in captured.err
    assert captured.out == ""


# ---------------------------------------------------------------------------
# Test 4: --check-only does NOT export (V1.5 has no export step; flag is a no-op)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_only_does_not_export(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--check-only flag accepted; no file is written (V1.5 has no export step)."""
    sentinel_file = tmp_path / "should_not_exist.jsonl"

    conn = _make_mock_conn(col_count=0, version=GOOD_VERSION)
    with patch("asyncpg.connect", AsyncMock(return_value=conn)):
        code = await run(connection_string=FAKE_DSN, check_only=True)

    assert code == 0
    assert not sentinel_file.exists(), "check_only must not create any file"
    captured = capsys.readouterr()
    assert "OK" in captured.out


# ---------------------------------------------------------------------------
# Test 5: later acceptable head versions (025, 026) also pass
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "version",
    ["025_create_agent_revisions", "026_create_custom_tool_defs"],
)
async def test_later_acceptable_versions_pass(
    version: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Later HEAD versions (025, 026) are acceptable and return exit 0."""
    conn = _make_mock_conn(col_count=0, version=version)
    with patch("asyncpg.connect", AsyncMock(return_value=conn)):
        code = await run(connection_string=FAKE_DSN, check_only=False)

    assert code == 0
    captured = capsys.readouterr()
    assert "OK" in captured.out


# ---------------------------------------------------------------------------
# Test 6: resolve_connection_string prefers CLI override
# ---------------------------------------------------------------------------


def test_resolve_connection_string_cli_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--connection-string takes precedence over DATABASE_URL."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://env-host/envdb")
    result = resolve_connection_string("postgresql://cli-host/clidb")
    assert result == "postgresql://cli-host/clidb"


# ---------------------------------------------------------------------------
# Test 7: resolve_connection_string normalises asyncpg scheme
# ---------------------------------------------------------------------------


def test_resolve_connection_string_strips_asyncpg_scheme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DATABASE_URL with +asyncpg driver is normalised for asyncpg.connect()."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@host/db")
    result = resolve_connection_string(None)
    assert result == "postgresql://u:p@host/db"
