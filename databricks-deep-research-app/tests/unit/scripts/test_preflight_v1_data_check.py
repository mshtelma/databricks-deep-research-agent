"""Unit tests for scripts/preflight_v1_data_check.py.

All database interactions are mocked — no live DB required.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helper: import the script as a module regardless of working directory
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).parent.parent.parent.parent / "scripts" / "preflight_v1_data_check.py"


def _import_script():  # type: ignore[return]
    """Dynamically import the preflight script module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("preflight_v1_data_check", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_mod = _import_script()
run = _mod.run
resolve_connection_string = _mod.resolve_connection_string
export_rows = _mod.export_rows
count_rows = _mod.count_rows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

POSTGRESQL_SCHEME = "postgresql://"
ASYNC_PG_SCHEME = "postgresql+asyncpg://"

FAKE_DSN = f"{POSTGRESQL_SCHEME}user:pass@localhost:5432/testdb"

SAMPLE_ROWS = [
    {"id": 1, "name": "agent-alpha", "config": '{"k": "v"}'},
    {"id": 2, "name": "agent-beta", "config": '{"k2": "v2"}'},
]


def _make_mock_conn(row_count: int = 0, rows: list[dict] | None = None) -> AsyncMock:
    """Return an async mock that behaves like an asyncpg connection."""
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=row_count)
    # asyncpg rows support dict() via Record.__iter__; we use plain dicts in tests
    conn.fetch = AsyncMock(return_value=[MagicMock(**{"__iter__": lambda self: iter(r.items())}) for r in (rows or [])])
    # close() must be awaitable
    conn.close = AsyncMock()
    return conn


# ---------------------------------------------------------------------------
# Test 1: Empty table → exit 0, safe message
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_table_returns_exit_0(capsys: pytest.CaptureFixture[str]) -> None:
    """Empty custom_agents table: exit 0 and safe-to-drop message."""
    conn = _make_mock_conn(row_count=0)
    with patch("asyncpg.connect", AsyncMock(return_value=conn)):
        code = await run(
            connection_string=FAKE_DSN,
            export_path=None,
            check_only=False,
        )

    assert code == 0
    captured = capsys.readouterr()
    assert "safe to drop" in captured.out
    assert captured.err == ""


# ---------------------------------------------------------------------------
# Test 2: Non-empty + --export-path → writes JSONL, exit 0
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nonempty_with_export_path_writes_jsonl_and_exits_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-empty table + --export-path: exports rows to JSONL and exits 0."""
    export_file = tmp_path / "export.jsonl"

    # Mock count query
    count_conn = _make_mock_conn(row_count=2)
    # Mock fetch query — records behave as dicts
    fetch_conn = AsyncMock()
    fetch_conn.fetch = AsyncMock(return_value=[dict(r) for r in SAMPLE_ROWS])
    fetch_conn.close = AsyncMock()

    call_count = 0

    async def fake_connect(dsn: str) -> AsyncMock:
        nonlocal call_count
        call_count += 1
        return count_conn if call_count == 1 else fetch_conn

    with patch("asyncpg.connect", side_effect=fake_connect):
        code = await run(
            connection_string=FAKE_DSN,
            export_path=export_file,
            check_only=False,
        )

    assert code == 0
    captured = capsys.readouterr()
    assert "EXPORTED" in captured.out
    assert str(export_file) in captured.out

    # Verify JSONL content
    lines = export_file.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["id"] == 1
    assert first["name"] == "agent-alpha"


# ---------------------------------------------------------------------------
# Test 3: Non-empty + no --export-path → exit 2, stderr BLOCK message
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nonempty_no_export_path_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    """Non-empty table with no --export-path: exits 2 and prints block message to stderr."""
    conn = _make_mock_conn(row_count=5)
    with patch("asyncpg.connect", AsyncMock(return_value=conn)):
        code = await run(
            connection_string=FAKE_DSN,
            export_path=None,
            check_only=False,
        )

    assert code == 2
    captured = capsys.readouterr()
    assert "ERROR" in captured.err
    assert "5" in captured.err
    assert captured.out == ""


# ---------------------------------------------------------------------------
# Test 4: --check-only with non-empty table → exits 2 (no export)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_only_nonempty_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--check-only with non-empty table: exits 2, does NOT write any file."""
    export_file = tmp_path / "should_not_exist.jsonl"

    conn = _make_mock_conn(row_count=3)
    with patch("asyncpg.connect", AsyncMock(return_value=conn)):
        code = await run(
            connection_string=FAKE_DSN,
            export_path=export_file,  # even though a path is given, check_only wins
            check_only=True,
        )

    assert code == 2
    assert not export_file.exists(), "check-only must not write any export file"
    captured = capsys.readouterr()
    assert "ERROR" in captured.err


# ---------------------------------------------------------------------------
# Test 5: --check-only with empty table → exits 0
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_only_empty_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    """--check-only with empty table: exits 0 with safe message."""
    conn = _make_mock_conn(row_count=0)
    with patch("asyncpg.connect", AsyncMock(return_value=conn)):
        code = await run(
            connection_string=FAKE_DSN,
            export_path=None,
            check_only=True,
        )

    assert code == 0
    captured = capsys.readouterr()
    assert "safe to drop" in captured.out


# ---------------------------------------------------------------------------
# Test 6: JSONL roundtrip — write then read returns same data shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_jsonl_roundtrip(tmp_path: Path) -> None:
    """Export rows to JSONL and re-read them; data shape is preserved."""
    export_file = tmp_path / "roundtrip.jsonl"

    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[dict(r) for r in SAMPLE_ROWS])
    conn.close = AsyncMock()

    with patch("asyncpg.connect", AsyncMock(return_value=conn)):
        n = await export_rows(FAKE_DSN, export_file)

    assert n == 2
    lines = export_file.read_text().splitlines()
    assert len(lines) == n

    for i, line in enumerate(lines):
        parsed = json.loads(line)
        assert parsed["id"] == SAMPLE_ROWS[i]["id"]
        assert parsed["name"] == SAMPLE_ROWS[i]["name"]
        assert parsed["config"] == SAMPLE_ROWS[i]["config"]


# ---------------------------------------------------------------------------
# Test 7: resolve_connection_string prefers CLI override over env var
# ---------------------------------------------------------------------------

def test_resolve_connection_string_cli_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """--connection-string takes precedence over DATABASE_URL."""
    env_dsn = f"{POSTGRESQL_SCHEME}env-host/envdb"
    cli_dsn = f"{POSTGRESQL_SCHEME}cli-host/clidb"
    monkeypatch.setenv("DATABASE_URL", env_dsn)
    result = resolve_connection_string(cli_dsn)
    assert result == cli_dsn


# ---------------------------------------------------------------------------
# Test 8: resolve_connection_string normalises asyncpg scheme
# ---------------------------------------------------------------------------

def test_resolve_connection_string_strips_asyncpg_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    """DATABASE_URL with +asyncpg driver is normalised for asyncpg.connect()."""
    monkeypatch.setenv("DATABASE_URL", f"{ASYNC_PG_SCHEME}u:p@host/db")
    result = resolve_connection_string(None)
    assert result == f"{POSTGRESQL_SCHEME}u:p@host/db"
