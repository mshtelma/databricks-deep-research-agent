"""Execute a .sql file against the configured Lakebase via the app's engine.

Reuses the app's credential provider / session maker, so it honours the same
ENDPOINT_NAME / LAKEBASE_INSTANCE_NAME / DATABRICKS_CONFIG_PROFILE env detection
as the migration targets. Intended for one-off maintenance SQL (e.g.
``scripts/cleanup_legacy_tables.sql``) where ``psql`` is unavailable locally.

Usage:
    uv run python scripts/run_sql_file.py <path-to.sql>

The file is executed as a single statement (suitable for a ``DO $$ ... $$``
block). A server-side ``RAISE EXCEPTION`` (e.g. the cleanup script's
non-empty-table guard) aborts with a non-zero exit and no commit.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import text

from deep_research.core.config import get_settings
from deep_research.db.session import get_session_maker


async def _run(sql_path: Path) -> None:
    sql = sql_path.read_text(encoding="utf-8")
    session_maker = get_session_maker(get_settings())
    async with session_maker() as session:
        await session.execute(text(sql))
        await session.commit()
    print(f"run_sql_file: executed {sql_path}")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/run_sql_file.py <path-to.sql>", file=sys.stderr)
        return 2
    sql_path = Path(sys.argv[1])
    if not sql_path.is_file():
        print(f"run_sql_file: file not found: {sql_path}", file=sys.stderr)
        return 2
    asyncio.run(_run(sql_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
