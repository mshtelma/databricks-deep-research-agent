#!/usr/bin/env python
"""Preflight check for the custom_agents drop migration (024).

Run this script before applying migration 024 in production to detect data loss.
If custom_agents contains rows, the deploy MUST export them or be blocked.

Exit codes:
  0  - Table is empty (safe to drop) OR rows exported successfully
  2  - Table has rows and no --export-path was given (deploy BLOCKED)
  1  - Unexpected error (connection failure, etc.)

Usage:
    python scripts/preflight_v1_data_check.py
    python scripts/preflight_v1_data_check.py --export-path /tmp/custom_agents.jsonl
    python scripts/preflight_v1_data_check.py --check-only
    python scripts/preflight_v1_data_check.py --connection-string "$DATABASE_URL"
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path


async def count_rows(connection_string: str) -> int:
    """Return the number of rows in custom_agents.

    Args:
        connection_string: asyncpg-compatible PostgreSQL connection string.

    Returns:
        Row count.
    """
    import asyncpg

    conn = await asyncpg.connect(connection_string)
    try:
        result: int = await conn.fetchval("SELECT COUNT(*) FROM custom_agents")
        return result
    finally:
        await conn.close()


async def export_rows(connection_string: str, export_path: Path) -> int:
    """Export all rows from custom_agents to a JSONL file.

    Each line in the output file is a JSON object representing one row.

    Args:
        connection_string: asyncpg-compatible PostgreSQL connection string.
        export_path: Destination file path for JSONL output.

    Returns:
        Number of rows exported.
    """
    import asyncpg

    conn = await asyncpg.connect(connection_string)
    try:
        rows = await conn.fetch("SELECT * FROM custom_agents")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(dict(row)) + "\n")
        return len(rows)
    finally:
        await conn.close()


def resolve_connection_string(override: str | None) -> str:
    """Resolve the database connection string from override or environment.

    Precedence:
        1. --connection-string CLI flag
        2. DATABASE_URL environment variable

    Args:
        override: Explicit connection string from CLI flag, or None.

    Returns:
        Connection string.

    Raises:
        SystemExit: If no connection string can be found.
    """
    import os

    if override:
        return override

    url = os.environ.get("DATABASE_URL")
    if url:
        # asyncpg expects 'postgresql://' not 'postgresql+asyncpg://'
        return url.replace("postgresql+asyncpg://", "postgresql://")

    print(
        "ERROR: no database connection configured. "
        "Set DATABASE_URL or pass --connection-string.",
        file=sys.stderr,
    )
    sys.exit(1)


async def run(
    connection_string: str,
    export_path: Path | None,
    check_only: bool,
) -> int:
    """Core preflight logic.

    Args:
        connection_string: PostgreSQL connection string.
        export_path: Path to write JSONL export, or None.
        check_only: If True, only count rows and report — never export.

    Returns:
        Process exit code (0 = OK, 2 = blocked).
    """
    count = await count_rows(connection_string)

    if count == 0:
        print("OK: custom_agents is empty; safe to drop")
        return 0

    # Table has rows
    if check_only:
        print(
            f"ERROR: custom_agents has {count} row(s); "
            "export with --export-path before dropping",
            file=sys.stderr,
        )
        return 2

    if export_path is None:
        print(
            f"ERROR: custom_agents has {count} row(s); "
            "export with --export-path before dropping",
            file=sys.stderr,
        )
        return 2

    # Export requested
    exported = await export_rows(connection_string, export_path)
    print(
        f"EXPORTED {exported} row(s) to {export_path}; "
        "safe to drop after backup verified"
    )
    return 0


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Preflight check before dropping the custom_agents table (migration 024). "
            "Exits 0 if the table is empty or rows are exported; exits 2 to block deploy."
        )
    )
    parser.add_argument(
        "--connection-string",
        metavar="DSN",
        help=(
            "PostgreSQL connection string (overrides DATABASE_URL env var). "
            "Prefer DATABASE_URL or a secret-manager value over inline literals."
        ),
    )
    parser.add_argument(
        "--export-path",
        metavar="PATH",
        type=Path,
        help="Write existing rows to this file as JSONL before the drop.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Only count and report rows; never export. "
            "Exits 2 if the table is non-empty regardless of --export-path."
        ),
    )
    args = parser.parse_args()

    connection_string = resolve_connection_string(args.connection_string)

    try:
        exit_code = asyncio.run(
            run(
                connection_string=connection_string,
                export_path=args.export_path,
                check_only=args.check_only,
            )
        )
    except Exception as exc:
        print(f"ERROR: unexpected failure: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
