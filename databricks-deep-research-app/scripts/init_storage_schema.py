"""Apply the chat-document storage DDL against the configured backend.

Idempotent (all CREATE IF NOT EXISTS). Reads `Settings.storage_backend` to
pick the right backend implementation, then calls `backend.migrate()` which
reads and applies the packaged DDL SQL file.

Usage (via Makefile):
    make db-reset                    # drop + alembic + this script (local)
    make db-reset TARGET=ais         # same against remote Lakebase
    make db-migrate-remote TARGET=ais  # alembic + this script (no drop)

Direct invocation (after sourcing the appropriate ../.env.<target>):
    uv run python -m scripts.init_storage_schema
"""

from __future__ import annotations

import asyncio
import logging
import sys

logger = logging.getLogger(__name__)


async def main() -> int:
    from deep_research.core.config import get_settings
    from deep_research.storage.factory import create_backend

    settings = get_settings()
    logger.info(
        "STORAGE_INIT backend=%s service_impl=%s",
        settings.storage_backend, settings.storage_service_impl,
    )

    backend = create_backend(settings)
    try:
        logger.info("APPLYING DDL…")
        await backend.migrate()
        logger.info("DDL_APPLIED_OK")

        # Sanity read — proves the chat_meta table is usable.
        metas = await backend.list_chat_metas("__nobody__", limit=1)
        logger.info("SANITY_CHECK list_chat_metas_rows=%d", len(metas))
        return 0
    finally:
        try:
            await backend.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    sys.exit(asyncio.run(main()))
