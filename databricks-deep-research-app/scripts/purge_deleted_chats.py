#!/usr/bin/env python
"""Background job to permanently purge soft-deleted chats.

This script should be run periodically (e.g., daily via cron) to permanently
delete chats that were soft-deleted more than 30 days ago.

Usage:
    python -m scripts.purge_deleted_chats [--days 30]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.session import async_session_factory
from src.services.chat_service import ChatService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def purge_chats(days_old: int = 30) -> int:
    """Purge soft-deleted chats older than specified days.

    Args:
        days_old: Number of days after soft-delete before permanent purge.

    Returns:
        Number of chats purged.
    """
    async with async_session_factory() as session:
        service = ChatService(session)
        purged_count = await service.purge_deleted_chats(days_old=days_old)
        await session.commit()
        return purged_count


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Permanently purge soft-deleted chats"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days after soft-delete before permanent purge (default: 30)",
    )
    args = parser.parse_args()

    logger.info(f"Starting chat purge job (days_old={args.days})")

    try:
        purged_count = asyncio.run(purge_chats(days_old=args.days))
        logger.info(f"Chat purge job completed. Purged {purged_count} chats.")
    except Exception as e:
        logger.exception(f"Chat purge job failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
