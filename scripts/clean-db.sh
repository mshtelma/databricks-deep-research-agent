#!/bin/bash
# Clean all chats, messages, and research data from the database
# Usage: ./scripts/clean-db.sh (run from project root with uv run)
#
# This script deletes all research-related data while preserving:
# - User preferences
# - Audit logs (optional - add to deletion if needed)
#
# Requires: LAKEBASE_INSTANCE_NAME and DATABRICKS_CONFIG_PROFILE, or DATABASE_URL

set -e

echo "Cleaning database..."

# Run cleanup via Python using project's Lakebase auth
python3 << 'EOF'
import asyncio
import os
import ssl
import sys

# Add project root to path for imports
sys.path.insert(0, os.getcwd())

async def clean_db():
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text
        from src.core.config import get_settings
        from src.db.lakebase_auth import LakebaseCredentialProvider
    except ImportError as e:
        print(f"Error: {e}")
        print("Run from project root with: uv run ./scripts/clean-db.sh")
        sys.exit(1)

    settings = get_settings()

    # Use Lakebase if configured, otherwise fall back to DATABASE_URL
    if settings.use_lakebase:
        print(f"Using Lakebase instance: {settings.lakebase_instance_name}")
        provider = LakebaseCredentialProvider(settings)
        url = provider.build_connection_url()

        # Lakebase requires SSL
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        connect_args = {"ssl": ssl_context}
    elif settings.database_url:
        print(f"Using local database: {settings.database_url.host}")
        url = str(settings.database_url)
        connect_args = {}
    else:
        print("Error: Neither LAKEBASE_INSTANCE_NAME nor DATABASE_URL is configured")
        print("Set LAKEBASE_INSTANCE_NAME + DATABRICKS_CONFIG_PROFILE for Lakebase")
        print("Or set DATABASE_URL for local PostgreSQL")
        sys.exit(1)

    engine = create_async_engine(url, connect_args=connect_args)

    async with engine.begin() as conn:
        # Delete in order to respect foreign keys:
        # 1. sources (depends on research_sessions)
        # 2. research_sessions (depends on messages)
        # 3. message_feedback (depends on messages)
        # 4. messages (depends on chats)
        # 5. chats (no dependencies)

        result = await conn.execute(text("DELETE FROM sources"))
        print(f"  Deleted {result.rowcount} sources")

        result = await conn.execute(text("DELETE FROM research_sessions"))
        print(f"  Deleted {result.rowcount} research sessions")

        result = await conn.execute(text("DELETE FROM message_feedback"))
        print(f"  Deleted {result.rowcount} message feedback entries")

        result = await conn.execute(text("DELETE FROM messages"))
        print(f"  Deleted {result.rowcount} messages")

        result = await conn.execute(text("DELETE FROM chats"))
        print(f"  Deleted {result.rowcount} chats")

    await engine.dispose()
    print("\nDatabase cleaned successfully!")

asyncio.run(clean_db())
EOF
