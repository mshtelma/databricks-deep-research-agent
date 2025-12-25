#!/usr/bin/env python3
"""Test Lakebase connection with hardcoded values."""

import asyncio
import base64
import json
import ssl

import asyncpg
from databricks.sdk import WorkspaceClient

# Hardcoded configuration
DATABRICKS_PROFILE = "ais"
LAKEBASE_INSTANCE_NAME = "msh-deep-research"
LAKEBASE_DATABASE = "databricks_postgres"
LAKEBASE_PORT = 5432


def get_lakebase_credentials() -> tuple[str, str, str]:
    """Get OAuth credentials for Lakebase.

    Returns:
        Tuple of (host, username, token)
    """
    print(f"Creating WorkspaceClient with profile: {DATABRICKS_PROFILE}")
    client = WorkspaceClient(profile=DATABRICKS_PROFILE)

    # Get actual hostname from instance metadata (NOT {name}.database.cloud.databricks.com)
    print(f"Looking up instance: {LAKEBASE_INSTANCE_NAME}")
    host = None
    for inst in client.database.list_database_instances():
        if inst.name == LAKEBASE_INSTANCE_NAME:
            host = inst.read_write_dns
            print(f"Found instance: {inst.name} -> {host}")
            break

    if not host:
        raise RuntimeError(f"Instance '{LAKEBASE_INSTANCE_NAME}' not found")

    print(f"Generating database credential for instance: {LAKEBASE_INSTANCE_NAME}")
    cred_response = client.database.generate_database_credential(
        instance_names=[LAKEBASE_INSTANCE_NAME],
    )

    if not cred_response.token:
        raise RuntimeError("No token returned from Databricks")

    token = cred_response.token

    # Extract username from JWT 'sub' claim (Databricks identity - email)
    payload_b64 = token.split(".")[1]
    payload_b64 += "=" * (4 - len(payload_b64) % 4)
    payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    username = payload["sub"]

    print(f"Got credential, expires: {cred_response.expiration_time}")
    print(f"Username (from JWT sub): {username}")
    return host, username, token


async def test_connection():
    """Test the Lakebase connection."""
    print("=" * 60)
    print("Testing Lakebase Connection")
    print("=" * 60)

    # Step 1: Get credentials
    print("\n[1] Getting OAuth credentials...")
    try:
        host, username, token = get_lakebase_credentials()
        print(f"    Host: {host}")
        print(f"    Username: {username}")
        print(f"    Token: {token[:20]}...{token[-10:]}")
    except Exception as e:
        print(f"    FAILED: {e}")
        return

    # Step 2: Test DNS resolution
    print(f"\n[2] Testing DNS resolution for {host}...")
    import socket
    try:
        ip = socket.gethostbyname(host)
        print(f"    Resolved to: {ip}")
    except socket.gaierror as e:
        print(f"    FAILED: {e}")
        print(f"    The instance '{LAKEBASE_INSTANCE_NAME}' may not exist.")
        print("    Check your Lakebase instance name in the Databricks console.")
        return

    # Step 3: Connect with asyncpg
    print(f"\n[3] Connecting to PostgreSQL at {host}:{LAKEBASE_PORT}/{LAKEBASE_DATABASE}...")
    try:
        # Create SSL context
        ssl_context = ssl.create_default_context()

        conn = await asyncpg.connect(
            host=host,
            port=LAKEBASE_PORT,
            user=username,
            password=token,
            database=LAKEBASE_DATABASE,
            ssl=ssl_context,
        )
        print("    Connected successfully!")

        # Test query
        print("\n[4] Running test query...")
        version = await conn.fetchval("SELECT version()")
        print(f"    PostgreSQL version: {version[:60]}...")

        # List tables
        print("\n[5] Listing tables...")
        tables = await conn.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            LIMIT 10
        """)
        if tables:
            for t in tables:
                print(f"    - {t['table_name']}")
        else:
            print("    (no tables found)")

        await conn.close()
        print("\n" + "=" * 60)
        print("SUCCESS: Lakebase connection working!")
        print("=" * 60)

    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(test_connection())
