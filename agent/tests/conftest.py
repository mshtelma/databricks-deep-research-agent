"""
Pytest configuration for all tests.

This conftest.py loads environment variables from .env.local before running tests,
ensuring that BRAVE_API_KEY and other secrets are available for real end-to-end tests.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def pytest_configure(config):
    """
    Load environment variables before test collection.

    Priority order:
    1. .env.local (highest priority - local development secrets)
    2. .env (fallback - template/defaults)
    3. System environment (lowest priority)
    """
    # Find the agent root directory (where .env.local should be)
    agent_root = Path(__file__).parent.parent  # tests/ -> agent/

    # Load .env.local if it exists (highest priority)
    env_local_path = agent_root / ".env.local"
    if env_local_path.exists():
        load_dotenv(env_local_path, override=True)
        print(f"✅ Loaded environment from: {env_local_path}")
    else:
        print(f"⚠️  No .env.local found at: {env_local_path}")

    # Load .env as fallback
    env_path = agent_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)  # Don't override .env.local values
        print(f"✅ Loaded environment from: {env_path}")

    # Verify critical environment variables
    if os.getenv("BRAVE_API_KEY"):
        print(f"✅ BRAVE_API_KEY is set (length: {len(os.getenv('BRAVE_API_KEY'))})")
    else:
        print("⚠️  BRAVE_API_KEY is not set - real end-to-end tests will be skipped")

    # Add src/ to Python path for imports
    src_path = agent_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"✅ Added to Python path: {src_path}")
