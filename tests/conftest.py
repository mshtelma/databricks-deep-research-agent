"""Pytest configuration for backend tests.

This file is automatically loaded by pytest and sets up:
1. PYTHONPATH to include project root (for `src.*` imports)
2. Loading of .env file for credentials
3. Logging configuration with third-party library suppression
"""

import os
import sys
from pathlib import Path

# Add project root to PYTHONPATH for `src.*` imports
# tests/conftest.py -> tests -> project_root
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load .env file
from dotenv import load_dotenv

_env_file = _project_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

# Configure logging with third-party suppression
# This ensures our custom logging (agent transitions, tool calls, etc.) is visible
# while suppressing verbose DEBUG output from openai, httpx, httpcore, etc.
from src.middleware.logging import setup_logging

_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
setup_logging(log_level=_log_level, log_format="text")
