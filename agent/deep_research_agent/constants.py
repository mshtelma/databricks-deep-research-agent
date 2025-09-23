"""
Central configuration constants for the Deep Research Agent.

This module provides a single source of truth for all configuration file names,
paths, and deployment-related constants used throughout the agent system.
"""

# Agent configuration file names
AGENT_CONFIG_FILENAME = "agent_config.yaml"
AGENT_CONFIG_BACKUP = "agent_config_single.yaml.backup"  # Backup of old single-agent config

# Deployment paths
AGENT_CONFIG_PATH = f"deep_research_agent/{AGENT_CONFIG_FILENAME}"

# Alternative config filename for environment override
# Can be overridden via environment variable: AGENT_CONFIG_OVERRIDE
AGENT_CONFIG_OVERRIDE_ENV = "AGENT_CONFIG_OVERRIDE"

# Default config search order (NEW: prioritize conf/ directory)
CONFIG_SEARCH_ORDER = [
    "conf/base.yaml",                # NEW: Primary config location
    AGENT_CONFIG_FILENAME,           # OLD: agent_config.yaml (backward compat)
    "agent_config_enhanced.yaml",    # OLD: Fallback enhanced config
    "agent_config_single.yaml",      # OLD: Fallback single-agent config
    # Note: .backup files intentionally excluded from search
]