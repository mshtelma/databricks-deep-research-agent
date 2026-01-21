"""Shared utilities for API v1 endpoints.

JSONB Migration (Migration 011):
Legacy transformer functions (claim_to_response, build_citation_response, etc.)
have been removed. Use the JSONB transformers instead.
"""

from deep_research.api.v1.utils.authorization import (
    verify_chat_access,
    verify_chat_ownership,
    verify_message_ownership,
)
from deep_research.api.v1.utils.transformers import (
    build_empty_verification_summary,
    generate_claim_uuid,
    jsonb_claim_to_response,
    jsonb_summary_to_response,
)

__all__ = [
    # Authorization
    "verify_chat_access",
    "verify_chat_ownership",
    "verify_message_ownership",
    # JSONB transformers
    "build_empty_verification_summary",
    "generate_claim_uuid",
    "jsonb_claim_to_response",
    "jsonb_summary_to_response",
]
