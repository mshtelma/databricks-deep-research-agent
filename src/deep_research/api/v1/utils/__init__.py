"""Shared utilities for API v1 endpoints."""

from deep_research.api.v1.utils.authorization import (
    verify_chat_access,
    verify_chat_ownership,
    verify_message_ownership,
)
from deep_research.api.v1.utils.transformers import (
    build_citation_response,
    build_empty_verification_summary,
    build_evidence_span_response,
    build_numeric_detail,
    build_source_metadata,
    build_verification_summary,
    claim_to_response,
)

__all__ = [
    # Authorization
    "verify_chat_access",
    "verify_chat_ownership",
    "verify_message_ownership",
    # Transformers
    "build_citation_response",
    "build_empty_verification_summary",
    "build_evidence_span_response",
    "build_numeric_detail",
    "build_source_metadata",
    "build_verification_summary",
    "claim_to_response",
]
