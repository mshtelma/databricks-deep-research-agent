"""Search services module."""

from deep_research.services.search.domain_filter import (
    DomainFilter,
    DomainMatchResult,
    extract_domain,
    match_domain_pattern,
)

__all__ = [
    "DomainFilter",
    "DomainMatchResult",
    "extract_domain",
    "match_domain_pattern",
]
