"""Framework-side memory module.

This module holds Pydantic projections, configuration models, and
universal helpers (spotlighting, system-prompt injection) for the
chat-scoped research memory layer. The persistent storage (SQLAlchemy
models, hydration, upserts, LLM-driven consolidation) lives in the app
repo's ``deep_research/services/chat_memory_service.py`` — the framework
stays DB-agnostic.

Agents interact with memory through:
1. A per-role rendered markdown projection placed in ``WorkflowState``
   under the reserved key ``__chat_memory_appendix`` (see
   ``inject_attached_context_block``).
2. Framework tools (``list_attached_files``, ``read_attached_file``,
   ``get_file_entities`` in PR 1; ``memory_search_findings``,
   ``memory_open_questions`` in PR 2) registered by the app
   orchestrator.
"""

from databricks_deep_research.memory.extraction_schema import (
    ExtractedEntity,
    ExtractedFact,
    FileExtraction,
)
from databricks_deep_research.memory.injection import (
    CHAT_MEMORY_APPENDIX_STATE_KEY,
    inject_attached_context_block,
)
from databricks_deep_research.memory.llm_extractor import (
    DEFAULT_HEAD_CHARS,
    extract_file_content,
)
from databricks_deep_research.memory.models import (
    ChatMemorySnapshot,
    CoverageEntry,
    EntityRecord,
    FileRef,
    KnowledgeFinding,
    MemoryConfig,
    RateLimitConfig,
)
from databricks_deep_research.memory.spotlighting import (
    DEFAULT_SPOTLIGHTING_MODE,
    SpotlightingMode,
    strip_datamark,
    wrap_attached_context,
)

__all__ = [
    "CHAT_MEMORY_APPENDIX_STATE_KEY",
    "ChatMemorySnapshot",
    "CoverageEntry",
    "DEFAULT_HEAD_CHARS",
    "DEFAULT_SPOTLIGHTING_MODE",
    "EntityRecord",
    "ExtractedEntity",
    "ExtractedFact",
    "FileExtraction",
    "FileRef",
    "KnowledgeFinding",
    "MemoryConfig",
    "RateLimitConfig",
    "SpotlightingMode",
    "extract_file_content",
    "inject_attached_context_block",
    "strip_datamark",
    "wrap_attached_context",
]
