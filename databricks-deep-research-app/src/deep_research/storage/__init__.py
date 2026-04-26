"""Storage subsystem: backend-agnostic persistence for chat state.

This package implements the chat-document storage architecture described in
plan `i-cannot-use-lakebase-frolicking-music.md`. It provides:

- `StorageBackend` — Protocol that backends implement.
- `ChatDocument` / `ChatState` / `UserDocument` / `PrepJobDocument` — pydantic
  models that define the on-wire JSON shape identically across backends.
- `ChatStateCache` + `Hydrator` — per-chat in-memory snapshot populated
  asynchronously, so hot-path requests never wait for storage.
- `WriteQueue` — coalescing, batched, fire-and-forget writer that funnels every
  mutation through `StorageBackend.write_*`.
- `ColdReadCache` — short-TTL memoization for cold-path list queries.
- Two permanent first-class backend implementations (added in later phases):
  `LakebaseBackend` (Postgres JSONB) and `SQLWarehouseBackend` (Delta STRING +
  MERGE). A `FakeBackend` lives under `tests/fakes/` for unit tests.

Principles enforced by this package (see plan section "Principles"):
1. Async on every code path; user never blocks on warm storage work.
2. In-memory state is the runtime source of truth; persistence is cache-behind.
3. One statement per basic runtime operation.
4. Two first-class backends sharing one schema and one runtime.
5. Data loss on crash is bounded by `flush_interval_sec` and measured.
"""

from deep_research.storage.backend import (
    BackendError,
    BatchResult,
    ConflictError,
    MigrationInProgressError,
    PermanentError,
    SchemaError,
    StorageBackend,
    TransientError,
    WriteOutcome,
)
from deep_research.storage.documents import (
    ChatDocument,
    ChatMeta,
    ChatState,
    Coverage,
    Entity,
    FileMemo,
    Finding,
    Memory,
    Message,
    PluginExtEntry,
    PrepJobDocument,
    ResearchSessionState,
    Source,
    UploadedFileMeta,
    UserDocument,
)

__all__ = [
    # Protocol + errors
    "StorageBackend",
    "BackendError",
    "TransientError",
    "PermanentError",
    "SchemaError",
    "ConflictError",
    "MigrationInProgressError",
    "BatchResult",
    "WriteOutcome",
    # Documents
    "ChatDocument",
    "ChatMeta",
    "ChatState",
    "Memory",
    "Finding",
    "Entity",
    "Coverage",
    "FileMemo",
    "PluginExtEntry",
    "Source",
    "ResearchSessionState",
    "UploadedFileMeta",
    "Message",
    "UserDocument",
    "PrepJobDocument",
]
