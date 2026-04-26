-- Postgres / Lakebase DDL for the chat-document storage layer.
-- Idempotent: every statement is IF NOT EXISTS so replicas can run `migrate()`
-- on startup concurrently without conflict.
--
-- Schema qualification: every table reference is prefixed with `{ns}.` — a
-- placeholder that `LakebaseBackend.migrate()` substitutes with the configured
-- schema (e.g. `"deep_research_state".`) before execution. This matches the
-- fully-qualified query style used everywhere else in the storage engine
-- (`f"{self._ns}.chat_meta"` in lakebase.py, `${ns}.table` in sql_warehouse.py)
-- and does not rely on server-side `search_path` resolution, which is unsafe
-- to pin at the connection level because the SQLAlchemy engine is shared with
-- the legacy ORM that reads public.*.
--
-- Schema-version tracking lives in `storage_meta(key, value)`. Bump the version
-- in code before applying a new set of statements, and record the version after
-- success so a restart can skip already-applied migrations.

-- === Meta ================================================================

CREATE TABLE IF NOT EXISTS {ns}.storage_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- === Tier A: document tables =============================================

CREATE TABLE IF NOT EXISTS {ns}.chat_meta (
    chat_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    preview TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    version BIGINT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chat_meta_user_updated
    ON {ns}.chat_meta (user_id, updated_at DESC)
    WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_chat_meta_deleted_at
    ON {ns}.chat_meta (deleted_at)
    WHERE deleted_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS {ns}.chat_state (
    chat_id UUID PRIMARY KEY,
    state JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS {ns}.user_documents (
    user_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    state JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS {ns}.prep_job_documents (
    prep_job_id UUID PRIMARY KEY,
    account_id TEXT NOT NULL,
    status TEXT NOT NULL,
    heartbeat TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    state JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_prep_job_status_heartbeat
    ON {ns}.prep_job_documents (status, heartbeat);
CREATE INDEX IF NOT EXISTS idx_prep_job_account
    ON {ns}.prep_job_documents (account_id);

CREATE TABLE IF NOT EXISTS {ns}.incognito_sessions (
    incognito_session_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    state JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_incognito_sessions_user
    ON {ns}.incognito_sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_incognito_sessions_expires
    ON {ns}.incognito_sessions (expires_at)
    WHERE expires_at IS NOT NULL;

-- === Tier B: list tables =================================================

CREATE TABLE IF NOT EXISTS {ns}.prompt_templates (
    template_id UUID PRIMARY KEY,
    owner_id TEXT NOT NULL,
    name TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    visibility TEXT NOT NULL DEFAULT 'private',
    template_type TEXT NOT NULL DEFAULT 'default',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_prompt_templates_owner
    ON {ns}.prompt_templates (owner_id);
CREATE INDEX IF NOT EXISTS idx_prompt_templates_visibility
    ON {ns}.prompt_templates (visibility);

CREATE TABLE IF NOT EXISTS {ns}.custom_agents (
    agent_id UUID PRIMARY KEY,
    owner_id TEXT NOT NULL,
    name TEXT NOT NULL,
    visibility TEXT NOT NULL DEFAULT 'private',
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    steps JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_custom_agents_owner
    ON {ns}.custom_agents (owner_id);

-- === Tier C: append-only tables ==========================================

CREATE TABLE IF NOT EXISTS {ns}.research_events (
    session_id UUID NOT NULL,
    sequence_number BIGINT NOT NULL,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    event JSONB NOT NULL,
    PRIMARY KEY (session_id, sequence_number)
);
CREATE INDEX IF NOT EXISTS idx_research_events_session_ts
    ON {ns}.research_events (session_id, ts);

CREATE TABLE IF NOT EXISTS {ns}.file_chunks (
    file_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (file_id, chunk_index)
);

-- F-FILE: uploaded-file metadata list. The cached `CachedFileUploadService`
-- reads/writes this table via `list_rows` / `upsert_row`. Column names match
-- the row dict produced by `_file_view_to_row` in
-- `services/cached/file_upload.py` — note `metadata_` with trailing underscore
-- (carried through from the legacy ORM attribute name so the cached service's
-- write+read round-trip is symmetric).
CREATE TABLE IF NOT EXISTS {ns}.uploaded_files (
    id UUID PRIMARY KEY,
    owner_id TEXT NOT NULL,
    session_id UUID,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    storage_path TEXT NOT NULL,
    processing_status TEXT NOT NULL DEFAULT 'pending',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    expires_at TIMESTAMPTZ,
    metadata_ JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_uploaded_files_owner
    ON {ns}.uploaded_files (owner_id);
CREATE INDEX IF NOT EXISTS idx_uploaded_files_session
    ON {ns}.uploaded_files (session_id);

-- F-DS: per-user data-source metadata list. Columns match `_view_to_row` in
-- `services/cached/data_source.py`. (The legacy pattern of embedding these
-- into `UserDocument.data_sources` was replaced by a dedicated cold-path
-- table; the legacy legacy doc-embed is only used for read fallback.)
CREATE TABLE IF NOT EXISTS {ns}.user_data_sources (
    id UUID PRIMARY KEY,
    owner_id TEXT NOT NULL,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    endpoint_identifier TEXT,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    visibility TEXT NOT NULL DEFAULT 'private',
    validation_status TEXT NOT NULL DEFAULT 'unknown',
    last_validated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_user_data_sources_owner
    ON {ns}.user_data_sources (owner_id);
CREATE INDEX IF NOT EXISTS idx_user_data_sources_visibility
    ON {ns}.user_data_sources (visibility);

CREATE TABLE IF NOT EXISTS {ns}.message_feedback (
    feedback_id UUID PRIMARY KEY,
    message_id UUID NOT NULL,
    user_id TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    feedback JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_message_feedback_message
    ON {ns}.message_feedback (message_id);
CREATE INDEX IF NOT EXISTS idx_message_feedback_user
    ON {ns}.message_feedback (user_id);

CREATE TABLE IF NOT EXISTS {ns}.audit_log (
    log_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    event JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_ts
    ON {ns}.audit_log (user_id, ts DESC);

CREATE TABLE IF NOT EXISTS {ns}.chat_deleted_files (
    chat_id UUID NOT NULL,
    file_id UUID NOT NULL,
    PRIMARY KEY (chat_id, file_id)
);
CREATE INDEX IF NOT EXISTS idx_chat_deleted_files_chat
    ON {ns}.chat_deleted_files (chat_id);
