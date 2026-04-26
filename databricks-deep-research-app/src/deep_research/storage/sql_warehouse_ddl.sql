-- Databricks SQL / Delta DDL for the chat-document storage layer.
--
-- Runs against a SQL Warehouse via the StatementExecution API. Every table
-- is managed (no LOCATION clause) and uses liquid clustering for partition
-- pruning on the hot access path.
--
-- Template placeholders: ${ns} is the fully qualified schema name, e.g.
-- `main.presales_state`. The backend substitutes this before sending the
-- statement — no user-controlled data is interpolated.

CREATE SCHEMA IF NOT EXISTS ${ns};

-- === Meta ================================================================

CREATE TABLE IF NOT EXISTS ${ns}.storage_meta (
    key STRING NOT NULL,
    value STRING NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    CONSTRAINT storage_meta_pk PRIMARY KEY (key) RELY
) USING DELTA;

-- === Tier A: document tables =============================================

CREATE TABLE IF NOT EXISTS ${ns}.chat_meta (
    chat_id STRING NOT NULL,
    user_id STRING NOT NULL,
    title STRING NOT NULL,
    preview STRING NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    deleted_at TIMESTAMP,
    version BIGINT NOT NULL,
    CONSTRAINT chat_meta_pk PRIMARY KEY (chat_id) RELY
) USING DELTA CLUSTER BY (user_id, chat_id);

CREATE TABLE IF NOT EXISTS ${ns}.chat_state (
    chat_id STRING NOT NULL,
    state STRING NOT NULL,
    CONSTRAINT chat_state_pk PRIMARY KEY (chat_id) RELY
) USING DELTA CLUSTER BY (chat_id);

CREATE TABLE IF NOT EXISTS ${ns}.user_documents (
    user_id STRING NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    state STRING NOT NULL,
    CONSTRAINT user_documents_pk PRIMARY KEY (user_id) RELY
) USING DELTA CLUSTER BY (user_id);

CREATE TABLE IF NOT EXISTS ${ns}.prep_job_documents (
    prep_job_id STRING NOT NULL,
    account_id STRING NOT NULL,
    status STRING NOT NULL,
    heartbeat TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    state STRING NOT NULL,
    CONSTRAINT prep_job_documents_pk PRIMARY KEY (prep_job_id) RELY
) USING DELTA CLUSTER BY (prep_job_id);

CREATE TABLE IF NOT EXISTS ${ns}.incognito_sessions (
    incognito_session_id STRING NOT NULL,
    user_id STRING NOT NULL,
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP,
    state STRING NOT NULL,
    CONSTRAINT incognito_sessions_pk PRIMARY KEY (incognito_session_id) RELY
) USING DELTA CLUSTER BY (user_id);

-- === Tier B: list tables =================================================

CREATE TABLE IF NOT EXISTS ${ns}.prompt_templates (
    template_id STRING NOT NULL,
    owner_id STRING NOT NULL,
    name STRING NOT NULL,
    content STRING NOT NULL,
    visibility STRING NOT NULL,
    template_type STRING NOT NULL,
    metadata STRING NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    CONSTRAINT prompt_templates_pk PRIMARY KEY (template_id) RELY
) USING DELTA CLUSTER BY (owner_id);

CREATE TABLE IF NOT EXISTS ${ns}.custom_agents (
    agent_id STRING NOT NULL,
    owner_id STRING NOT NULL,
    name STRING NOT NULL,
    visibility STRING NOT NULL,
    config STRING NOT NULL,
    steps STRING NOT NULL,
    metadata STRING NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    CONSTRAINT custom_agents_pk PRIMARY KEY (agent_id) RELY
) USING DELTA CLUSTER BY (owner_id);

-- === Tier C: append-only tables ==========================================

CREATE TABLE IF NOT EXISTS ${ns}.research_events (
    session_id STRING NOT NULL,
    sequence_number BIGINT NOT NULL,
    ts TIMESTAMP NOT NULL,
    event STRING NOT NULL
) USING DELTA CLUSTER BY (session_id);

CREATE TABLE IF NOT EXISTS ${ns}.file_chunks (
    file_id STRING NOT NULL,
    chunk_index INT NOT NULL,
    ts TIMESTAMP NOT NULL,
    content STRING NOT NULL,
    metadata STRING NOT NULL
) USING DELTA CLUSTER BY (file_id);

-- F-FILE: uploaded-file metadata list. Mirrors the Lakebase `uploaded_files`
-- schema; column names match the row dict produced by `_file_view_to_row` in
-- `services/cached/file_upload.py` (note `metadata_` with trailing
-- underscore — legacy ORM attribute name carried through as the row key).
CREATE TABLE IF NOT EXISTS ${ns}.uploaded_files (
    id STRING NOT NULL,
    owner_id STRING NOT NULL,
    session_id STRING,
    filename STRING NOT NULL,
    file_type STRING NOT NULL,
    file_size INT NOT NULL,
    storage_path STRING NOT NULL,
    processing_status STRING NOT NULL,
    chunk_count INT NOT NULL,
    expires_at TIMESTAMP,
    metadata_ STRING NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
) USING DELTA CLUSTER BY (owner_id);

-- F-DS: per-user data-source metadata list. Mirrors the Lakebase
-- `user_data_sources` schema; column names match the row dict produced by
-- `_view_to_row` in `services/cached/data_source.py`.
CREATE TABLE IF NOT EXISTS ${ns}.user_data_sources (
    id STRING NOT NULL,
    owner_id STRING NOT NULL,
    type STRING NOT NULL,
    name STRING NOT NULL,
    description STRING,
    endpoint_identifier STRING,
    config STRING NOT NULL,
    visibility STRING NOT NULL,
    validation_status STRING NOT NULL,
    last_validated_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
) USING DELTA CLUSTER BY (owner_id);

CREATE TABLE IF NOT EXISTS ${ns}.message_feedback (
    feedback_id STRING NOT NULL,
    message_id STRING NOT NULL,
    user_id STRING NOT NULL,
    ts TIMESTAMP NOT NULL,
    feedback STRING NOT NULL,
    CONSTRAINT message_feedback_pk PRIMARY KEY (feedback_id) RELY
) USING DELTA CLUSTER BY (message_id);

CREATE TABLE IF NOT EXISTS ${ns}.audit_log (
    log_id STRING NOT NULL,
    user_id STRING NOT NULL,
    ts TIMESTAMP NOT NULL,
    event STRING NOT NULL,
    CONSTRAINT audit_log_pk PRIMARY KEY (log_id) RELY
) USING DELTA CLUSTER BY (user_id);

CREATE TABLE IF NOT EXISTS ${ns}.chat_deleted_files (
    chat_id STRING NOT NULL,
    file_id STRING NOT NULL,
    CONSTRAINT chat_deleted_files_pk PRIMARY KEY (chat_id, file_id) RELY
) USING DELTA CLUSTER BY (chat_id);
