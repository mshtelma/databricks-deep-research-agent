-- cleanup_legacy_tables.sql
-- =============================================================================
-- Drop the legacy normalized public.* tables that are superseded by the
-- event-sourced storage stack (schema "deep_research_state": chat_state,
-- chat_meta, research_events, ...). After the ORM->storage migration, no code
-- path reads or writes these tables.
--
-- !!! RUN ONLY against a deployment whose binary already routes jobs,
-- !!! citations, research, and messages through the storage stack (i.e. has
-- !!! the "Complete the ORM->storage-stack migration" change deployed). Running
-- !!! against an older binary that still issues select(ResearchSession)/
-- !!! MessageService(db) etc. will turn empty-table reads into 500s.
--
-- KEEPS (still live, NOT in the drop list): agents_v2, agent_revisions,
-- custom_tool_defs, agent_deployments (Agent Designer V2), users,
-- user_preferences, and alembic_version.
--
-- Safety: this script REFUSES to drop a table that still has rows (data-loss
-- guard) and is idempotent (DROP ... IF EXISTS ... CASCADE). The active store
-- in schema "deep_research_state" is created by storage/lakebase.py and is
-- never touched here.
--
-- Usage (developer, against the target Lakebase):
--   make db-drop-legacy TARGET=<dev|ais>
-- or directly:
--   psql "$CONN" -f scripts/cleanup_legacy_tables.sql
-- =============================================================================

DO $$
DECLARE
    t text;
    n bigint;
    drop_list text[] := ARRAY[
        'research_sessions',
        'messages',
        'chats',
        'sources',
        'research_events',
        'chat_memory_coverage',
        'chat_memory_entities',
        'chat_memory_files',
        'chat_memory_findings',
        'chat_memory_plugin_ext',
        'incognito_sessions',
        'message_feedback',
        'uploaded_files',
        'file_chunks',
        'prompt_templates',
        'user_data_sources',
        'audit_logs'
    ];
BEGIN
    -- Phase 1 — safety precheck: refuse to drop any table that still has rows.
    FOREACH t IN ARRAY drop_list LOOP
        IF to_regclass('public.' || quote_ident(t)) IS NOT NULL THEN
            EXECUTE format('SELECT count(*) FROM public.%I', t) INTO n;
            IF n > 0 THEN
                RAISE EXCEPTION
                    'Refusing to drop non-empty table public.% (% rows). The storage migration is supposed to leave these empty — investigate before dropping.',
                    t, n;
            END IF;
        END IF;
    END LOOP;

    -- Phase 2 — drop (idempotent; CASCADE clears dependent FKs/views/indexes).
    FOREACH t IN ARRAY drop_list LOOP
        EXECUTE format('DROP TABLE IF EXISTS public.%I CASCADE', t);
        RAISE NOTICE 'cleanup_legacy_tables: dropped public.% (if it existed)', t;
    END LOOP;

    RAISE NOTICE 'cleanup_legacy_tables: done. Kept agents_v2/agent_revisions/custom_tool_defs/agent_deployments/users/user_preferences/alembic_version.';
END $$;
