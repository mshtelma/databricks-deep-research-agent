"""One-time bulk migration from the legacy normalized schema to the new
chat-document schema, both living in Lakebase.

**Critical invariants:**

* **Refuses to run** unless `STORAGE_MIGRATION_MODE=1` is set — prevents
  accidental production mutation.
* **Targets a fresh schema** chosen via `settings.storage_schema`. The legacy
  normalized tables live in the default (`public`) schema; the new schema is
  separate so the live app (reading from legacy) and the migration (writing
  to new) never contend.
* **Resumable**: stores the last-migrated chat_id in `storage_meta` under the
  key `migration_cursor_chat_id`. On restart the migration picks up where it
  left off.
* **Idempotent**: every write is `ON CONFLICT DO UPDATE`, so repeated runs
  converge.
* **Chunked**: processes chats in batches of `--chunk-size` per transaction
  so a crash leaves a clean boundary.

Usage::

    STORAGE_MIGRATION_MODE=1 \\
        python -m deep_research.storage.migrate_lakebase \\
            --chunk-size=500 [--dry-run] [--since=<uuid>]

Out of scope for v1:

* Migrating `research_events`, `file_chunks`, `message_feedback`, `audit_log`
  — these live in *append-only* tables whose new-schema shape is identical to
  the legacy one, so they can be migrated via plain ``INSERT … SELECT``
  without involving the document encoder. A second script, or a manual SQL
  run, handles them. Implemented here as ``migrate_append_only_tables`` but
  only when ``--include-append=true``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger("deep_research.storage.migrate_lakebase")


DEFAULT_CHUNK_SIZE = 500


async def migrate(
    session_maker: async_sessionmaker[AsyncSession],
    *,
    target_schema: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    since: uuid.UUID | None = None,
    dry_run: bool = False,
    include_append: bool = False,
) -> dict[str, Any]:
    """Run the full migration and return a summary dict."""
    _require_migration_mode()
    _assert_safe_schema(target_schema)

    summary = {
        "target_schema": target_schema,
        "chunk_size": chunk_size,
        "chats_migrated": 0,
        "dry_run": dry_run,
        "errors": 0,
    }
    cursor = since
    if cursor is None:
        cursor = await _load_cursor(session_maker, target_schema)

    logger.info(
        "migration starting target_schema=%s chunk_size=%d since=%s dry_run=%s",
        target_schema,
        chunk_size,
        cursor,
        dry_run,
    )

    while True:
        async with session_maker() as session:
            chat_ids = await _fetch_next_chat_ids(session, after=cursor, limit=chunk_size)
        if not chat_ids:
            break

        logger.info("migrating chunk of %d chats (first=%s)", len(chat_ids), chat_ids[0])
        if not dry_run:
            try:
                async with session_maker() as session:
                    async with session.begin():
                        await _migrate_chunk(session, target_schema, chat_ids)
                        await _save_cursor(session, target_schema, chat_ids[-1])
            except Exception:  # noqa: BLE001
                summary["errors"] = int(summary["errors"]) + 1
                logger.exception("chunk failed; leaving cursor unchanged for retry")
                # Stop so operator can investigate; safer than silent skip.
                break
        summary["chats_migrated"] = int(summary["chats_migrated"]) + len(chat_ids)
        cursor = chat_ids[-1]

    if include_append and not dry_run:
        async with session_maker() as session:
            async with session.begin():
                await _migrate_append_only_tables(session, target_schema)

    logger.info("migration done: %s", summary)
    return summary


# --- SQL helpers ------------------------------------------------------------


async def _fetch_next_chat_ids(
    session: AsyncSession,
    *,
    after: uuid.UUID | None,
    limit: int,
) -> list[uuid.UUID]:
    if after is None:
        sql = "SELECT id FROM chats ORDER BY id ASC LIMIT :lim"
        params: dict[str, Any] = {"lim": limit}
    else:
        sql = "SELECT id FROM chats WHERE id > :after ORDER BY id ASC LIMIT :lim"
        params = {"after": after, "lim": limit}
    rows = (await session.execute(text(sql), params)).all()
    return [r[0] for r in rows]


async def _migrate_chunk(
    session: AsyncSession,
    target_schema: str,
    chat_ids: list[uuid.UUID],
) -> None:
    """Migrate a batch of chat_ids in a single transaction.

    Emits two statements per chunk: one for `chat_meta`, one for `chat_state`.
    All document-level substructures are assembled via `jsonb_build_object`
    + correlated `jsonb_agg` subqueries — no per-row round-trip.
    """
    # Bind the chat_id array once; both statements reuse it.
    params = {"ids": chat_ids}

    meta_sql = text(
        f"""
        INSERT INTO {target_schema}.chat_meta
          (chat_id, user_id, title, preview, created_at, updated_at, deleted_at, version)
        SELECT
          c.id,
          c.user_id,
          COALESCE(c.title, ''),
          COALESCE(
            LEFT(
              (SELECT m.content FROM messages m
               WHERE m.chat_id = c.id ORDER BY m.created_at ASC LIMIT 1),
              120
            ),
            ''
          ),
          c.created_at,
          c.updated_at,
          c.deleted_at,
          1
        FROM chats c
        WHERE c.id = ANY(:ids)
        ON CONFLICT (chat_id) DO UPDATE SET
          user_id = EXCLUDED.user_id,
          title = EXCLUDED.title,
          preview = EXCLUDED.preview,
          created_at = EXCLUDED.created_at,
          updated_at = EXCLUDED.updated_at,
          deleted_at = EXCLUDED.deleted_at,
          version = {target_schema}.chat_meta.version + 1
        """
    )
    await session.execute(meta_sql, params)

    state_sql = text(
        f"""
        INSERT INTO {target_schema}.chat_state (chat_id, state)
        SELECT
          c.id,
          jsonb_build_object(
            'schema_version', 1,
            'chat', jsonb_build_object(
              'type', COALESCE(c.chat_type, 'native'),
              'title', COALESCE(c.title, ''),
              'incognito_session_id', c.incognito_session_id,
              'custom_agent_id', NULL,
              'metadata', COALESCE(c.metadata, '{{}}'::jsonb)
            ),
            'messages', COALESCE((
              SELECT jsonb_agg(jsonb_build_object(
                'id', m.id,
                'role', m.role,
                'content', m.content,
                'ts', m.created_at,
                'metadata', COALESCE(m.metadata, '{{}}'::jsonb)
              ) ORDER BY m.created_at ASC)
              FROM (
                SELECT * FROM messages WHERE chat_id = c.id
                ORDER BY created_at DESC LIMIT 50
              ) m
            ), '[]'::jsonb),
            'memory', jsonb_build_object(
              'findings', COALESCE((
                SELECT jsonb_agg(jsonb_build_object(
                  'id', f.id,
                  'content_hash', f.content_hash,
                  'content', f.content,
                  'summary', f.summary,
                  'step', COALESCE(f.source_step, 0),
                  'origin', COALESCE(f.origin, 'web'),
                  'confidence', COALESCE(f.confidence, 'medium'),
                  'entity_ids', COALESCE(f.entity_ids, '[]'::jsonb),
                  'source_ids', COALESCE(f.source_ids, '[]'::jsonb)
                ))
                FROM chat_memory_findings f WHERE f.chat_id = c.id
              ), '[]'::jsonb),
              'entities', COALESCE((
                SELECT jsonb_agg(jsonb_build_object(
                  'id', e.id, 'name', e.name, 'type', COALESCE(e.entity_type, 'other'),
                  'aliases', COALESCE(e.aliases, '[]'::jsonb),
                  'supporting_finding_ids', COALESCE(e.supporting_finding_ids, '[]'::jsonb)
                ))
                FROM chat_memory_entities e WHERE e.chat_id = c.id
              ), '[]'::jsonb),
              'coverage', COALESCE((
                SELECT jsonb_agg(jsonb_build_object(
                  'id', cov.id, 'topic', cov.topic,
                  'status', COALESCE(cov.status, 'gap'),
                  'depth', cov.depth
                ))
                FROM chat_memory_coverage cov WHERE cov.chat_id = c.id
              ), '[]'::jsonb),
              'files', COALESCE((
                SELECT jsonb_agg(jsonb_build_object(
                  'id', cmf.file_id, 'name', cmf.name,
                  'status', COALESCE(cmf.status, 'processed'),
                  'entity_ids', COALESCE(cmf.entity_ids, '[]'::jsonb),
                  'summary', cmf.summary
                ))
                FROM chat_memory_files cmf WHERE cmf.chat_id = c.id
              ), '[]'::jsonb),
              'plugin_ext', COALESCE((
                SELECT jsonb_object_agg(
                  pe.plugin_name,
                  jsonb_build_object('payload', COALESCE(pe.payload_json, '{{}}'::jsonb))
                )
                FROM chat_memory_plugin_ext pe WHERE pe.chat_id = c.id
              ), '{{}}'::jsonb)
            ),
            'sources', COALESCE((
              SELECT jsonb_agg(jsonb_build_object(
                'id', s.id, 'url', s.url, 'title', s.title,
                'last_used_step', COALESCE(s.last_used_step, 0),
                'source_type', COALESCE(s.source_type, 'web'),
                'metadata', COALESCE(s.source_metadata, '{{}}'::jsonb)
              ))
              FROM (
                SELECT * FROM sources WHERE chat_id = c.id
                ORDER BY last_used_step DESC NULLS LAST LIMIT 200
              ) s
            ), '[]'::jsonb),
            'research_sessions', COALESCE((
              SELECT jsonb_agg(jsonb_build_object(
                'id', rs.id, 'message_id', rs.message_id,
                'status', rs.status,
                'plan', COALESCE(rs.plan, '{{}}'::jsonb),
                'observations', COALESCE(rs.observations, '{{}}'::jsonb),
                'query_classification', COALESCE(rs.query_classification, '{{}}'::jsonb),
                'execution_state', COALESCE(rs.execution_state, '{{}}'::jsonb),
                'verification_data', COALESCE(rs.verification_data, '{{}}'::jsonb),
                'current_step', COALESCE(rs.current_step, 0),
                'started_at', rs.started_at,
                'completed_at', rs.completed_at
              ))
              FROM (
                SELECT * FROM research_sessions WHERE chat_id = c.id
                ORDER BY started_at DESC NULLS LAST LIMIT 10
              ) rs
            ), '[]'::jsonb),
            'uploaded_files', COALESCE((
              SELECT jsonb_agg(jsonb_build_object(
                'id', uf.id, 'name', uf.name, 'size', uf.size,
                'mime', uf.mime_type, 'status', COALESCE(uf.status, 'processed'),
                'summary_ref', NULL
              ))
              FROM uploaded_files uf
              WHERE uf.session_id IN (
                SELECT id FROM incognito_sessions WHERE id = c.incognito_session_id
              ) OR uf.owner_id = c.user_id
            ), '[]'::jsonb)
          )
        FROM chats c
        WHERE c.id = ANY(:ids)
        ON CONFLICT (chat_id) DO UPDATE SET state = EXCLUDED.state
        """
    )
    await session.execute(state_sql, params)


async def _migrate_append_only_tables(
    session: AsyncSession,
    target_schema: str,
) -> None:
    """Straight-through INSERT…SELECT for append-only data.

    Schema shape is identical between legacy and new; only the *schema
    namespace* differs. On conflict we DO NOTHING because rows are immutable.
    """
    for stmt in (
        (f"INSERT INTO {target_schema}.research_events "
         "(session_id, sequence_number, ts, event) "
         "SELECT research_session_id, sequence_number, created_at, event_data FROM research_events "
         "ON CONFLICT (session_id, sequence_number) DO NOTHING"),
        (f"INSERT INTO {target_schema}.file_chunks "
         "(file_id, chunk_index, ts, content, metadata) "
         "SELECT file_id, chunk_index, created_at, content, COALESCE(metadata, '{}'::jsonb) FROM file_chunks "
         "ON CONFLICT (file_id, chunk_index) DO NOTHING"),
        (f"INSERT INTO {target_schema}.message_feedback "
         "(feedback_id, message_id, user_id, ts, feedback) "
         "SELECT id, message_id, user_id, created_at, jsonb_build_object('rating', rating, 'comment', comment) "
         "FROM message_feedback ON CONFLICT (feedback_id) DO NOTHING"),
        (f"INSERT INTO {target_schema}.audit_log "
         "(log_id, user_id, ts, event) "
         "SELECT id, user_id, created_at, jsonb_build_object('action', action, 'target_id', target_id, 'details', details) "
         "FROM audit_log ON CONFLICT (log_id) DO NOTHING"),
    ):
        await session.execute(text(stmt))


# --- Cursor ----------------------------------------------------------------


_CURSOR_KEY = "migration_cursor_chat_id"


async def _load_cursor(
    session_maker: async_sessionmaker[AsyncSession],
    target_schema: str,
) -> uuid.UUID | None:
    async with session_maker() as session:
        row = (
            await session.execute(
                text(f"SELECT value FROM {target_schema}.storage_meta WHERE key = :k"),
                {"k": _CURSOR_KEY},
            )
        ).first()
    if row is None:
        return None
    try:
        return uuid.UUID(row[0])
    except (ValueError, TypeError):
        return None


async def _save_cursor(
    session: AsyncSession,
    target_schema: str,
    cursor: uuid.UUID,
) -> None:
    await session.execute(
        text(
            f"INSERT INTO {target_schema}.storage_meta (key, value, updated_at) "
            f"VALUES (:k, :v, now()) "
            f"ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()"
        ),
        {"k": _CURSOR_KEY, "v": str(cursor)},
    )


# --- Safety -----------------------------------------------------------------


def _require_migration_mode() -> None:
    if os.environ.get("STORAGE_MIGRATION_MODE") != "1":
        raise SystemExit(
            "migrate_lakebase.py refuses to run without STORAGE_MIGRATION_MODE=1.\n"
            "Set STORAGE_MIGRATION_MODE=1 in the environment to proceed."
        )


def _assert_safe_schema(name: str) -> None:
    import re

    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", name):
        raise SystemExit(f"unsafe target schema name: {name!r}")


# --- CLI --------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--since", type=uuid.UUID, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-append",
        action="store_true",
        help="Also migrate research_events / file_chunks / feedback / audit_log.",
    )
    parser.add_argument(
        "--target-schema",
        default=None,
        help="Override Settings.storage_schema (useful for test runs).",
    )
    return parser


async def _main_async() -> None:
    args = _build_parser().parse_args()

    # Import the app-side session maker lazily so unit tests don't need it.
    from deep_research.core.config import get_settings
    from deep_research.db.session import get_session_maker

    settings = get_settings()
    target_schema = args.target_schema or settings.storage_schema

    summary = await migrate(
        get_session_maker(settings),
        target_schema=target_schema,
        chunk_size=args.chunk_size,
        since=args.since,
        dry_run=args.dry_run,
        include_append=args.include_append,
    )
    print(json.dumps(summary, indent=2, default=str))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
