"""DB-backed implementation of :class:`ValidationCacheProto` (migration 034).

Persists authoritative workflow-validation verdicts in ``workflow_validation_cache``
keyed by ``(validator_version, intent_hash, semantic_hash)`` so an unchanged
workflow reuses its verdict at save with no LLM call. Writes are idempotent
upserts (``ON CONFLICT DO NOTHING``) so concurrent same-hash saves never collide.
"""
from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.agent_designer.workflow_validation import WorkflowValidationResult
from deep_research.models.agent_v2 import WorkflowValidationCache


class DbValidationCache:
    """:class:`ValidationCacheProto` backed by ``workflow_validation_cache``.

    Shares the request's :class:`AsyncSession`; writes are flushed with the
    surrounding save transaction (the caller commits).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(
        self,
        *,
        validator_version: str,
        intent_hash: str,
        semantic_hash: str,
    ) -> WorkflowValidationResult | None:
        row = await self._session.get(
            WorkflowValidationCache,
            (validator_version, intent_hash, semantic_hash),
        )
        if row is None:
            return None
        return WorkflowValidationResult.model_validate(row.result)

    async def put(self, result: WorkflowValidationResult) -> None:
        stmt = (
            pg_insert(WorkflowValidationCache)
            .values(
                validator_version=result.validator_version,
                intent_hash=result.intent_hash,
                semantic_hash=result.semantic_hash,
                result=result.model_dump(mode="json"),
            )
            .on_conflict_do_nothing(
                index_elements=[
                    "validator_version",
                    "intent_hash",
                    "semantic_hash",
                ],
            )
        )
        await self._session.execute(stmt)
