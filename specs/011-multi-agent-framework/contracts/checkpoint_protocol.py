"""
Contract: Checkpoint Handler Protocol.

The framework defines this protocol for state persistence.
The consuming application provides the implementation (e.g., DB-backed).
The framework itself has zero database dependencies.

NOTE (2026-03-08 session): Checkpointing is defined in P0 as a protocol
but the full implementation (CheckpointConfig, granularity settings,
automatic checkpoint after every N nodes) is deferred beyond P0.
P0 includes the protocol definition so the app can optionally persist
state, but the executor does not automatically checkpoint.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CheckpointHandler(Protocol):
    """Protocol for persisting workflow state checkpoints.

    The framework calls save() after leaf nodes complete (based on CheckpointConfig).
    The framework calls load() when resuming from a checkpoint.

    Example implementation (Deep Research app):
        class DatabaseCheckpointHandler:
            def __init__(self, session_id: str, db_session: AsyncSession):
                self._session_id = session_id
                self._db = db_session

            async def save(self, execution_id: str, workflow_id: str, state_dict: dict[str, Any]) -> None:
                await self._db.execute(
                    update(ResearchSession)
                    .where(ResearchSession.id == self._session_id)
                    .values(execution_state=state_dict)
                )
                await self._db.commit()

            async def load(self, execution_id: str, workflow_id: str) -> dict[str, Any] | None:
                result = await self._db.execute(
                    select(ResearchSession.execution_state)
                    .where(ResearchSession.id == self._session_id)
                )
                row = result.scalar_one_or_none()
                return row if row else None
    """

    async def save(self, execution_id: str, workflow_id: str, state_dict: dict[str, Any]) -> None:
        """Persist a workflow state checkpoint.

        Args:
            execution_id: Unique identifier for this execution run.
            workflow_id: The workflow definition ID.
            state_dict: Serialized WorkflowState (from state.to_dict()).
        """
        ...

    async def load(self, execution_id: str, workflow_id: str) -> dict[str, Any] | None:
        """Load the most recent checkpoint for a workflow execution.

        Args:
            execution_id: Unique identifier for this execution run.
            workflow_id: The workflow definition ID.

        Returns:
            Serialized state dict, or None if no checkpoint exists.
        """
        ...
