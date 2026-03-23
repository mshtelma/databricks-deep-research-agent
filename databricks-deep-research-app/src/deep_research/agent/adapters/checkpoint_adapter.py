"""Checkpoint adapter — wraps app DB persistence as CheckpointHandler.

Maps framework checkpoint save/load operations to the app's existing
session state persistence layer.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class AppCheckpointHandler:
    """Wraps app's DB persistence as a framework CheckpointHandler.

    The framework calls ``save()`` to checkpoint workflow state and
    ``load()`` to resume from a checkpoint.  This adapter maps those
    calls to the app's session/message persistence tables.

    Parameters
    ----------
    session_service:
        The app's session service for reading/writing session state.
    """

    def __init__(self, session_service: Any) -> None:
        self._session_service = session_service

    async def save(
        self,
        execution_id: str,
        workflow_id: str,
        state_dict: dict[str, Any],
    ) -> None:
        """Persist workflow state to the app's database.

        Args:
            execution_id: Unique execution instance ID.
            workflow_id: Workflow definition ID.
            state_dict: Serialised WorkflowState (from ``state.to_dict()``).
        """
        try:
            checkpoint_data = json.dumps({
                "workflow_id": workflow_id,
                "state": state_dict,
            })
            await self._session_service.update_session_metadata(
                session_id=execution_id,
                metadata_key="framework_checkpoint",
                metadata_value=checkpoint_data,
            )
            logger.debug(
                "CHECKPOINT_SAVED execution_id=%s workflow_id=%s",
                execution_id,
                workflow_id,
            )
        except Exception as e:
            logger.warning(
                "CHECKPOINT_SAVE_FAILED execution_id=%s error=%s",
                execution_id,
                str(e)[:200],
            )

    async def load(
        self,
        execution_id: str,
        _workflow_id: str,
    ) -> dict[str, Any] | None:
        """Load workflow state from the app's database.

        Args:
            execution_id: Unique execution instance ID.
            _workflow_id: Workflow definition ID (unused; kept for interface compat).

        Returns:
            Serialised state dict, or None if no checkpoint exists.
        """
        try:
            metadata = await self._session_service.get_session_metadata(
                session_id=execution_id,
                metadata_key="framework_checkpoint",
            )
            if metadata:
                data = json.loads(metadata)
                result: dict[str, Any] | None = data.get("state")
                return result
        except Exception as e:
            logger.warning(
                "CHECKPOINT_LOAD_FAILED execution_id=%s error=%s",
                execution_id,
                str(e)[:200],
            )
        return None


__all__ = ["AppCheckpointHandler"]
