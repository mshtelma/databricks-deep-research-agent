"""
No-operation checkpointer for LangGraph.

This checkpointer satisfies the LangGraph interface requirements without
actually storing any data, making it ideal for tests and scenarios where
checkpoint functionality is not needed but memory usage needs to be minimized.
"""

from typing import Iterator, Any, Dict, Optional, Sequence, Tuple
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from langchain_core.runnables import RunnableConfig

from . import get_logger

logger = get_logger(__name__)


class NoOpCheckpointSaver(BaseCheckpointSaver):
    """
    A no-operation checkpointer that implements the LangGraph interface
    but doesn't actually store any data.
    
    This is useful for:
    - Tests that don't need checkpoint functionality
    - Memory-constrained environments
    - Single-run scenarios where state persistence isn't required
    """
    
    def __init__(self):
        """Initialize the no-op checkpointer."""
        logger.info("Initialized NoOpCheckpointSaver - no checkpoints will be stored")
    
    def put(
        self, 
        config: RunnableConfig, 
        checkpoint: Checkpoint, 
        metadata: CheckpointMetadata,
        new_versions: Optional[Dict[str, Any]] = None
    ) -> RunnableConfig:
        """
        Store a checkpoint (no-op implementation).
        
        Args:
            config: The runnable configuration
            checkpoint: The checkpoint to store
            metadata: Checkpoint metadata
            new_versions: Channel versions (optional)
            
        Returns:
            The same config (unchanged)
        """
        # Don't actually store anything
        return config
    
    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        """
        Get a checkpoint (no-op implementation).
        
        Args:
            config: The runnable configuration
            
        Returns:
            None (no checkpoints are stored)
        """
        return None
    
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """
        Get a checkpoint tuple (no-op implementation).
        
        Args:
            config: The runnable configuration
            
        Returns:
            None (no checkpoints are stored)
        """
        return None
    
    def list(
        self,
        config: Optional[RunnableConfig] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None
    ) -> Iterator[CheckpointTuple]:
        """
        List checkpoints (no-op implementation).
        
        Args:
            config: Optional runnable configuration
            filter: Optional filter criteria
            before: Optional before configuration
            limit: Optional limit on results
            
        Yields:
            Nothing (no checkpoints are stored)
        """
        # Return empty iterator
        return iter([])
    
    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = ""
    ) -> None:
        """
        Store intermediate writes (no-op implementation).
        
        Args:
            config: The runnable configuration
            writes: The writes to store
            task_id: Task identifier
            task_path: Task path
        """
        # Don't actually store anything
        pass
    
    # Async versions (if needed)
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[Dict[str, Any]] = None
    ) -> RunnableConfig:
        """Async version of put (no-op implementation)."""
        return config
    
    async def aget(self, config: RunnableConfig) -> Optional[Checkpoint]:
        """Async version of get (no-op implementation)."""
        return None
    
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Async version of get_tuple (no-op implementation)."""
        return None
    
    async def alist(
        self,
        config: Optional[RunnableConfig] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None
    ) -> Iterator[CheckpointTuple]:
        """Async version of list (no-op implementation)."""
        return iter([])
    
    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = ""
    ) -> None:
        """Async version of put_writes (no-op implementation)."""
        pass
    
    @property
    def config_specs(self) -> list:
        """
        Configuration specs (empty for no-op).
        
        Returns:
            Empty list (no configuration needed)
        """
        return []


# Convenience function for easy import
def create_noop_checkpointer() -> NoOpCheckpointSaver:
    """Create a new NoOpCheckpointSaver instance."""
    return NoOpCheckpointSaver()