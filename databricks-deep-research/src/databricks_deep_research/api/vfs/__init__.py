"""Virtual filesystem package: protocol + InMemory + UCVolume backends."""

from databricks_deep_research.api.vfs.in_memory import InMemoryBackend
from databricks_deep_research.api.vfs.protocol import VirtualFilesystem
from databricks_deep_research.api.vfs.uc_volume import UCVolumeBackend

__all__ = ["InMemoryBackend", "UCVolumeBackend", "VirtualFilesystem"]
