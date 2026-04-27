"""Virtual filesystem protocol for DeepAgents-style scratch storage."""

from __future__ import annotations

from typing import Protocol


class VirtualFilesystem(Protocol):
    """Async filesystem-like store backed by an in-memory dict, UC volume, etc.

    Phase-2 ships :class:`InMemoryBackend` and :class:`UCVolumeBackend`.
    Additional backends (LocalDisk, WorkspaceFiles, fsspec) are deferred.
    """

    async def ls(self, path: str = "/") -> list[str]: ...
    async def read(self, path: str) -> bytes: ...
    async def write(self, path: str, content: bytes | str) -> None: ...
    async def edit(self, path: str, old: str, new: str, *, unique: bool = True) -> None: ...
    async def grep(self, pattern: str, path: str = "/", *, max_matches: int = 100) -> list[dict]: ...
    async def delete(self, path: str) -> None: ...
    async def exists(self, path: str) -> bool: ...


__all__ = ["VirtualFilesystem"]
