"""In-memory :class:`VirtualFilesystem` — zero-config dev/test backend."""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict


class InMemoryBackend:
    """Pure in-memory virtual filesystem.

    Per-path :class:`asyncio.Lock` guards write/edit so concurrent agents
    inside a ``Parallel(...)`` node don't corrupt files. Reads and ``ls``
    don't lock — last-writer-wins is acceptable for read-after-write
    eventual consistency.
    """

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _normalize(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return path

    async def ls(self, path: str = "/") -> list[str]:
        prefix = self._normalize(path).rstrip("/") + "/"
        if path in ("", "/"):
            prefix = "/"
        return sorted(
            p[len(prefix):].split("/", 1)[0]
            for p in self._store
            if p.startswith(prefix) and p != prefix
        )

    async def read(self, path: str) -> bytes:
        path = self._normalize(path)
        if path not in self._store:
            raise FileNotFoundError(path)
        return self._store[path]

    async def write(self, path: str, content: bytes | str) -> None:
        path = self._normalize(path)
        async with self._locks[path]:
            self._store[path] = (
                content if isinstance(content, bytes) else content.encode("utf-8")
            )

    async def edit(
        self, path: str, old: str, new: str, *, unique: bool = True,
    ) -> None:
        path = self._normalize(path)
        async with self._locks[path]:
            if path not in self._store:
                raise FileNotFoundError(path)
            text = self._store[path].decode("utf-8", errors="replace")
            count = text.count(old)
            if count == 0:
                raise ValueError(f"edit: {old!r} not found in {path}")
            if unique and count > 1:
                raise ValueError(
                    f"edit: {old!r} matches {count} locations in {path}; "
                    "set unique=False to replace all"
                )
            self._store[path] = text.replace(old, new).encode("utf-8")

    async def grep(
        self, pattern: str, path: str = "/", *, max_matches: int = 100,
    ) -> list[dict[str, str | int]]:
        regex = re.compile(pattern)
        prefix = self._normalize(path).rstrip("/") + "/" if path not in ("", "/") else "/"
        results: list[dict[str, str | int]] = []
        for fpath, data in self._store.items():
            if not fpath.startswith(prefix):
                continue
            text = data.decode("utf-8", errors="replace")
            for line_num, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    results.append({
                        "path": fpath,
                        "line": line_num,
                        "text": line,
                    })
                    if len(results) >= max_matches:
                        return results
        return results

    async def delete(self, path: str) -> None:
        path = self._normalize(path)
        async with self._locks[path]:
            self._store.pop(path, None)

    async def exists(self, path: str) -> bool:
        return self._normalize(path) in self._store


__all__ = ["InMemoryBackend"]
