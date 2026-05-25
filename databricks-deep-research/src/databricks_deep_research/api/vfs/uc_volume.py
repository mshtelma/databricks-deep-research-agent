"""Databricks-native :class:`VirtualFilesystem` backed by a UC Volume.

Uses ``WorkspaceClient.files`` (upload / download / list_directory_contents
/ delete). Per-run LRU cache invalidated on write. ``grep`` is client-side
in Phase 2; SQL pushdown via ``read_files()`` is deferred to a follow-up.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _parse_volume_path(volume: str) -> str:
    """Return a Files-API absolute path from a ``catalog.schema.volume`` spec.

    Accepts both forms:
    - ``"catalog.schema.volume"`` → ``"/Volumes/catalog/schema/volume"``
    - ``"/Volumes/catalog/schema/volume"`` → unchanged
    """
    if volume.startswith("/Volumes/"):
        return volume.rstrip("/")
    parts = volume.split(".")
    if len(parts) != 3:
        raise ValueError(
            f"UC volume must be 'catalog.schema.volume' or '/Volumes/...'; got {volume!r}"
        )
    catalog, schema, vol = parts
    return f"/Volumes/{catalog}/{schema}/{vol}"


class UCVolumeBackend:
    """Databricks Unity Catalog Volume-backed virtual filesystem.

    Args:
        volume: ``catalog.schema.volume`` or ``/Volumes/...`` path.
        workspace_client: Optional :class:`WorkspaceClient`. When ``None``,
            constructs one via SDK auth chain.
        cache_size: Maximum entries kept in the per-instance read cache.
    """

    def __init__(
        self,
        volume: str,
        workspace_client: Any | None = None,
        *,
        cache_size: int = 64,
    ) -> None:
        self._volume_path = _parse_volume_path(volume)
        self._wc = workspace_client or self._build_workspace_client()
        self._cache: dict[str, bytes] = {}
        self._cache_order: list[str] = []
        self._cache_size = cache_size

    def _build_workspace_client(self) -> Any:
        try:
            from databricks.sdk import WorkspaceClient
        except ImportError as exc:
            raise RuntimeError(
                "UCVolumeBackend requires databricks-sdk; "
                "install with `pip install databricks-sdk`."
            ) from exc
        return WorkspaceClient()

    def _full(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self._volume_path}{path}"

    def _cache_get(self, key: str) -> bytes | None:
        return self._cache.get(key)

    def _cache_put(self, key: str, value: bytes) -> None:
        if key in self._cache:
            self._cache_order.remove(key)
        elif len(self._cache_order) >= self._cache_size:
            evicted = self._cache_order.pop(0)
            self._cache.pop(evicted, None)
        self._cache[key] = value
        self._cache_order.append(key)

    def _cache_invalidate(self, key: str) -> None:
        if key in self._cache:
            self._cache.pop(key, None)
            with contextlib.suppress(ValueError):
                self._cache_order.remove(key)

    async def ls(self, path: str = "/") -> list[str]:
        full = self._full(path).rstrip("/")
        loop = asyncio.get_event_loop()

        def _list() -> list[str]:
            try:
                entries = list(self._wc.files.list_directory_contents(full))
            except Exception:  # noqa: BLE001
                return []
            return sorted(getattr(e, "name", "") for e in entries if getattr(e, "name", ""))

        return await loop.run_in_executor(None, _list)

    async def read(self, path: str) -> bytes:
        cached = self._cache_get(path)
        if cached is not None:
            return cached
        full = self._full(path)
        loop = asyncio.get_event_loop()

        def _download() -> bytes:
            response = self._wc.files.download(full)
            contents = response.contents
            if hasattr(contents, "read"):
                return contents.read()  # type: ignore[no-any-return]
            return bytes(contents) if not isinstance(contents, bytes) else contents

        data = await loop.run_in_executor(None, _download)
        self._cache_put(path, data)
        return data

    async def write(self, path: str, content: bytes | str) -> None:
        full = self._full(path)
        payload = content if isinstance(content, bytes) else content.encode("utf-8")
        loop = asyncio.get_event_loop()

        def _upload() -> None:
            self._wc.files.upload(full, io.BytesIO(payload), overwrite=True)

        await loop.run_in_executor(None, _upload)
        self._cache_invalidate(path)
        # Also pre-warm the cache so an immediate read-after-write is fast.
        self._cache_put(path, payload)

    async def edit(
        self, path: str, old: str, new: str, *, unique: bool = True,
    ) -> None:
        data = await self.read(path)
        text = data.decode("utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            raise ValueError(f"edit: {old!r} not found in {path}")
        if unique and count > 1:
            raise ValueError(
                f"edit: {old!r} matches {count} locations in {path}; "
                "set unique=False to replace all"
            )
        await self.write(path, text.replace(old, new))

    async def grep(
        self, pattern: str, path: str = "/", *, max_matches: int = 100,
    ) -> list[dict[str, str | int]]:
        regex = re.compile(pattern)
        names = await self.ls(path)
        results: list[dict[str, str | int]] = []
        prefix = path.rstrip("/")
        for name in names:
            full_path = f"{prefix}/{name}" if prefix not in ("", "/") else f"/{name}"
            try:
                data = await self.read(full_path)
            except Exception:  # noqa: BLE001
                continue
            text = data.decode("utf-8", errors="replace")
            for line_num, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    results.append({
                        "path": full_path,
                        "line": line_num,
                        "text": line,
                    })
                    if len(results) >= max_matches:
                        return results
        return results

    async def delete(self, path: str) -> None:
        full = self._full(path)
        loop = asyncio.get_event_loop()

        def _delete() -> None:
            self._wc.files.delete(full)

        await loop.run_in_executor(None, _delete)
        self._cache_invalidate(path)

    async def exists(self, path: str) -> bool:
        full = self._full(path)
        loop = asyncio.get_event_loop()

        def _check() -> bool:
            try:
                self._wc.files.get_metadata(full)
                return True
            except Exception:  # noqa: BLE001
                return False

        return await loop.run_in_executor(None, _check)


__all__ = ["UCVolumeBackend"]
