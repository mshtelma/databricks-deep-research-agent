"""``@tool``-decorated wrappers exposing :class:`VirtualFilesystem` to LLMs.

Each tool reads ``ctx.extras["_framework_vfs"]`` so the runtime VFS is
attached via :class:`Agent.files` (Phase 2). Without an attached VFS, the
tools return a clear error message rather than crashing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from databricks_deep_research.tools.api import tool

if TYPE_CHECKING:
    from databricks_deep_research.api.vfs.protocol import VirtualFilesystem

_INJECT = {"_inject_vfs": "_framework_vfs"}


def _missing(op: str) -> str:
    return f"[error] {op}: no _framework_vfs attached to the agent"


@tool(name="ls", inject=_INJECT)
async def ls(path: str = "/", *, _inject_vfs: VirtualFilesystem | None = None) -> str:
    """List directory entries at the given path.

    Args:
        path: Directory path. Defaults to the root.

    Returns:
        Newline-separated entry names, or an error marker.
    """
    if _inject_vfs is None:
        return _missing("ls")
    entries = await _inject_vfs.ls(path)
    return "\n".join(entries) if entries else ""


@tool(name="read_file", inject=_INJECT)
async def read_file(path: str, *, _inject_vfs: VirtualFilesystem | None = None) -> str:
    """Read a file as UTF-8 text.

    Args:
        path: Path to the file.
    """
    if _inject_vfs is None:
        return _missing("read_file")
    data = await _inject_vfs.read(path)
    return data.decode("utf-8", errors="replace")


@tool(name="write_file", inject=_INJECT)
async def write_file(path: str, content: str, *, _inject_vfs: VirtualFilesystem | None = None) -> str:
    """Overwrite a file with the given UTF-8 content.

    Args:
        path: Destination path.
        content: New content.
    """
    if _inject_vfs is None:
        return _missing("write_file")
    await _inject_vfs.write(path, content)
    return f"wrote {len(content)} chars to {path}"


@tool(name="edit_file", inject=_INJECT)
async def edit_file(
    path: str,
    old: str,
    new: str,
    unique: bool = True,
    *,
    _inject_vfs: VirtualFilesystem | None = None,
) -> str:
    """Replace ``old`` with ``new`` in the file at ``path``.

    Args:
        path: File path.
        old: Substring to replace.
        new: Replacement.
        unique: If true, fail when ``old`` appears more than once.
    """
    if _inject_vfs is None:
        return _missing("edit_file")
    await _inject_vfs.edit(path, old, new, unique=unique)
    return f"edited {path}"


@tool(name="grep", inject=_INJECT)
async def grep(
    pattern: str,
    path: str = "/",
    max_matches: int = 100,
    *,
    _inject_vfs: VirtualFilesystem | None = None,
) -> str:
    """Regex-search files in the given path.

    Args:
        pattern: Python regex.
        path: Directory to scan.
        max_matches: Cap on the number of matches returned.
    """
    if _inject_vfs is None:
        return _missing("grep")
    matches = await _inject_vfs.grep(pattern, path, max_matches=max_matches)
    if not matches:
        return f"no matches for {pattern!r} in {path}"
    return "\n".join(
        f"{m.get('path')}:{m.get('line')}: {m.get('text')}" for m in matches
    )


@tool(name="delete_file", inject=_INJECT)
async def delete_file(path: str, *, _inject_vfs: VirtualFilesystem | None = None) -> str:
    """Delete a file.

    Args:
        path: Path to the file.
    """
    if _inject_vfs is None:
        return _missing("delete_file")
    await _inject_vfs.delete(path)
    return f"deleted {path}"


@tool(name="exists", inject=_INJECT)
async def exists(path: str, *, _inject_vfs: VirtualFilesystem | None = None) -> str:
    """Check whether a path exists.

    Args:
        path: Path to test.
    """
    if _inject_vfs is None:
        return _missing("exists")
    return "true" if await _inject_vfs.exists(path) else "false"


VFS_TOOLS = [ls, read_file, write_file, edit_file, grep, delete_file, exists]


__all__ = [
    "VFS_TOOLS",
    "delete_file",
    "edit_file",
    "exists",
    "grep",
    "ls",
    "read_file",
    "write_file",
]
