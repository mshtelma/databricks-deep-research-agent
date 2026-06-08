"""InMemoryBackend tests — 7 ops + edit-unique guard + concurrent locks."""

from __future__ import annotations

import asyncio

import pytest

from databricks_deep_research.api import InMemoryBackend


@pytest.mark.asyncio
async def test_write_and_read() -> None:
    fs = InMemoryBackend()
    await fs.write("/foo.txt", "hello")
    assert (await fs.read("/foo.txt")).decode() == "hello"


@pytest.mark.asyncio
async def test_exists_and_delete() -> None:
    fs = InMemoryBackend()
    await fs.write("/foo.txt", "hi")
    assert await fs.exists("/foo.txt")
    await fs.delete("/foo.txt")
    assert not await fs.exists("/foo.txt")


@pytest.mark.asyncio
async def test_ls_root() -> None:
    fs = InMemoryBackend()
    await fs.write("/a.txt", "")
    await fs.write("/b.txt", "")
    entries = await fs.ls("/")
    assert "a.txt" in entries
    assert "b.txt" in entries


@pytest.mark.asyncio
async def test_edit_replaces_unique_match() -> None:
    fs = InMemoryBackend()
    await fs.write("/foo.txt", "hello world")
    await fs.edit("/foo.txt", "world", "there")
    assert (await fs.read("/foo.txt")).decode() == "hello there"


@pytest.mark.asyncio
async def test_edit_unique_fails_on_multiple_matches() -> None:
    fs = InMemoryBackend()
    await fs.write("/foo.txt", "AAA AAA")
    with pytest.raises(ValueError, match="2 locations"):
        await fs.edit("/foo.txt", "AAA", "BBB", unique=True)


@pytest.mark.asyncio
async def test_edit_non_unique_replaces_all() -> None:
    fs = InMemoryBackend()
    await fs.write("/foo.txt", "AAA AAA")
    await fs.edit("/foo.txt", "AAA", "BBB", unique=False)
    assert (await fs.read("/foo.txt")).decode() == "BBB BBB"


@pytest.mark.asyncio
async def test_edit_missing_substring_raises() -> None:
    fs = InMemoryBackend()
    await fs.write("/foo.txt", "hello")
    with pytest.raises(ValueError, match="not found"):
        await fs.edit("/foo.txt", "MISSING", "X")


@pytest.mark.asyncio
async def test_grep_finds_matches() -> None:
    fs = InMemoryBackend()
    await fs.write("/a.txt", "foo bar\nbaz qux")
    matches = await fs.grep("ba", "/")
    assert len(matches) == 2
    assert any("foo bar" in m["text"] for m in matches)
    assert any("baz qux" in m["text"] for m in matches)


@pytest.mark.asyncio
async def test_concurrent_writes_serialize_via_lock() -> None:
    fs = InMemoryBackend()

    async def writer(i: int) -> None:
        await fs.write("/shared.txt", f"value-{i}")

    await asyncio.gather(*(writer(i) for i in range(5)))
    final = (await fs.read("/shared.txt")).decode()
    # Last write wins, but no corruption.
    assert final.startswith("value-")


@pytest.mark.asyncio
async def test_read_missing_path_raises() -> None:
    fs = InMemoryBackend()
    with pytest.raises(FileNotFoundError):
        await fs.read("/missing")
