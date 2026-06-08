"""Concurrent write safety tests for :class:`InMemoryBackend`."""

from __future__ import annotations

import asyncio

import pytest

from databricks_deep_research.api import InMemoryBackend


@pytest.mark.asyncio
async def test_parallel_writes_to_same_path_dont_corrupt() -> None:
    """Per-path lock keeps writes serialized; final state is one of the inputs."""
    fs = InMemoryBackend()
    payloads = [f"line-{i}" * 10 for i in range(20)]

    await asyncio.gather(*(fs.write("/shared.txt", p) for p in payloads))

    final = (await fs.read("/shared.txt")).decode()
    assert final in payloads


@pytest.mark.asyncio
async def test_parallel_writes_to_different_paths_independent() -> None:
    fs = InMemoryBackend()

    async def writer(i: int) -> None:
        await fs.write(f"/path-{i}.txt", str(i))

    await asyncio.gather(*(writer(i) for i in range(10)))
    for i in range(10):
        assert (await fs.read(f"/path-{i}.txt")).decode() == str(i)


@pytest.mark.asyncio
async def test_concurrent_edits_on_same_path_serialize() -> None:
    fs = InMemoryBackend()
    await fs.write("/shared.txt", "AAAA")

    # Two edits both replace AAAA → first wins, second raises (no AAAA left).
    async def edit_to_b() -> None:
        await fs.edit("/shared.txt", "AAAA", "BBBB", unique=False)

    async def edit_to_c() -> None:
        await fs.edit("/shared.txt", "AAAA", "CCCC", unique=False)

    results = await asyncio.gather(edit_to_b(), edit_to_c(), return_exceptions=True)
    # Exactly one succeeded, one raised ValueError("not found").
    successes = [r for r in results if not isinstance(r, Exception)]
    failures = [r for r in results if isinstance(r, Exception)]
    assert len(successes) == 1
    assert len(failures) == 1
    final = (await fs.read("/shared.txt")).decode()
    assert final in ("BBBB", "CCCC")
