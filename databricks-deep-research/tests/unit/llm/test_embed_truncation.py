"""Per-doc embedding truncation (Wave 1.5 WIN B).

``FrameworkLLMClient.embed`` must cap each input document at
``MAX_EMBED_CHARS`` before sending it to the embedding endpoint. An
over-long document otherwise blows the embedding model's token limit and
fails the whole batch; truncating the head is strictly safer. Short docs
must be left untouched.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from databricks_deep_research.llm.client import MAX_EMBED_CHARS, FrameworkLLMClient


def _make_client() -> tuple[FrameworkLLMClient, AsyncMock]:
    """Build a client whose embeddings endpoint is a recording mock."""
    fake_openai = MagicMock()
    create = AsyncMock(
        return_value=MagicMock(data=[MagicMock(embedding=[0.0, 1.0])]),
    )
    fake_openai.embeddings.create = create
    client = FrameworkLLMClient(
        openai_client=fake_openai,
        model_mapping={"complex": "x"},
        embedding_model="databricks-bge-large-en",
    )
    return client, create


@pytest.mark.asyncio
async def test_embed_truncates_overlong_doc() -> None:
    client, create = _make_client()
    # Make the mock return one vector per input so embed() does not IndexError.
    over_long = "a" * (MAX_EMBED_CHARS + 5_000)
    create.return_value = MagicMock(data=[MagicMock(embedding=[0.0])])

    await client.embed([over_long])

    sent = create.await_args.kwargs["input"]
    assert len(sent) == 1
    assert len(sent[0]) == MAX_EMBED_CHARS  # truncated to the cap


@pytest.mark.asyncio
async def test_embed_leaves_short_docs_untouched() -> None:
    client, create = _make_client()
    short = "a short document well under the cap"
    create.return_value = MagicMock(data=[MagicMock(embedding=[0.0])])

    await client.embed([short])

    sent = create.await_args.kwargs["input"]
    assert sent == [short]  # byte-identical, no truncation


@pytest.mark.asyncio
async def test_embed_mixed_batch_truncates_only_overlong() -> None:
    client, create = _make_client()
    short = "tiny"
    over_long = "b" * (MAX_EMBED_CHARS + 1)
    create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.0]), MagicMock(embedding=[1.0])],
    )

    await client.embed([short, over_long])

    sent = create.await_args.kwargs["input"]
    assert sent[0] == short
    assert len(sent[1]) == MAX_EMBED_CHARS
