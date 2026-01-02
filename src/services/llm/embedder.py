"""GTE embedding client for hybrid search in ReAct synthesis.

Computes text embeddings using Databricks GTE endpoint for semantic
similarity search in the evidence registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from openai import AsyncOpenAI

from src.core.config import get_settings
from src.core.logging_utils import get_logger
from src.services.llm.auth import LLMCredentialProvider

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

# Default GTE endpoint name (configurable)
DEFAULT_GTE_ENDPOINT = "databricks-gte-large"


class GteEmbedder:
    """Compute text embeddings via Databricks GTE endpoint.

    Uses the same authentication pattern as LLMClient (profile-based OAuth
    or direct token). Embeddings are used for semantic search in the
    evidence registry alongside BM25 keyword search.

    Example:
        embedder = GteEmbedder()
        # Single text
        vec = await embedder.embed("machine learning achieves 95% accuracy")
        # Batch (more efficient)
        vecs = await embedder.embed_batch(["text1", "text2", "text3"])
    """

    def __init__(self, endpoint_name: str = DEFAULT_GTE_ENDPOINT):
        """Initialize the GTE embedder.

        Args:
            endpoint_name: Databricks model serving endpoint name for GTE.
        """
        settings = get_settings()
        self._endpoint = endpoint_name

        # Auth mode tracking (same pattern as LLMClient)
        self._credential_provider: LLMCredentialProvider | None = None
        self._current_token: str | None = None

        # Get token - either from env or from WorkspaceClient
        token = settings.databricks_token
        self._base_url = f"{settings.databricks_host}/serving-endpoints"

        if not token and settings.databricks_config_profile:
            # Profile-based OAuth auth with refresh support
            self._credential_provider = LLMCredentialProvider(
                profile=settings.databricks_config_profile
            )
            credential = self._credential_provider.get_credential()
            token = credential.token
            self._base_url = self._credential_provider.get_base_url()

        if not token:
            raise ValueError("No Databricks token available for embedder")

        self._current_token = token

        # Initialize OpenAI client for Databricks embeddings
        self._client = AsyncOpenAI(
            api_key=token,
            base_url=self._base_url,
        )

        logger.info(
            "GTE_EMBEDDER_INITIALIZED",
            endpoint=endpoint_name,
        )

    def _ensure_fresh_client(self) -> None:
        """Ensure the OpenAI client has a fresh OAuth token.

        For profile-based OAuth auth, checks if token is expired and
        refreshes if needed. For direct token auth, this is a no-op.
        """
        if self._credential_provider is None:
            return

        credential = self._credential_provider.get_credential()

        if credential.token != self._current_token:
            logger.info("GTE_TOKEN_REFRESHED", provider="oauth")
            self._current_token = credential.token
            self._client = AsyncOpenAI(
                api_key=credential.token,
                base_url=self._base_url,
            )

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 50,
    ) -> NDArray[np.float32]:
        """Embed multiple texts in batches.

        Args:
            texts: List of texts to embed.
            batch_size: Maximum texts per API call (default 50).

        Returns:
            2D numpy array of shape (n_texts, embed_dim).
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, 0)

        self._ensure_fresh_client()

        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            try:
                response = await self._client.embeddings.create(
                    model=self._endpoint,
                    input=batch,
                )

                # Sort by index to ensure correct order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                batch_embeddings = [item.embedding for item in sorted_data]
                all_embeddings.extend(batch_embeddings)

                logger.debug(
                    "GTE_BATCH_EMBEDDED",
                    batch_start=i,
                    batch_size=len(batch),
                    embed_dim=len(batch_embeddings[0]) if batch_embeddings else 0,
                )

            except Exception as e:
                logger.error(
                    "GTE_EMBED_ERROR",
                    batch_start=i,
                    batch_size=len(batch),
                    error=str(e)[:200],
                )
                raise

        return np.array(all_embeddings, dtype=np.float32)

    async def embed(self, text: str) -> NDArray[np.float32]:
        """Embed a single text.

        Args:
            text: Text to embed.

        Returns:
            1D numpy array of shape (embed_dim,).
        """
        result = await self.embed_batch([text])
        return result[0] if len(result) > 0 else np.array([], dtype=np.float32)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()


# Singleton instance for reuse
_embedder_instance: GteEmbedder | None = None


def get_embedder(endpoint_name: str = DEFAULT_GTE_ENDPOINT) -> GteEmbedder:
    """Get or create a singleton GTE embedder instance.

    Args:
        endpoint_name: Databricks endpoint name for GTE model.

    Returns:
        GteEmbedder instance.
    """
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = GteEmbedder(endpoint_name)
    return _embedder_instance
