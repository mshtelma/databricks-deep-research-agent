"""GTE embedding client for hybrid search in ReAct synthesis.

Computes text embeddings using Databricks GTE endpoint for semantic
similarity search in the evidence registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from openai import AsyncOpenAI

from deep_research.core.databricks_auth import get_databricks_auth
from deep_research.core.logging_utils import get_logger

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

        Uses centralized DatabricksAuth for authentication which supports:
        1. Direct token: DATABRICKS_TOKEN environment variable
        2. Profile-based OAuth: DATABRICKS_CONFIG_PROFILE from ~/.databrickscfg
        3. Automatic OAuth: Databricks Apps environment (service principal)

        Args:
            endpoint_name: Databricks model serving endpoint name for GTE.
        """
        self._endpoint = endpoint_name

        # Use centralized auth (same as LLMClient)
        self._auth = get_databricks_auth()
        self._current_token = self._auth.get_token()
        self._base_url = self._auth.get_base_url()

        # Initialize OpenAI client for Databricks embeddings
        self._client = AsyncOpenAI(
            api_key=self._current_token,
            base_url=self._base_url,
        )

        logger.info(
            "GTE_EMBEDDER_INITIALIZED",
            endpoint=endpoint_name,
            auth_mode=self._auth.auth_mode,
        )

    def _ensure_fresh_client(self) -> None:
        """Ensure the OpenAI client has a fresh OAuth token.

        For OAuth-based auth, checks if token is expired and refreshes if needed.
        For direct token auth, this is a no-op.
        """
        if not self._auth.is_oauth:
            return

        token = self._auth.get_token()

        if token != self._current_token:
            logger.info("GTE_TOKEN_REFRESHED", auth_mode=self._auth.auth_mode)
            self._current_token = token
            self._client = AsyncOpenAI(
                api_key=token,
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
