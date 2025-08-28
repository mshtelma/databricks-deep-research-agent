"""
Databricks embedding client for semantic similarity and deduplication.

This module provides a client for Databricks serving endpoints to generate
embeddings for text content, optimized for deduplication and similarity analysis.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional
import requests
import json

from .logging import get_logger
from .exceptions import ResearchAgentError
from .utils import retry_with_exponential_backoff, chunk_list

logger = get_logger(__name__)


class DatabricksEmbeddingError(ResearchAgentError):
    """Exception for Databricks embedding service errors."""
    pass


class DatabricksEmbeddingClient:
    """
    Client for Databricks embedding model serving endpoints.
    
    Provides efficient batch processing, rate limiting, and error handling
    for generating embeddings used in semantic deduplication.
    """
    
    def __init__(
        self,
        endpoint_name: str = "databricks-bge-large-en",
        workspace_url: Optional[str] = None,
        token: Optional[str] = None,
        profile: Optional[str] = None,
        batch_size: int = 50,
        max_retries: int = 3,
        timeout_seconds: int = 30
    ):
        """
        Initialize Databricks embedding client.
        
        Args:
            endpoint_name: Name of the embedding model endpoint
            workspace_url: Databricks workspace URL (auto-detected if None)
            token: Access token (auto-detected if None)
            profile: Databricks CLI profile name (auto-detected if None)
            batch_size: Maximum texts per batch request
            max_retries: Number of retry attempts
            timeout_seconds: Request timeout
        """
        self.endpoint_name = endpoint_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.profile = profile
        
        # Check if running inside Databricks
        self._inside_databricks = self._is_inside_databricks()
        
        # Auto-detect Databricks environment (still needed for backwards compatibility)
        self.workspace_url = workspace_url or self._get_workspace_url()
        self.token = token or self._get_token()
        
        # Build endpoint URL (used for logging and external access)
        self.endpoint_url = f"{self._ensure_url_scheme(self.workspace_url)}/serving-endpoints/{endpoint_name}/invocations"
        
        # Initialize REST session only when outside Databricks AND not using profile
        if not self._inside_databricks and not self.profile:
            # Session for connection pooling (external access with explicit token)
            if self.token:
                self.session = requests.Session()
                self.session.headers.update({
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json"
                })
            else:
                self.session = None
        else:
            # Inside Databricks or using profile - we'll use SDK, no REST session needed
            self.session = None
        
        # Determine which mode will be used for embeddings
        if self._inside_databricks:
            mode = "SDK (inside workspace)"
        elif self.profile:
            mode = f"SDK (profile: {self.profile})"
        else:
            mode = "REST API (external)"
        
        logger.info(
            "Initialized Databricks embedding client",
            endpoint=endpoint_name,
            batch_size=batch_size,
            workspace_url=self.workspace_url[:50] + "..." if self.workspace_url else None,
            mode=mode
        )
    
    def _get_workspace_url(self) -> str:
        """Get workspace URL from environment or context."""
        import os
        
        # Try environment variable first
        workspace_url = os.getenv("DATABRICKS_HOST")
        if workspace_url:
            return self._ensure_url_scheme(workspace_url.rstrip("/"))
        
        # Try to get from SDK profile (if profile is available)
        if self.profile:
            try:
                from databricks.sdk import WorkspaceClient
                client = WorkspaceClient(profile=self.profile)
                # The config.host contains the workspace URL
                workspace_url = client.config.host
                if workspace_url:
                    logger.debug(f"Got workspace URL from profile '{self.profile}': {workspace_url}")
                    return self._ensure_url_scheme(workspace_url.rstrip("/"))
            except (ImportError, Exception) as e:
                logger.debug(f"Could not get workspace URL from profile '{self.profile}': {e}")
                pass
        
        # Try to detect from databricks context
        try:
            from databricks.connect import DatabricksSession
            spark = DatabricksSession.builder.getOrCreate()
            workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
            return self._ensure_url_scheme(workspace_url)
        except (ImportError, Exception) as e:
            # This can fail with "Cluster id or serverless are required" when running locally
            # without proper Databricks Connect configuration
            logger.debug(f"Could not get workspace URL from Databricks Connect: {e}")
            pass
        
        # Try notebook context
        try:
            import os
            workspace_url = os.environ.get("DATABRICKS_WORKSPACE_URL")
            if workspace_url:
                return self._ensure_url_scheme(workspace_url.rstrip("/"))
        except Exception:
            pass
        
        raise DatabricksEmbeddingError(
            "Could not auto-detect Databricks workspace URL. "
            "Please provide it explicitly or set DATABRICKS_HOST environment variable."
        )
    
    def _ensure_url_scheme(self, url: str) -> str:
        """
        Ensure URL has https:// scheme.
        
        Args:
            url: URL that may or may not have a scheme
            
        Returns:
            URL with https:// scheme
        """
        if not url:
            return url
        
        # If URL already has a scheme, return as-is
        if url.startswith(("http://", "https://")):
            return url
        
        # Add https:// scheme
        return f"https://{url}"
    
    def _get_token(self) -> str:
        """Get access token from environment or context."""
        import os
        
        # Try environment variable first
        token = os.getenv("DATABRICKS_TOKEN")
        if token:
            return token
        
        # In model serving, token should be in environment
        # Don't try to use SDK as it will fail
        if self._is_model_serving_environment():
            # In model serving, authentication is handled automatically
            # Return empty string to avoid errors, REST API will use proper auth
            logger.debug("Model serving environment - authentication handled by platform")
            return ""
        
        # Try to get from databricks CLI profile (only outside model serving)  
        try:
            from databricks.sdk import WorkspaceClient
            if self.profile:
                client = WorkspaceClient(profile=self.profile)
                logger.debug(f"Using profile '{self.profile}' for token retrieval")
            else:
                client = WorkspaceClient()
                logger.debug("Using default profile for token retrieval")
            
            # The SDK handles token automatically
            token = client.config.token
            if token:
                logger.debug(f"Successfully retrieved token from SDK (length: {len(token)})")
                return token
            else:
                # When using databricks-cli auth, token might be None but SDK still works
                # In this case, return a placeholder since we'll use SDK for API calls anyway
                if client.config.auth_type == "databricks-cli" and self.profile:
                    logger.debug(f"Profile '{self.profile}' uses databricks-cli auth (token not stored in config)")
                    return None  # Will use SDK directly instead of REST
                else:
                    logger.warning("SDK client initialized but token is None/empty and auth type is not databricks-cli")
                
        except (ImportError, Exception) as e:
            profile_info = f" with profile '{self.profile}'" if self.profile else " with default profile"
            logger.debug(f"Could not get token from SDK{profile_info}: {e}")
            pass
        
        raise DatabricksEmbeddingError(
            "Could not auto-detect Databricks token. "
            "Please provide it explicitly or set DATABRICKS_TOKEN environment variable."
        )
    
    def _is_model_serving_environment(self) -> bool:
        """
        Check if running in a model serving environment.
        
        Model serving environments should use REST API instead of SDK
        because they don't have access to cluster_id or serverless compute.
        
        Returns:
            True if running in model serving, False otherwise
        """
        import os
        
        # Check for model serving specific environment variables
        serving_indicators = [
            "DATABRICKS_SERVING_ENDPOINT",
            "DATABRICKS_MODEL_SERVING", 
            "DATABRICKS_ENDPOINT_NAME",
            "MLFLOW_SERVING_PORT",
            "SERVING_PORT"
        ]
        
        for indicator in serving_indicators:
            if os.getenv(indicator):
                logger.debug(f"Model serving environment detected via {indicator}")
                return True
                
        return False
    
    def _is_inside_databricks(self) -> bool:
        """
        Check if running inside Databricks environment (notebook/cluster).
        
        Note: Model serving environments are excluded as they should use REST API.
        
        Returns:
            True if running inside Databricks workspace (notebook/cluster), False otherwise
        """
        import os
        
        # First check if we're in model serving - if so, return False
        # Model serving should use REST API, not SDK
        if self._is_model_serving_environment():
            logger.debug("Model serving detected - will use REST API instead of SDK")
            return False
        
        # Check for Databricks runtime environment variable
        if os.getenv("DATABRICKS_RUNTIME_VERSION") is not None:
            return True
            
        # Check for Databricks driver directory
        if os.path.exists("/databricks/driver"):
            return True
            
        # Check for Databricks workspace path
        if os.path.exists("/databricks") or os.getcwd().startswith("/Workspace"):
            return True
            
        return False
    
    @retry_with_exponential_backoff(max_retries=3, exceptions=(requests.RequestException,))
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
            
        Raises:
            DatabricksEmbeddingError: If embedding generation fails
        """
        if not texts:
            return np.array([])
        
        start_time = time.time()
        
        try:
            # Process in batches to avoid overwhelming the endpoint
            all_embeddings = []
            
            for batch in chunk_list(texts, self.batch_size):
                batch_embeddings = self._get_batch_embeddings(batch)
                all_embeddings.extend(batch_embeddings)
            
            # Convert to numpy array and normalize
            embeddings_array = np.array(all_embeddings)
            normalized_embeddings = self._normalize_embeddings(embeddings_array)
            
            processing_time = time.time() - start_time
            
            logger.info(
                "Generated embeddings successfully",
                text_count=len(texts),
                embedding_dim=normalized_embeddings.shape[1] if normalized_embeddings.size > 0 else 0,
                processing_time=processing_time,
                batches=len(list(chunk_list(texts, self.batch_size)))
            )
            
            return normalized_embeddings
            
        except Exception as e:
            logger.error(
                "Failed to generate embeddings",
                error=str(e),
                text_count=len(texts),
                endpoint=self.endpoint_name
            )
            raise DatabricksEmbeddingError(f"Embedding generation failed: {e}")
    
    def _get_batch_embeddings_sdk(self, batch_texts: List[str]) -> List[List[float]]:
        """
        Get embeddings using Databricks SDK (for inside workspace).
        
        Args:
            batch_texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            DatabricksEmbeddingError: If embedding generation fails
        """
        try:
            from databricks.sdk import WorkspaceClient
            
            if self.profile:
                w = WorkspaceClient(profile=self.profile)
                logger.debug(f"Using profile '{self.profile}' for embeddings")
            else:
                w = WorkspaceClient()
                logger.debug("Using default profile for embeddings")
            
            # Try batch processing first
            try:
                response = w.serving_endpoints.query(
                    name=self.endpoint_name,
                    input=batch_texts
                )
                
                # Extract embeddings from SDK response
                embeddings = []
                if hasattr(response, 'data') and response.data:
                    for item in response.data:
                        if hasattr(item, 'embedding'):
                            embeddings.append(item.embedding)
                        else:
                            # Handle case where item is the embedding directly
                            embeddings.append(item)
                else:
                    # Try alternative response format
                    if hasattr(response, 'outputs'):
                        embeddings = response.outputs
                    elif hasattr(response, 'predictions'):
                        embeddings = response.predictions
                    else:
                        raise DatabricksEmbeddingError(f"Unexpected SDK response format: {response}")
                        
                if not embeddings:
                    raise DatabricksEmbeddingError("No embeddings found in SDK response")
                    
                return embeddings
                
            except Exception as batch_error:
                logger.warning(
                    "Batch embedding failed, trying individual processing",
                    batch_error=str(batch_error),
                    batch_size=len(batch_texts)
                )
                
                # Fallback: process each text individually
                embeddings = []
                for text in batch_texts:
                    try:
                        individual_response = w.serving_endpoints.query(
                            name=self.endpoint_name,
                            input=[text]  # Single item as list
                        )
                        
                        # Extract single embedding
                        if hasattr(individual_response, 'data') and individual_response.data:
                            if hasattr(individual_response.data[0], 'embedding'):
                                embeddings.append(individual_response.data[0].embedding)
                            else:
                                embeddings.append(individual_response.data[0])
                        elif hasattr(individual_response, 'outputs'):
                            embeddings.append(individual_response.outputs[0])
                        elif hasattr(individual_response, 'predictions'):
                            embeddings.append(individual_response.predictions[0])
                        else:
                            raise DatabricksEmbeddingError(f"Unexpected individual response format: {individual_response}")
                    except Exception as individual_error:
                        logger.error(
                            "Individual embedding failed",
                            text_preview=text[:50] + "..." if len(text) > 50 else text,
                            error=str(individual_error)
                        )
                        raise DatabricksEmbeddingError(f"Both batch and individual embedding failed: {individual_error}")
                
                if not embeddings:
                    raise DatabricksEmbeddingError("No embeddings generated from individual processing")
                
                return embeddings
            
        except ImportError as e:
            raise DatabricksEmbeddingError(f"Databricks SDK not available: {e}")
        except Exception as e:
            logger.error(
                "SDK embedding request failed",
                error=str(e),
                endpoint=self.endpoint_name,
                batch_size=len(batch_texts)
            )
            raise DatabricksEmbeddingError(f"SDK embedding request failed: {e}")
    
    def _get_batch_embeddings_rest(self, batch_texts: List[str]) -> List[List[float]]:
        """Get embeddings for a single batch of texts using REST API (for external access)."""
        # Prepare request payload according to Databricks API specification
        payload = {
            "input": batch_texts
        }
        
        try:
            response = self.session.post(
                self.endpoint_url,
                json=payload,
                timeout=self.timeout_seconds
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract embeddings from Databricks response format
            if "data" in result:
                # Standard Databricks embedding response format
                embeddings = []
                for item in result["data"]:
                    if isinstance(item, dict) and "embedding" in item:
                        embeddings.append(item["embedding"])
                    elif isinstance(item, list):
                        # Direct embedding array
                        embeddings.append(item)
                    else:
                        raise DatabricksEmbeddingError(f"Unexpected data item format: {item}")
                
                if not embeddings:
                    raise DatabricksEmbeddingError("No embeddings found in response data")
                return embeddings
            elif "predictions" in result:
                # Legacy format support
                return result["predictions"]
            elif "outputs" in result:
                # Alternative format support  
                return result["outputs"]
            else:
                # Handle unexpected formats
                raise DatabricksEmbeddingError(f"Unexpected response format: {result}")
                
        except requests.exceptions.HTTPError as e:
            if response.status_code == 403:
                logger.error(
                    "Databricks embedding endpoint authentication failed",
                    error=str(e),
                    endpoint=self.endpoint_name,
                    batch_size=len(batch_texts),
                    status_code=response.status_code
                )
                raise DatabricksEmbeddingError(
                    f"Authentication failed for endpoint '{self.endpoint_name}'. "
                    f"Please check your DATABRICKS_TOKEN and ensure you have access to the embedding endpoint. "
                    f"Status: {response.status_code}, Error: {e}"
                )
            elif response.status_code == 404:
                logger.error(
                    "Databricks embedding endpoint not found",
                    error=str(e),
                    endpoint=self.endpoint_name,
                    endpoint_url=self.endpoint_url,
                    status_code=response.status_code
                )
                raise DatabricksEmbeddingError(
                    f"Embedding endpoint '{self.endpoint_name}' not found. "
                    f"Please verify the endpoint name is correct and the model is deployed. "
                    f"URL: {self.endpoint_url}, Error: {e}"
                )
            else:
                logger.error(
                    "Embedding request failed",
                    error=str(e),
                    endpoint=self.endpoint_name,
                    batch_size=len(batch_texts),
                    status_code=response.status_code
                )
                raise DatabricksEmbeddingError(f"HTTP error {response.status_code}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(
                "Embedding request failed",
                error=str(e),
                endpoint=self.endpoint_name,
                batch_size=len(batch_texts)
            )
            raise DatabricksEmbeddingError(f"Request failed: {e}")
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse embedding response",
                error=str(e),
                endpoint=self.endpoint_name
            )
            raise DatabricksEmbeddingError(f"Invalid JSON response: {e}")
    
    def _get_batch_embeddings(self, batch_texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts using the appropriate method.
        
        Routes to SDK method when inside Databricks, REST method when outside.
        
        Args:
            batch_texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            DatabricksEmbeddingError: If embedding generation fails
        """
        # Use SDK if inside Databricks OR if we have a profile configured
        if self._inside_databricks or self.profile:
            logger.debug(f"Using Databricks SDK for embeddings (profile: {self.profile or 'workspace default'})")
            return self._get_batch_embeddings_sdk(batch_texts)
        else:
            logger.debug("Using REST API for embeddings (external access)")
            return self._get_batch_embeddings_rest(batch_texts)
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length for cosine similarity.
        
        Args:
            embeddings: Raw embeddings array
            
        Returns:
            Normalized embeddings array
        """
        if embeddings.size == 0:
            return embeddings
        
        # Calculate L2 norm for each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        
        # Normalize
        normalized = embeddings / norms
        
        return normalized
    
    def compute_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity matrix with shape (len(embeddings1), len(embeddings2))
        """
        if embeddings1.size == 0 or embeddings2.size == 0:
            return np.array([])
        
        # Ensure embeddings are normalized
        embeddings1_norm = self._normalize_embeddings(embeddings1)
        embeddings2_norm = self._normalize_embeddings(embeddings2)
        
        # Compute cosine similarity via dot product (since embeddings are normalized)
        similarity_matrix = np.dot(embeddings1_norm, embeddings2_norm.T)
        
        return similarity_matrix
    
    def find_similar_texts(
        self, 
        query_text: str, 
        candidate_texts: List[str], 
        threshold: float = 0.8,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find texts similar to a query text.
        
        Args:
            query_text: Text to search for
            candidate_texts: List of candidate texts to search in
            threshold: Minimum similarity threshold
            top_k: Maximum number of results to return
            
        Returns:
            List of similar texts with similarity scores
        """
        if not candidate_texts:
            return []
        
        # Get embeddings
        query_embedding = self.get_embeddings([query_text])
        candidate_embeddings = self.get_embeddings(candidate_texts)
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, candidate_embeddings)[0]
        
        # Find similar texts above threshold
        similar_indices = np.where(similarities >= threshold)[0]
        
        # Create results with scores
        results = []
        for idx in similar_indices:
            results.append({
                "text": candidate_texts[idx],
                "similarity": float(similarities[idx]),
                "index": int(idx)
            })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Apply top_k limit
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics and configuration."""
        return {
            "endpoint_name": self.endpoint_name,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "workspace_url": self.workspace_url,
        }