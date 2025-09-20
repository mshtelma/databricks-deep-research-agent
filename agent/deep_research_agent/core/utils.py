"""
Common utility functions for the research agent.

This module provides reusable utility functions for common operations
like retries, validation, circuit breaker, and URL handling.
"""

import time
import json
import hashlib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from functools import wraps
from urllib.parse import urlparse
from datetime import datetime, timedelta

from .exceptions import (
    RetryExhaustedError, 
    CircuitBreakerError, 
    ValidationError, 
    TimeoutError
)
from .logging import get_logger

T = TypeVar('T')
logger = get_logger(__name__)


class CircuitBreaker:
    """Circuit breaker implementation for external service calls."""
    
    def __init__(
        self, 
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures to open circuit
            recovery_timeout: Seconds to wait before trying again
            expected_exception: Exception type that triggers circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Call function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_multiplier: Multiplier for exponential backoff
        exceptions: Exception types to retry on
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries",
                            error=e,
                            function=func.__name__,
                            attempt=attempt + 1
                        )
                        raise RetryExhaustedError(
                            f"Function {func.__name__} failed after {max_retries} retries: {e}"
                        )
                    
                    logger.warning(
                        f"Function {func.__name__} failed, retrying in {delay}s",
                        function=func.__name__,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e)
                    )
                    
                    time.sleep(delay)
                    delay = min(delay * backoff_multiplier, max_delay)
            
            # This should never be reached
            raise last_exception
        
        return wrapper
    return decorator


def validate_url(url: str) -> bool:
    """
    Validate if a string is a valid URL.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid URL, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:250] + "..."
    
    return sanitized


def extract_domain(url: str) -> Optional[str]:
    """
    Extract domain from URL.
    
    Args:
        url: URL string
        
    Returns:
        Domain string or None if invalid URL
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return None


def generate_hash(content: str, algorithm: str = "md5") -> str:
    """
    Generate hash for content.
    
    Args:
        content: Content to hash
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hex digest of hash
    """
    hasher = hashlib.new(algorithm)
    hasher.update(content.encode('utf-8'))
    return hasher.hexdigest()


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with fallback.
    
    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"Failed to parse JSON: {e}", json_string=json_str[:100])
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    Safely serialize object to JSON string.
    
    Args:
        obj: Object to serialize
        default: Default string if serialization fails
        
    Returns:
        JSON string or default
    """
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.debug(f"Failed to serialize to JSON: {e}", object_type=type(obj).__name__)
        return default


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
    """
    Validate that required fields are present in data.
    
    Args:
        data: Data dictionary to validate
        required_fields: List of required field names
        
    Returns:
        List of missing field names
    """
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    return missing_fields


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def parse_search_keywords(text: str) -> List[str]:
    """
    Parse search keywords from text.
    
    Args:
        text: Text to extract keywords from
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction - could be enhanced with NLP
    # Remove common stop words and extract meaningful terms
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "will", "with", "the", "this", "but", "they", "have",
        "had", "what", "said", "each", "which", "she", "do", "how", "their",
        "if", "up", "out", "many", "then", "them", "these", "so", "some",
        "her", "would", "make", "like", "into", "him", "time", "two", "more",
        "go", "no", "way", "could", "my", "than", "first", "been", "call",
        "who", "oil", "sit", "now", "find", "down", "day", "did", "get",
        "come", "made", "may", "part"
    }
    
    # Extract words and filter
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    keywords = [word for word in words if word not in stop_words]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    return unique_keywords[:10]  # Return top 10 keywords


def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries with later ones taking precedence.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def timeout_after(seconds: int):
    """
    Decorator to add timeout to function calls.
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Reset alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


class URLResolver:
    """URL resolution and shortening service for token efficiency."""
    
    def __init__(self, base_url: str = "https://search.internal/id/"):
        """
        Initialize URL resolver.
        
        Args:
            base_url: Base URL for shortened links
        """
        self.base_url = base_url
        self.url_to_short: Dict[str, str] = {}
        self.short_to_url: Dict[str, str] = {}
        self._counter = 0
    
    def resolve_urls(self, urls: List[str], prefix_id: Optional[str] = None) -> Dict[str, str]:
        """
        Create mapping of long URLs to short URLs.
        
        Args:
            urls: List of URLs to resolve
            prefix_id: Optional prefix for unique identification
            
        Returns:
            Dictionary mapping original URL to short URL
        """
        resolved_map = {}
        
        for idx, url in enumerate(urls):
            if url not in self.url_to_short:
                # Generate short URL
                if prefix_id:
                    short_id = f"{prefix_id}-{idx}"
                else:
                    short_id = str(self._counter)
                    self._counter += 1
                
                short_url = f"{self.base_url}{short_id}"
                
                # Store bidirectional mapping
                self.url_to_short[url] = short_url
                self.short_to_url[short_url] = url
            
            resolved_map[url] = self.url_to_short[url]
        
        return resolved_map
    
    def get_original_url(self, short_url: str) -> Optional[str]:
        """
        Get original URL from short URL.
        
        Args:
            short_url: Short URL to resolve
            
        Returns:
            Original URL or None if not found
        """
        return self.short_to_url.get(short_url)
    
    def replace_urls_in_text(self, text: str, url_mapping: Dict[str, str]) -> str:
        """
        Replace long URLs in text with short URLs.
        
        Args:
            text: Text containing URLs
            url_mapping: Mapping of original URL to short URL
            
        Returns:
            Text with URLs replaced
        """
        modified_text = text
        
        # Sort by URL length (longest first) to avoid partial replacements
        sorted_urls = sorted(url_mapping.items(), key=lambda x: len(x[0]), reverse=True)
        
        for original_url, short_url in sorted_urls:
            modified_text = modified_text.replace(original_url, short_url)
        
        return modified_text
    
    def restore_urls_in_text(self, text: str) -> str:
        """
        Restore original URLs in text from short URLs.
        
        Args:
            text: Text containing short URLs
            
        Returns:
            Text with original URLs restored
        """
        modified_text = text
        
        # Sort by short URL length (longest first) to avoid partial replacements
        sorted_urls = sorted(self.short_to_url.items(), key=lambda x: len(x[0]), reverse=True)
        
        for short_url, original_url in sorted_urls:
            modified_text = modified_text.replace(short_url, original_url)
        
        return modified_text
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get URL resolver statistics.
        
        Returns:
            Dictionary with resolver statistics
        """
        return {
            "total_urls": len(self.url_to_short),
            "counter": self._counter,
            "average_original_length": sum(len(url) for url in self.url_to_short.keys()) / max(1, len(self.url_to_short)),
            "average_short_length": sum(len(url) for url in self.short_to_url.keys()) / max(1, len(self.short_to_url)),
            "token_savings_estimate": self._calculate_token_savings()
        }
    
    def _calculate_token_savings(self) -> float:
        """Calculate estimated token savings from URL shortening."""
        if not self.url_to_short:
            return 0.0
        
        original_chars = sum(len(url) for url in self.url_to_short.keys())
        short_chars = sum(len(url) for url in self.short_to_url.keys())
        
        # Rough estimation: 4 chars per token
        original_tokens = original_chars / 4
        short_tokens = short_chars / 4
        
        return max(0, original_tokens - short_tokens)


def insert_citation_markers(text: str, citations: List[Dict[str, Any]]) -> str:
    """
    Insert citation markers into text based on start and end indices.
    
    Args:
        text: Original text string
        citations: List of citation dictionaries with indices and segments
        
    Returns:
        Text with citation markers inserted
    """
    # Sort citations by end_index in descending order to preserve indices
    sorted_citations = sorted(
        citations, 
        key=lambda c: (c.get("end_index", 0), c.get("start_index", 0)), 
        reverse=True
    )
    
    modified_text = text
    for citation in sorted_citations:
        end_idx = citation.get("end_index", 0)
        segments = citation.get("segments", [])
        
        # Build citation marker
        marker = ""
        for segment in segments:
            label = segment.get("label", "Source")
            url = segment.get("short_url", segment.get("url", "#"))
            marker += f" [{label}]({url})"
        
        # Insert citation marker at end index
        modified_text = modified_text[:end_idx] + marker + modified_text[end_idx:]
    
    return modified_text


class SecretResolver:
    """
    Centralized secret resolution for all tools and configurations.
    
    This class provides a consistent mechanism for resolving secrets from:
    1. Databricks dbutils.secrets (notebooks/jobs)
    2. MLflow get_secret() (model serving)
    3. Environment variables (all contexts)
    
    Includes caching to avoid repeated resolution attempts.
    """
    
    def __init__(self):
        """Initialize the secret resolver with empty cache."""
        self._cache: Dict[str, Any] = {}
        self._failed_secrets: set = set()  # Track failed secrets to avoid retries
        
    def resolve_secret(self, value: Any) -> Any:
        """
        Resolve a secret reference to its actual value.
        
        Handles {{secrets/scope/key}} syntax and returns the resolved value.
        Uses caching to avoid repeated resolution attempts.
        
        Args:
            value: Value to resolve (may be a secret reference or regular value)
            
        Returns:
            Resolved value or original if not a secret reference
        """
        if not isinstance(value, str) or not value.startswith("{{secrets/"):
            return value
        
        # Check cache first
        if value in self._cache:
            cached_value = self._cache[value]
            if cached_value is not None:
                logger.debug(f"Using cached secret resolution for {value}")
                return cached_value
        
        # Check if we've already failed to resolve this secret
        if value in self._failed_secrets:
            logger.debug(f"Skipping previously failed secret: {value}")
            return value
        
        # Parse secret reference
        import re
        match = re.match(r'\{\{secrets/([^/]+)/([^}]+)\}\}', value)
        if not match:
            logger.warning(f"Invalid secret format: {value}")
            self._failed_secrets.add(value)
            return value
        
        scope, key = match.groups()
        logger.debug(f"Attempting to resolve secret: scope={scope}, key={key}")
        
        # Try resolution methods in order
        resolved_value = self._try_resolve_secret(scope, key, value)
        
        # Cache the result
        if resolved_value != value:
            self._cache[value] = resolved_value
            logger.info(f"Successfully resolved and cached secret {key}")
        else:
            self._failed_secrets.add(value)
            self._cache[value] = None  # Cache failure to avoid repeated attempts
            
        return resolved_value
    
    def _try_resolve_secret(self, scope: str, key: str, original_value: str) -> Any:
        """Try all available secret resolution methods."""
        
        # Method 1: dbutils (Databricks notebooks/jobs)
        resolved = self._try_dbutils_resolution(scope, key)
        if resolved:
            logger.info(f"Successfully resolved secret {key} using dbutils")
            return resolved
        
        # Method 2: MLflow get_secret (model serving)
        resolved = self._try_mlflow_resolution(scope, key)
        if resolved:
            logger.info(f"Successfully resolved secret {key} using MLflow")
            return resolved
        
        # Method 3: Environment variables (all contexts)
        resolved = self._try_env_resolution(scope, key)
        if resolved:
            logger.info(f"Successfully resolved secret {key} using environment variable")
            return resolved
        
        # All methods failed
        self._log_resolution_failure(scope, key, original_value)
        return original_value
    
    def _try_dbutils_resolution(self, scope: str, key: str) -> Optional[str]:
        """Try to resolve secret using dbutils."""
        try:
            # Try direct import first
            import dbutils as db
            secret_val = db.secrets.get(scope=scope, key=key)
            if secret_val:
                return secret_val
        except (ImportError, NameError):
            logger.debug("Direct dbutils import not available")
        
        try:
            # Check global namespace
            import builtins
            if hasattr(builtins, '__dict__') and 'dbutils' in builtins.__dict__:
                dbutils = builtins.__dict__['dbutils']
                secret_val = dbutils.secrets.get(scope=scope, key=key)
                if secret_val:
                    return secret_val
        except Exception as e:
            logger.debug(f"Global dbutils access failed: {e}")
        
        try:
            # Frame-based search for dbutils
            import sys
            frame = sys._getframe(1)
            while frame and frame.f_back:
                if 'dbutils' in frame.f_globals:
                    dbutils = frame.f_globals['dbutils']
                    secret_val = dbutils.secrets.get(scope=scope, key=key)
                    if secret_val:
                        return secret_val
                frame = frame.f_back
        except Exception as e:
            logger.debug(f"Frame-based dbutils search failed: {e}")
        
        return None
    
    def _try_mlflow_resolution(self, scope: str, key: str) -> Optional[str]:
        """Try to resolve secret using MLflow."""
        try:
            import mlflow
            if hasattr(mlflow, 'get_secret'):
                # Try different formats for MLflow secret resolution
                for secret_format in [f"{scope}/{key}", f"{scope}.{key}", key]:
                    try:
                        secret_value = mlflow.get_secret(key=secret_format)
                        if secret_value:
                            logger.debug(f"MLflow secret resolved with format: {secret_format}")
                            return secret_value
                    except Exception as e:
                        logger.debug(f"Failed MLflow format {secret_format}: {e}")
        except Exception as e:
            logger.debug(f"MLflow secret resolution not available: {e}")
        
        return None
    
    def _try_env_resolution(self, scope: str, key: str) -> Optional[str]:
        """Try to resolve secret using environment variables."""
        import os
        
        # Try multiple environment variable formats
        env_patterns = [
            key,                          # BRAVE_API_KEY
            f"{scope}_{key}",            # msh_BRAVE_API_KEY
            f"{scope.upper()}_{key}",    # MSH_BRAVE_API_KEY
            f"{key.upper()}",            # BRAVE_API_KEY (uppercase)
        ]
        
        for env_name in env_patterns:
            env_value = os.getenv(env_name)
            if env_value:
                logger.debug(f"Environment variable found: {env_name}")
                return env_value
        
        return None
    
    def _log_resolution_failure(self, scope: str, key: str, original_value: str):
        """Log detailed information about resolution failure."""
        logger.warning(
            f"Could not resolve secret {original_value}. Tried:\n"
            f"  1. dbutils.secrets.get(scope='{scope}', key='{key}')\n"
            f"  2. MLflow get_secret() with formats: {scope}/{key}, {scope}.{key}, {key}\n"
            f"  3. Environment variables: {key}, {scope}_{key}, {scope.upper()}_{key}, {key.upper()}\n"
            f"Make sure the secret is accessible in this context."
        )
    
    def clear_cache(self):
        """Clear the resolution cache."""
        self._cache.clear()
        self._failed_secrets.clear()
        logger.info("Secret resolution cache cleared")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status for debugging."""
        return {
            "cached_secrets": len(self._cache),
            "failed_secrets": len(self._failed_secrets),
            "cached_keys": list(self._cache.keys()),
            "failed_keys": list(self._failed_secrets)
        }


# Global instance for shared use
_secret_resolver = SecretResolver()

def resolve_secret(value: Any) -> Any:
    """
    Convenience function to resolve secrets using the global resolver.
    
    Args:
        value: Value to resolve
        
    Returns:
        Resolved value
    """
    return _secret_resolver.resolve_secret(value)

def clear_secret_cache():
    """Clear the global secret resolution cache."""
    _secret_resolver.clear_cache()

def get_secret_cache_status() -> Dict[str, Any]:
    """Get the global secret cache status."""
    return _secret_resolver.get_cache_status()


def extract_token_usage(response) -> Dict[str, int]:
    """
    Extract token usage from LangChain/OpenAI response objects.
    
    Supports both structured response objects and response_metadata formats.
    Returns actual token counts, not estimates.
    
    Args:
        response: LangChain response object with usage metadata
        
    Returns:
        Dict with input_tokens, output_tokens, and total_tokens counts
    """
    default_usage = {
        "input_tokens": 0,
        "output_tokens": 0, 
        "total_tokens": 0
    }
    
    if response is None:
        return default_usage
    
    # Try usage_metadata first (LangChain format)
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage = response.usage_metadata
        return {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
    
    # Try response_metadata (alternative format)
    if hasattr(response, 'response_metadata') and response.response_metadata:
        metadata = response.response_metadata
        
        # Check for token_usage within response_metadata
        if "token_usage" in metadata:
            usage = metadata["token_usage"]
            return {
                "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
                "total_tokens": usage.get("total_tokens", 0)
            }
        
        # Check for direct usage fields in response_metadata
        if any(key in metadata for key in ["input_tokens", "output_tokens", "total_tokens"]):
            return {
                "input_tokens": metadata.get("input_tokens", 0),
                "output_tokens": metadata.get("output_tokens", 0),
                "total_tokens": metadata.get("total_tokens", 0)
            }
    
    # Try direct attributes on response object
    if hasattr(response, 'token_usage'):
        usage = response.token_usage
        return {
            "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            "total_tokens": usage.get("total_tokens", 0)
        }
    
    # Log when we can't extract usage data
    logger.debug(f"Could not extract token usage from response type: {type(response)}")
    
    return default_usage