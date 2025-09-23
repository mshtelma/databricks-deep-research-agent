"""
Global rate limiter for coordinating search requests across all components.
Implements singleton pattern with thread-safe rate limiting.
"""

import time
import threading
from typing import Optional, Dict, Any
from collections import deque
from deep_research_agent.core.logging import get_logger

logger = get_logger(__name__)


class GlobalSearchRateLimiter:
    """
    Singleton rate limiter for coordinating all search requests.
    
    Features:
    - Global coordination across all search components
    - Per-provider rate limiting
    - Circuit breaker pattern for consecutive failures
    - Thread-safe operations
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the rate limiter (only once due to singleton)."""
        if self._initialized:
            return
            
        self._initialized = True
        self._provider_states: Dict[str, Dict[str, Any]] = {}
        self._global_lock = threading.Lock()
        
        # Load configuration from config files
        config = self._load_config()
        
        # Check for provider-specific rate limits (per second)
        brave_rate_limit = config.get("search", {}).get("providers", {}).get("brave", {}).get("rate_limit")
        if brave_rate_limit:
            # rate_limit means requests per second
            self.default_cooldown_seconds = 1.0 / brave_rate_limit
            logger.info(f"Using Brave rate limit: {brave_rate_limit} req/sec, {self.default_cooldown_seconds:.2f}s cooldown")
        else:
            # Default: 1 request per second
            self.default_cooldown_seconds = 1.0
            logger.info("Using default rate limit: 1 req/sec, 1.0s cooldown")
        
        self.circuit_breaker_threshold = 3
        self.circuit_breaker_reset_seconds = 20
        
        logger.info("GlobalSearchRateLimiter initialized with configuration-based settings")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from base.yaml files"""
        try:
            import yaml
            from pathlib import Path
            
            config_paths = [
                Path(__file__).parent.parent.parent / "conf" / "base.yaml",
                Path.cwd() / "conf" / "base.yaml",
                Path(__file__).parent.parent / "agent_config.yaml",  # Legacy support
            ]
            
            for path in config_paths:
                if path.exists():
                    with open(path, 'r') as f:
                        config = yaml.safe_load(f)
                        logger.info(f"Loaded config from: {path}")
                        return config
                        
            logger.warning("No config file found, using defaults")
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return {}
    
    def _get_provider_state(self, provider: str) -> Dict[str, Any]:
        """Get or create state for a provider."""
        if provider not in self._provider_states:
            self._provider_states[provider] = {
                'consecutive_failures': 0,
                'circuit_open': False,
                'circuit_open_until': 0,
                'last_request_time': 0,
                'total_requests': 0,
                'total_failures': 0
            }
        return self._provider_states[provider]
    
    def acquire(self, provider: str = 'default', 
                cooldown_seconds: Optional[float] = None) -> bool:
        """
        Acquire permission to make a search request.
        
        Args:
            provider: Name of the search provider
            cooldown_seconds: Minimum seconds between requests (uses configured rate limit)
            
        Returns:
            True if request can proceed, False if circuit is open
        """
        with self._global_lock:
            state = self._get_provider_state(provider)
            current_time = time.time()
            
            # Check circuit breaker
            if state['circuit_open']:
                if current_time < state['circuit_open_until']:
                    logger.warning(
                        f"Circuit breaker OPEN for {provider}",
                        extra={
                            'provider': provider,
                            'open_until': state['circuit_open_until'],
                            'consecutive_failures': state['consecutive_failures']
                        }
                    )
                    return False
                else:
                    # Reset circuit breaker
                    logger.info(f"Circuit breaker RESET for {provider}")
                    state['circuit_open'] = False
                    state['consecutive_failures'] = 0
            
            # Apply simple cooldown-based rate limiting
            cooldown = cooldown_seconds or self.default_cooldown_seconds
            
            # Check cooldown period
            time_since_last = current_time - state['last_request_time']
            if time_since_last < cooldown:
                wait_time = cooldown - time_since_last
                logger.info(
                    f"Rate limiting {provider}: waiting {wait_time:.2f}s",
                    extra={'provider': provider, 'cooldown': cooldown}
                )
                time.sleep(wait_time)
                current_time = time.time()
            
            # Record this request
            state['last_request_time'] = current_time
            state['total_requests'] += 1
            
            logger.debug(
                f"Request acquired for {provider}",
                extra={
                    'provider': provider,
                    'total_requests': state['total_requests']
                }
            )
            
            return True
    
    def report_success(self, provider: str = 'default'):
        """Report successful request completion."""
        with self._global_lock:
            state = self._get_provider_state(provider)
            state['consecutive_failures'] = 0
            logger.debug(f"Request succeeded for {provider}")
    
    def report_failure(self, provider: str = 'default', is_rate_limit: bool = False):
        """
        Report request failure.
        
        Args:
            provider: Name of the search provider
            is_rate_limit: Whether failure was due to rate limiting
        """
        with self._global_lock:
            state = self._get_provider_state(provider)
            state['consecutive_failures'] += 1
            state['total_failures'] += 1
            
            # If rate limited, apply longer cooldown
            if is_rate_limit:
                state['last_request_time'] = time.time() + 15  # Extra 15 second penalty
                logger.warning(
                    f"Rate limit failure for {provider}, applying 15s penalty",
                    extra={'provider': provider}
                )
            
            # Open circuit breaker if threshold reached
            if state['consecutive_failures'] >= self.circuit_breaker_threshold:
                state['circuit_open'] = True
                state['circuit_open_until'] = time.time() + self.circuit_breaker_reset_seconds
                logger.error(
                    f"Circuit breaker OPENED for {provider} after {state['consecutive_failures']} failures",
                    extra={
                        'provider': provider,
                        'consecutive_failures': state['consecutive_failures'],
                        'reset_seconds': self.circuit_breaker_reset_seconds
                    }
                )
    
    def update_cooldown(self, provider: str, additional_cooldown: float):
        """Update the cooldown period for a provider after rate limiting"""
        with self._global_lock:
            state = self._get_provider_state(provider)
            state['last_request_time'] = time.time() + additional_cooldown
            logger.info(
                f"Updated cooldown for {provider}: +{additional_cooldown}s",
                extra={'provider': provider, 'additional_cooldown': additional_cooldown}
            )
    
    def get_stats(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a provider or all providers."""
        with self._global_lock:
            if provider:
                state = self._get_provider_state(provider)
                return {
                    'provider': provider,
                    'total_requests': state['total_requests'],
                    'total_failures': state['total_failures'],
                    'consecutive_failures': state['consecutive_failures'],
                    'circuit_open': state['circuit_open'],
                    'last_request_time': state['last_request_time']
                }
            else:
                return {
                    p: self.get_stats(p) for p in self._provider_states.keys()
                }
    
    def reset(self, provider: Optional[str] = None):
        """Reset rate limiter state for testing."""
        with self._global_lock:
            if provider:
                if provider in self._provider_states:
                    del self._provider_states[provider]
                logger.info(f"Reset rate limiter for {provider}")
            else:
                self._provider_states.clear()
                logger.info("Reset all rate limiter states")


# Global singleton instance
global_rate_limiter = GlobalSearchRateLimiter()