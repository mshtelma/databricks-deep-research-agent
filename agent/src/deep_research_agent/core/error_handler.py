"""
Unified error handling system with consistent retry policies and context tracking.

This module provides centralized error handling to eliminate the inconsistent patterns
scattered across the codebase and improve reliability through proper retry logic.
"""

import time
import functools
import random
import logging
from typing import Any, Callable, Optional, TypeVar, Dict, List, Type
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels for proper categorization."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context for error tracking and reporting with detailed metrics."""
    
    operation: str
    attempt: int = 0
    max_attempts: int = 3
    errors: List[Exception] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: Exception, severity: Optional[ErrorSeverity] = None):
        """Add error to context with optional severity override."""
        self.errors.append(error)
        self.attempt += 1
        if severity:
            self.severity = severity
    
    def should_retry(self) -> bool:
        """Check if operation should be retried based on attempts and severity."""
        if self.severity == ErrorSeverity.CRITICAL:
            return False  # Never retry critical errors
        return self.attempt < self.max_attempts
    
    def get_duration(self) -> float:
        """Get operation duration in seconds."""
        return time.time() - self.start_time
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors for reporting."""
        return {
            "operation": self.operation,
            "attempts": self.attempt,
            "duration": self.get_duration(),
            "severity": self.severity.value,
            "error_count": len(self.errors),
            "error_types": [type(e).__name__ for e in self.errors],
            "last_error": str(self.errors[-1]) if self.errors else None,
            "metadata": self.metadata
        }


@dataclass
class RetryPolicy:
    """Comprehensive retry policy configuration."""
    
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)
    non_retryable_exceptions: tuple = ()
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff and jitter."""
        delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            jitter_factor = 0.1 * random.random()  # Up to 10% jitter
            delay *= (1.0 + jitter_factor)
        
        return delay
    
    def should_retry(self, error: Exception) -> bool:
        """Determine if error should trigger a retry."""
        if self.non_retryable_exceptions and isinstance(error, self.non_retryable_exceptions):
            return False
        return isinstance(error, self.retryable_exceptions)


class UnifiedErrorHandler:
    """
    Centralized error handling with consistent patterns across the codebase.
    
    This class provides decorators and context managers for unified error handling,
    eliminating the scattered try/catch patterns throughout the application.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Default policies for different operation types
        self.default_policies = {
            "search": RetryPolicy(
                max_attempts=3,
                base_delay=2.0,
                max_delay=30.0,
                retryable_exceptions=(Exception,),
                non_retryable_exceptions=(KeyboardInterrupt, SystemExit)
            ),
            "api_call": RetryPolicy(
                max_attempts=3,
                base_delay=1.0,
                max_delay=60.0,
                exponential_base=2.0,
                retryable_exceptions=(Exception,)
            ),
            "llm_call": RetryPolicy(
                max_attempts=2,  # Lower for expensive LLM calls
                base_delay=3.0,
                max_delay=45.0,
                retryable_exceptions=(Exception,)
            ),
            "io_operation": RetryPolicy(
                max_attempts=3,
                base_delay=0.5,
                max_delay=10.0,
                retryable_exceptions=(OSError, IOError)
            )
        }
    
    def get_policy(self, operation_type: str) -> RetryPolicy:
        """Get retry policy for operation type."""
        return self.default_policies.get(operation_type, RetryPolicy())
    
    def with_retry(
        self, 
        operation: str,
        operation_type: str = "default",
        policy: Optional[RetryPolicy] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> Callable:
        """
        Decorator for adding retry logic to functions.
        
        Args:
            operation: Human-readable operation name
            operation_type: Type of operation for policy selection
            policy: Custom retry policy (overrides operation_type)
            severity: Error severity level
        """
        policy = policy or self.get_policy(operation_type)
        
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                context = ErrorContext(
                    operation=operation,
                    max_attempts=policy.max_attempts,
                    severity=severity
                )
                
                while True:
                    try:
                        result = func(*args, **kwargs)
                        
                        # Log successful retry if it took multiple attempts
                        if context.attempt > 0:
                            self.logger.info(
                                f"Operation '{operation}' succeeded after {context.attempt} retries",
                                extra={"retry_context": context.get_error_summary()}
                            )
                        
                        return result
                    
                    except Exception as e:
                        context.add_error(e, self._classify_error_severity(e))
                        
                        # Check if we should retry this error
                        if not policy.should_retry(e) or not context.should_retry():
                            self.logger.error(
                                f"Operation '{operation}' failed after {context.attempt} attempts",
                                extra={
                                    "error_context": context.get_error_summary(),
                                    "final_error": str(e)
                                }
                            )
                            # Re-raise the last error with context
                            raise self._create_contextual_error(e, context)
                        
                        delay = policy.get_delay(context.attempt)
                        self.logger.warning(
                            f"Operation '{operation}' failed (attempt {context.attempt}/{policy.max_attempts}), "
                            f"retrying in {delay:.1f}s: {e}"
                        )
                        
                        time.sleep(delay)
            
            return wrapper
        return decorator
    
    @contextmanager
    def error_context(
        self, 
        operation: str, 
        fallback: Optional[Any] = None,
        suppress_exceptions: tuple = (),
        log_level: int = logging.ERROR
    ):
        """
        Context manager for consistent error handling with optional fallback.
        
        Args:
            operation: Operation name for logging
            fallback: Fallback value to return on error
            suppress_exceptions: Exception types to suppress
            log_level: Logging level for errors
        """
        start_time = time.time()
        try:
            yield
        except suppress_exceptions as e:
            duration = time.time() - start_time
            self.logger.log(
                log_level,
                f"Operation '{operation}' failed after {duration:.2f}s (suppressed): {e}"
            )
            if fallback is not None:
                return fallback
        except Exception as e:
            duration = time.time() - start_time
            self.logger.log(
                log_level,
                f"Operation '{operation}' failed after {duration:.2f}s",
                exc_info=True,
                extra={
                    "operation": operation,
                    "duration": duration,
                    "error_type": type(e).__name__
                }
            )
            
            if fallback is not None:
                self.logger.info(f"Using fallback value for operation '{operation}'")
                return fallback
            raise
    
    def safe_execute(
        self,
        func: Callable[..., T],
        operation: str,
        *args,
        fallback: Optional[T] = None,
        **kwargs
    ) -> Optional[T]:
        """
        Safely execute a function with error handling.
        
        Args:
            func: Function to execute
            operation: Operation name
            fallback: Fallback value on error
            *args, **kwargs: Arguments for function
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(
                f"Safe execution of '{operation}' failed: {e}",
                exc_info=True
            )
            return fallback
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on exception type."""
        # Critical system errors
        if isinstance(error, (SystemExit, KeyboardInterrupt, MemoryError)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.HIGH
        
        # Medium severity errors  
        elif isinstance(error, (ValueError, TypeError, AttributeError)):
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        else:
            return ErrorSeverity.LOW
    
    def _create_contextual_error(self, original_error: Exception, context: ErrorContext) -> Exception:
        """Create error with additional context information."""
        error_msg = (
            f"Operation '{context.operation}' failed after {context.attempt} attempts "
            f"over {context.get_duration():.2f}s. Original error: {original_error}"
        )
        
        # Create new exception with same type but enhanced message
        error_type = type(original_error)
        try:
            return error_type(error_msg)
        except:
            # Fallback to RuntimeError if we can't create the original type
            return RuntimeError(error_msg)


# Global error handler instance for easy access
global_error_handler = UnifiedErrorHandler()

# Convenience decorators
def retry(operation: str, operation_type: str = "default", **kwargs):
    """Convenience decorator for retry functionality."""
    return global_error_handler.with_retry(operation, operation_type, **kwargs)

def safe_call(operation: str, fallback: Any = None):
    """Convenience decorator for safe function execution."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return global_error_handler.safe_execute(func, operation, *args, fallback=fallback, **kwargs)
        return wrapper
    return decorator