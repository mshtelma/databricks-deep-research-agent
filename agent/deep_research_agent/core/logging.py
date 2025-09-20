"""
Structured logging utilities for the research agent.

This module provides consistent logging configuration and utilities
for the research agent, replacing ad-hoc print statements.
"""

import logging
import sys
import json
import time
import traceback
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from .types import WorkflowMetrics


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ("name", "msg", "args", "levelname", "levelno", "pathname", 
                          "filename", "module", "exc_info", "exc_text", "stack_info",
                          "lineno", "funcName", "created", "msecs", "relativeCreated",
                          "thread", "threadName", "processName", "process", "getMessage"):
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class AgentLogger:
    """Enhanced logger for the research agent."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize agent logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers."""
        # Console handler with structured formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(console_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional extra data."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional extra data."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception."""
        if error:
            self.logger.error(message, exc_info=error, extra=kwargs)
        else:
            self.logger.error(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional extra data."""
        self.logger.debug(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, extra=kwargs)
    
    def workflow_step(self, step_name: str, status: str, **kwargs):
        """Log workflow step with consistent formatting."""
        self.info(
            f"Workflow step: {step_name}",
            step_name=step_name,
            status=status,
            **kwargs
        )
    
    def tool_operation(self, tool_name: str, operation: str, success: bool, **kwargs):
        """Log tool operation with consistent formatting."""
        level_method = self.info if success else self.error
        level_method(
            f"Tool operation: {tool_name}.{operation}",
            tool_name=tool_name,
            operation=operation,
            success=success,
            **kwargs
        )
    
    def performance_metric(self, metric_name: str, value: Union[int, float], **kwargs):
        """Log performance metric."""
        self.info(
            f"Performance metric: {metric_name}",
            metric_name=metric_name,
            metric_value=value,
            **kwargs
        )
    
    def search_results(self, query: str, source: str, count: int, **kwargs):
        """Log search results."""
        self.info(
            f"Search results: {count} results for query",
            query=query,
            source=source,
            result_count=count,
            **kwargs
        )


class MLflowLogger:
    """MLflow integration for logging metrics and artifacts."""
    
    def __init__(self, logger: AgentLogger):
        """Initialize MLflow logger."""
        self.logger = logger
        self.mlflow_available = MLFLOW_AVAILABLE
    
    def log_metric(self, key: str, value: Union[int, float], step: Optional[int] = None):
        """Log metric to MLflow."""
        if not self.mlflow_available:
            self.logger.debug(f"MLflow not available, skipping metric: {key}={value}")
            return
        
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            self.logger.error(f"Failed to log MLflow metric {key}", error=e)
    
    def log_param(self, key: str, value: Any):
        """Log parameter to MLflow."""
        if not self.mlflow_available:
            self.logger.debug(f"MLflow not available, skipping param: {key}={value}")
            return
        
        try:
            mlflow.log_param(key, value)
        except Exception as e:
            self.logger.error(f"Failed to log MLflow param {key}", error=e)
    
    def log_text(self, text: str, artifact_file: str):
        """Log text artifact to MLflow."""
        if not self.mlflow_available:
            self.logger.debug(f"MLflow not available, skipping text artifact: {artifact_file}")
            return
        
        try:
            mlflow.log_text(text, artifact_file)
        except Exception as e:
            self.logger.error(f"Failed to log MLflow text artifact {artifact_file}", error=e)
    
    def log_workflow_metrics(self, metrics: WorkflowMetrics):
        """Log workflow metrics to MLflow."""
        metric_dict = {
            "total_queries_generated": metrics.total_queries_generated,
            "total_web_results": metrics.total_web_results,
            "total_vector_results": metrics.total_vector_results,
            "total_research_loops": metrics.total_research_loops,
            "execution_time_seconds": metrics.execution_time_seconds,
            "error_count": metrics.error_count,
            "success_rate": metrics.success_rate,
        }
        
        for key, value in metric_dict.items():
            self.log_metric(key, value)


@contextmanager
def log_execution_time(logger: AgentLogger, operation_name: str):
    """Context manager to log execution time."""
    start_time = time.time()
    logger.info(f"Starting operation: {operation_name}")
    
    try:
        yield
        duration = time.time() - start_time
        logger.performance_metric(
            f"{operation_name}_duration",
            duration,
            operation=operation_name,
            success=True
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Operation failed: {operation_name}",
            error=e,
            operation=operation_name,
            duration=duration
        )
        raise


def log_function_calls(logger: AgentLogger):
    """Decorator to log function calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(
                f"Calling function: {func_name}",
                function=func_name,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
            
            try:
                with log_execution_time(logger, func_name):
                    result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Function {func_name} failed", error=e, function=func_name)
                raise
        
        return wrapper
    return decorator


def setup_logging(name: str, level: int = logging.INFO) -> AgentLogger:
    """
    Setup logging for the application.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured AgentLogger instance
    """
    return AgentLogger(name, level)


def get_logger(name: str) -> AgentLogger:
    """Get or create logger for the given name."""
    return AgentLogger(name)


# Create default logger
default_logger = get_logger("research_agent")