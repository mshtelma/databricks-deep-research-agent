"""
Memory monitoring utilities for the Deep Research Agent.

Provides lightweight memory tracking without requiring external dependencies.
"""

import gc
import sys
from typing import Dict, Any, Optional
from deep_research_agent.core import get_logger

logger = get_logger(__name__)


class MemoryMonitor:
    """Lightweight memory monitoring for the research agent."""
    
    def __init__(self, memory_limit_mb: Optional[int] = None):
        """
        Initialize memory monitor.
        
        Args:
            memory_limit_mb: Optional memory limit in MB. If exceeded, warnings are logged.
        """
        # Default to 3.5GB limit for 4GB serving endpoint (leave 500MB headroom)
        self.memory_limit_mb = memory_limit_mb or 3500
        self.start_memory = self._get_memory_usage()
        self.aggressive_gc_threshold = self.memory_limit_mb * 0.8  # Start aggressive GC at 80%
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB using sys.getsizeof for objects."""
        try:
            # Use garbage collection to get all tracked objects
            gc.collect()  # Force garbage collection
            
            # Estimate memory usage from object count and sizes
            # This is a rough approximation, but doesn't require external dependencies
            objects = gc.get_objects()
            total_size = sum(sys.getsizeof(obj) for obj in objects)
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return 0.0
    
    def log_memory_usage(self, context: str, state: Optional[Dict[str, Any]] = None):
        """
        Log current memory usage with context.
        
        Args:
            context: Description of current operation
            state: Optional state dict to analyze
        """
        try:
            current_memory = self._get_memory_usage()
            memory_increase = current_memory - self.start_memory
            
            # Basic memory info
            log_msg = f"Memory usage at {context}: {current_memory:.1f} MB (+ {memory_increase:.1f} MB from start)"
            
            # Analyze state size if provided
            if state:
                state_info = self._analyze_state_size(state)
                log_msg += f" | State: {state_info}"
            
            logger.info(log_msg)
            
            # PROACTIVE MEMORY MANAGEMENT
            # Start aggressive GC when reaching 80% of memory limit
            if current_memory > self.aggressive_gc_threshold:
                logger.warning(f"Memory usage high ({current_memory:.1f} MB), triggering aggressive GC")
                self.force_garbage_collection(f"high_memory_at_{context}")
                
                # Re-check memory after GC
                current_memory_post_gc = self._get_memory_usage()
                freed = current_memory - current_memory_post_gc
                logger.info(f"Post-GC memory: {current_memory_post_gc:.1f} MB (freed {freed:.1f} MB)")
            
            # Check memory limit and log critical warning
            if self.memory_limit_mb and current_memory > self.memory_limit_mb:
                logger.error(
                    f"CRITICAL: Memory usage ({current_memory:.1f} MB) exceeded limit ({self.memory_limit_mb} MB) at {context}"
                )
                # Force additional GC as emergency measure
                self.force_garbage_collection(f"emergency_gc_at_{context}")
                
        except Exception as e:
            logger.warning(f"Error monitoring memory: {e}")
    
    def _analyze_state_size(self, state: Dict[str, Any]) -> str:
        """Analyze state size and return summary string."""
        try:
            # Count elements in key arrays
            observations_count = len(state.get("observations", []))
            search_results_count = len(state.get("search_results", []))
            messages_count = len(state.get("messages", []))
            citations_count = len(state.get("citations", []))
            
            # Estimate state size
            state_size_mb = sys.getsizeof(state) / (1024 * 1024)
            
            return (f"obs:{observations_count}, results:{search_results_count}, "
                   f"msgs:{messages_count}, citations:{citations_count} "
                   f"(~{state_size_mb:.2f}MB)")
                   
        except Exception as e:
            return f"analysis_error:{e}"
    
    def check_memory_limit(self, context: str) -> bool:
        """
        Check if memory limit has been exceeded.
        
        Args:
            context: Description of current operation
            
        Returns:
            True if under limit (or no limit set), False if exceeded
        """
        if not self.memory_limit_mb:
            return True
            
        current_memory = self._get_memory_usage()
        if current_memory > self.memory_limit_mb:
            logger.error(
                f"Memory limit exceeded at {context}: {current_memory:.1f} MB > {self.memory_limit_mb} MB"
            )
            return False
            
        return True
    
    def force_garbage_collection(self, context: str):
        """Force garbage collection and log memory change."""
        try:
            before_memory = self._get_memory_usage()
            
            # Force garbage collection multiple times for better cleanup
            for _ in range(3):
                gc.collect()
            
            after_memory = self._get_memory_usage()
            memory_freed = before_memory - after_memory
            
            if memory_freed > 1.0:  # Only log if significant memory was freed
                logger.info(f"Garbage collection at {context}: freed {memory_freed:.1f} MB")
                
        except Exception as e:
            logger.warning(f"Error during garbage collection: {e}")


# Global memory monitor instance
_global_memory_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor(memory_limit_mb: Optional[int] = None) -> MemoryMonitor:
    """Get global memory monitor instance."""
    global _global_memory_monitor
    if _global_memory_monitor is None:
        _global_memory_monitor = MemoryMonitor(memory_limit_mb)
    return _global_memory_monitor


def reset_memory_monitor():
    """Reset global memory monitor (for testing)."""
    global _global_memory_monitor
    _global_memory_monitor = None