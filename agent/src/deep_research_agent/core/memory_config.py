"""
Memory optimization configuration for production serving endpoints.

This module provides memory-constrained defaults for 4GB serving endpoints.
"""

from typing import Dict, Any


class MemoryOptimizedConfig:
    """Memory optimization settings for 4GB RAM limit serving endpoints."""

    # Cache memory limits (reduced from defaults)
    CACHE_MAX_MEMORY_MB = 150        # Down from 500MB
    CACHE_MAX_SIZE = 2000            # Down from 10000 entries

    # Content truncation limits
    MAX_SEARCH_CONTENT_LENGTH = 8000     # ~8KB per search result
    MAX_CITATION_SNIPPET_LENGTH = 500    # ~500 chars per citation

    # State list size limits - These are now configurable via base.yaml
    # Defaults maintained for backward compatibility
    MAX_OBSERVATIONS = 200            # CRITICAL FIX: Match base.yaml config (was 50, losing 75% of data!)
    MAX_SEARCH_RESULTS = 500         # Default - overridden by config.memory.max_search_results
    MAX_CITATIONS = 200              # Citations are smaller, allow more
    MAX_REFLECTIONS = 30             # Limit reflection history
    MAX_AGENT_HANDOFFS = 20          # Limit handoff history
    MAX_GENERAL_LIST_SIZE = 100      # Default for other lists
    MAX_OBSERVATIONS_PER_STEP = 100  # CRITICAL FIX: Match base.yaml config (was 10)

    # Memory pruning limits (for prune_state_for_memory)
    MAX_PRUNED_SEARCH_RESULTS = 200  # Target size after intelligent pruning (was 20)

    @classmethod
    def get_observations_limit(cls, config=None):
        """Get max observations limit from config or default."""
        if config and "memory" in config:
            return config["memory"].get("max_observations", cls.MAX_OBSERVATIONS)
        return cls.MAX_OBSERVATIONS

    @classmethod
    def get_observations_per_step_limit(cls, config=None):
        """Get max observations per step limit from config or default."""
        if config and "memory" in config:
            return config["memory"].get("max_observations_per_step", cls.MAX_OBSERVATIONS_PER_STEP)
        return cls.MAX_OBSERVATIONS_PER_STEP

    @classmethod
    def get_search_results_limit(cls, config=None):
        """Get max search results limit from config or default."""
        if config and "memory" in config:
            return config["memory"].get("max_search_results", cls.MAX_SEARCH_RESULTS)
        return cls.MAX_SEARCH_RESULTS
    
    # Memory monitoring
    MEMORY_LIMIT_MB = 3500           # 3.5GB limit (500MB headroom)
    AGGRESSIVE_GC_THRESHOLD = 0.8    # Start GC at 80% of limit
    
    # Workflow limits for complex research
    MAX_PLAN_ITERATIONS = 2          # Reduce from 3
    MAX_RESEARCH_LOOPS = 3           # Reduce from 5
    MAX_FACT_CHECK_LOOPS = 1         # Reduce from 2
    
    @classmethod
    def get_optimized_config(cls) -> Dict[str, Any]:
        """Get memory-optimized configuration dictionary."""
        return {
            "memory_optimization": {
                "enabled": True,
                "cache_max_memory_mb": cls.CACHE_MAX_MEMORY_MB,
                "cache_max_size": cls.CACHE_MAX_SIZE,
                "memory_limit_mb": cls.MEMORY_LIMIT_MB,
                "aggressive_gc_threshold": cls.AGGRESSIVE_GC_THRESHOLD,
                "content_truncation": {
                    "search_content_length": cls.MAX_SEARCH_CONTENT_LENGTH,
                    "citation_snippet_length": cls.MAX_CITATION_SNIPPET_LENGTH,
                },
                "list_size_limits": {
                    "observations": cls.MAX_OBSERVATIONS,
                    "search_results": cls.MAX_SEARCH_RESULTS,
                    "pruned_search_results": cls.MAX_PRUNED_SEARCH_RESULTS,
                    "citations": cls.MAX_CITATIONS,
                    "reflections": cls.MAX_REFLECTIONS,
                    "agent_handoffs": cls.MAX_AGENT_HANDOFFS,
                    "general": cls.MAX_GENERAL_LIST_SIZE,
                },
                "workflow_limits": {
                    "max_plan_iterations": cls.MAX_PLAN_ITERATIONS,
                    "max_research_loops": cls.MAX_RESEARCH_LOOPS,
                    "max_fact_check_loops": cls.MAX_FACT_CHECK_LOOPS,
                }
            }
        }
    
    @classmethod
    def apply_to_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply memory optimizations to existing configuration."""
        optimized = cls.get_optimized_config()
        
        # Merge with existing config, prioritizing memory optimizations
        config.update(optimized)
        
        # Override specific workflow settings
        if "workflow" in config:
            config["workflow"]["max_plan_iterations"] = cls.MAX_PLAN_ITERATIONS
            config["workflow"]["max_research_loops"] = cls.MAX_RESEARCH_LOOPS
            config["workflow"]["max_fact_check_loops"] = cls.MAX_FACT_CHECK_LOOPS
        
        # Override multi-agent settings
        if "multi_agent" in config:
            config["multi_agent"]["max_plan_iterations"] = cls.MAX_PLAN_ITERATIONS
            
        return config


def enable_memory_optimizations() -> Dict[str, Any]:
    """
    Enable memory optimizations for production serving.
    
    Returns:
        Configuration dictionary with memory optimizations enabled.
    """
    return MemoryOptimizedConfig.get_optimized_config()


# Environment variable to enable memory optimizations
import os
MEMORY_OPTIMIZED = os.environ.get("MEMORY_OPTIMIZED", "true").lower() == "true"