"""
Pydantic configuration schema for the research agent.

This module defines the complete configuration structure with type safety,
validation, and environment variable support.
"""

from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path

class VerificationLevel(str, Enum):
    """Fact checking verification levels"""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"

class ReportStyle(str, Enum):
    """Available report styles"""
    PROFESSIONAL = "professional"
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    EXECUTIVE = "executive"
    CONCISE = "concise"
    DETAILED = "detailed"
    COMPARATIVE = "comparative"
    DEFAULT = "default"  # Adaptive structure based on query context

class ModelConfig(BaseSettings):
    """Model endpoint configuration - supports both single-endpoint and multi-endpoint tier configs"""
    # Single-endpoint mode (legacy - for agents that don't use rate limiting)
    endpoint: Optional[str] = Field(default=None, description="Single model endpoint")

    # Multi-endpoint tier mode (new - for rate limiting tiers)
    endpoints: Optional[List[str]] = Field(default=None, description="List of endpoints for tier")
    tokens_per_minute: Optional[int] = Field(default=None, ge=1000, description="Token budget per minute")
    rotation_strategy: Optional[str] = Field(default=None, description="Endpoint rotation strategy")
    fallback_on_429: Optional[bool] = Field(default=None, description="Fallback to next endpoint on 429")

    # Common fields (existing)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4000, gt=0, le=32000)
    reasoning_effort: Optional[str] = Field(
        default=None,
        description="Reasoning effort level for models that support it (low, medium, high)"
    )
    reasoning_budget: Optional[int] = Field(
        default=None,
        ge=0,
        le=100000,
        description="Token budget for reasoning in hybrid models like Claude"
    )

    @validator('reasoning_effort')
    def validate_reasoning_effort(cls, v):
        """Validate reasoning effort is one of the allowed values"""
        if v is not None and v not in ['low', 'medium', 'high']:
            raise ValueError(f"reasoning_effort must be 'low', 'medium', or 'high', got {v}")
        return v

    @validator('rotation_strategy')
    def validate_rotation_strategy(cls, v):
        """Validate rotation strategy is one of the allowed values"""
        if v and v not in ['round_robin', 'lru', 'random', 'priority']:
            raise ValueError(f"rotation_strategy must be one of: round_robin, lru, random, priority")
        return v

    model_config = SettingsConfigDict(extra="allow")  # Allow additional fields for rate limiting configs

class RetryConfig(BaseSettings):
    """Retry configuration for rate limiting"""
    base_delay_seconds: float = Field(default=2.0, ge=0.1, description="Base delay between retries")
    max_delay_seconds: float = Field(default=60.0, ge=1.0, description="Maximum delay between retries")
    max_retries: int = Field(default=5, ge=1, le=10, description="Maximum number of retries")
    jitter: float = Field(default=0.3, ge=0.0, le=1.0, description="Jitter factor for retry delays")

class CoordinationConfig(BaseSettings):
    """Request coordination configuration"""
    max_concurrent_per_endpoint: int = Field(default=2, ge=1, le=10, description="Max concurrent requests per endpoint")

class CooldownConfig(BaseSettings):
    """Cooldown configuration for 429 handling"""
    default_cooldown_seconds: float = Field(default=60.0, ge=1.0, description="Default cooldown after 429")
    max_cooldown_seconds: float = Field(default=300.0, ge=60.0, description="Maximum cooldown duration")
    respect_retry_after_header: bool = Field(default=True, description="Respect Retry-After header from 429 responses")

class TokenTrackingConfig(BaseSettings):
    """Token budget tracking configuration"""
    enable_sliding_window: bool = Field(default=True, description="Use sliding window for token tracking")
    window_seconds: int = Field(default=60, ge=10, le=300, description="Sliding window duration")
    safety_margin: float = Field(default=0.9, ge=0.5, le=1.0, description="Safety margin for token budgets (90% = use 90% of limit)")

class PhaseDelaysConfig(BaseSettings):
    """Phase delay configuration to prevent traffic bursts"""
    after_research_before_fact_check: float = Field(default=3.0, ge=0.0, description="Delay after research phase")
    after_fact_check_before_report: float = Field(default=5.0, ge=0.0, description="Delay after fact checking")
    between_section_generations: float = Field(default=2.0, ge=0.0, description="Delay between report sections")

class CoordinatorConfig(BaseSettings):
    """Coordinator agent configuration"""
    enabled: bool = True
    model: str = "default"
    enable_safety_filter: bool = True

class PlannerConfig(BaseSettings):
    """Planner agent configuration"""
    enabled: bool = True
    model: str = "default"
    reasoning_model: Optional[str] = None
    max_iterations: int = Field(default=3, ge=1, le=10)
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_deep_thinking: bool = False

class ResearcherConfig(BaseSettings):
    """Researcher agent configuration"""
    enabled: bool = True
    model: str = "web_research"
    max_steps_per_execution: int = Field(default=5, ge=1)
    enable_reflexion: bool = True

class FactCheckerConfig(BaseSettings):
    """Fact checker agent configuration"""
    enabled: bool = True
    model: str = "default"
    verification_level: VerificationLevel = VerificationLevel.MODERATE
    enable_contradiction_detection: bool = True

class HybridSettingsConfig(BaseSettings):
    """Hybrid multi-pass report generation settings"""
    enable_async_blocks: bool = Field(
        default=False,
        description="Enable async table generation (test in staging first)"
    )
    fallback_on_empty_observations: bool = Field(
        default=True,
        description="Auto-fallback to section-by-section on errors"
    )
    table_anchor_format: str = Field(
        default="[TABLE: {id}]",
        description="Placeholder format in Phase 2"
    )
    calc_selector_top_k: int = Field(
        default=60,
        ge=10,
        le=200,
        description="Top-scoring observations for calculation prompt"
    )
    calc_recent_tail: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Recent observations appended after scoring"
    )
    max_calc_prompt_chars: int = Field(
        default=12000,
        ge=1000,
        le=50000,
        description="Hard limit on Phase 1 prompt length"
    )
    table_fallback_max_rows: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum rows in bullet fallback summaries"
    )
    enable_table_fallback_summary: bool = Field(
        default=True,
        description="Generate bullet summaries when tables fail"
    )
    enable_file_reference_filter: bool = Field(
        default=True,
        description="Remove file/tool references from observations"
    )
    contamination_patterns: List[str] = Field(
        default_factory=lambda: [
            r'\b[\w\-]+\.(xlsx|xlsm|csv|json|pdf|doc|docx|xls)\b',
            r'github\.com[/\w\-\.]*',
            r'gitlab\.com[/\w\-\.]*',
            r'\b(spreadsheet|repository|download|attachment)\b'
        ],
        description="Regex patterns for content sanitization"
    )
    holistic_timeout_seconds: int = Field(
        default=240,
        ge=30,
        le=600,
        description="Timeout for Phase 2 holistic report generation"
    )

class ReporterConfig(BaseSettings):
    """Reporter agent configuration"""
    enabled: bool = True
    model: str = "synthesis"
    default_style: ReportStyle = ReportStyle.DEFAULT
    citation_style: str = "APA"
    enable_grounding_markers: bool = True
    use_semantic_extraction: bool = True
    generation_mode: str = Field(
        default="section_by_section",
        description="Report generation mode: section_by_section or hybrid"
    )
    fail_on_empty_observations: bool = Field(
        default=True,
        description="Raise error if observation filtering produces empty sections"
    )
    max_concurrent_blocks: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Throttle for concurrent structured output calls"
    )
    hybrid_settings: HybridSettingsConfig = Field(
        default_factory=HybridSettingsConfig,
        description="Hybrid multi-pass generation configuration"
    )

class SearchProviderConfig(BaseSettings):
    """Search provider configuration"""
    enabled: bool = True
    api_key: Optional[str] = None
    max_results: int = Field(default=10, ge=1, le=50)
    rate_limit: int = Field(default=60, ge=1)  # requests per minute
    timeout: int = Field(default=30, ge=5)

class SearchConfig(BaseSettings):
    """Search configuration"""
    providers: Dict[str, SearchProviderConfig] = Field(default_factory=lambda: {
        "brave": SearchProviderConfig(),
        "tavily": SearchProviderConfig(enabled=False)
    })
    max_concurrent_searches: int = Field(default=3, ge=1)
    batch_delay_seconds: float = Field(default=1.0, ge=0.1)
    max_results_per_query: int = Field(default=10, ge=1)
    enable_safe_search: bool = True

class WorkflowConfig(BaseSettings):
    """Workflow and execution configuration"""
    max_research_loops: int = Field(default=3, ge=1, le=10)
    max_fact_check_loops: int = Field(default=2, ge=1, le=5)
    max_total_steps: int = Field(default=50, ge=1)
    max_wall_clock_seconds: int = Field(default=300, ge=30)
    recursion_limit: int = Field(default=100, ge=10)
    enable_background_investigation: bool = True
    enable_human_feedback: bool = False
    auto_accept_plan: bool = False
    enable_circuit_breakers: bool = True
    enable_progress_tracking: bool = True

    # Async Architecture (Phase 3 - Hybrid Async Implementation)
    experimental_async_nodes: bool = Field(
        default=True,
        description="Enable async agent nodes for proper LLM await handling. "
                    "Context-aware streaming bridge automatically detects FastAPI (async) vs MLflow (sync). "
                    "Set to false to rollback to legacy sync nodes."
    )

class RateLimitingConfig(BaseSettings):
    """Rate limiting configuration - supports both legacy and enhanced modes"""
    # Legacy fields (kept for backward compatibility)
    max_parallel_requests: int = Field(default=10, ge=1)
    max_requests_per_minute: int = Field(default=60, ge=1)
    max_tokens_per_minute: int = Field(default=100000, ge=1000)

    # Enhanced rate limiting (new - optional, comes from YAML)
    enabled: bool = Field(default=False, description="Enable advanced rate limiting system")
    retry: Optional[RetryConfig] = Field(default=None, description="Retry configuration")
    coordination: Optional[CoordinationConfig] = Field(default=None, description="Request coordination")
    cooldown: Optional[CooldownConfig] = Field(default=None, description="Cooldown configuration")
    token_tracking: Optional[TokenTrackingConfig] = Field(default=None, description="Token tracking")
    phase_delays: Optional[PhaseDelaysConfig] = Field(default=None, description="Phase delays")

class GroundingConfig(BaseSettings):
    """Grounding and factuality configuration"""
    enable_grounding: bool = True
    verification_level: VerificationLevel = VerificationLevel.MODERATE
    enable_contradiction_detection: bool = True
    factuality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

class ReflexionConfig(BaseSettings):
    """Reflexion and self-improvement configuration"""
    enable_reflexion: bool = True
    reflection_memory_size: int = Field(default=5, ge=1)

class StreamingConfig(BaseSettings):
    """Streaming and events configuration"""
    enable_streaming: bool = True
    max_events_per_second: int = Field(default=20, ge=1)
    batch_events: bool = True
    batch_size: int = Field(default=3, ge=1)
    batch_timeout_ms: int = Field(default=200, ge=50)

class CitationsConfig(BaseSettings):
    """Citations configuration"""
    citation_style: str = "APA"
    enable_citations: bool = True
    max_citations_per_section: int = Field(default=10, ge=1)

class ReportConfig(BaseSettings):
    """Report generation configuration"""
    default_style: ReportStyle = ReportStyle.DEFAULT
    enable_grounding_markers: bool = True
    citation_style: str = "APA"
    max_table_observations: int = Field(
        default=30,
        ge=1,
        le=500,
        description="Maximum observations to feed to table generation"
    )
    max_paragraph_observations: int = Field(
        default=15,
        ge=1,
        le=200,
        description="Maximum observations per paragraph"
    )
    section_observation_limits: Dict[str, int] = Field(
        default_factory=dict,
        description="Section-specific observation limits (optional overrides for hybrid mode)"
    )

class QualityEnhancementConfig(BaseSettings):
    """Quality enhancement configuration for report generation"""
    enabled: bool = True
    detect_redundancy: bool = True
    eliminate_redundancy: bool = True
    optimize_structure: bool = True

class AdaptiveStructureConfig(BaseSettings):
    """Adaptive structure configuration for context-specific report sections"""
    enable_adaptive_structure: bool = True
    adaptive_structure_cache_ttl: int = 3600
    use_llm_for_structure: bool = True
    max_structure_sections: int = 7
    min_structure_sections: int = 4


class EntityValidationConfig(BaseSettings):
    """Entity validation configuration"""
    enabled: bool = True
    validation_mode: str = "strict"  # strict, moderate, lenient
    enable_synthesis_validation: bool = True
    enable_observation_validation: bool = True
    enable_section_validation: bool = True
    track_violations: bool = True

class MemoryConfig(BaseSettings):
    """Memory and state management configuration"""
    max_observations: int = Field(default=50, ge=10, le=1000, description="Maximum observations to keep in state")
    max_observations_per_step: int = Field(default=10, ge=5, le=500, description="Maximum observations per research step")
    max_search_results: int = Field(default=100, ge=10, le=1000, description="Maximum search results to accumulate")
    relevance_buffer: int = Field(
        default=40,
        ge=10,
        le=200,
        description="Minimum high-scoring observations per section before chronological pruning"
    )

class TierFallbackConfig(BaseSettings):
    """Tier fallback configuration for cross-tier degradation"""
    enable_cross_tier_fallback: bool = Field(default=True, description="Enable cross-tier fallback on rate limits")
    fallback_chain: Dict[str, Optional[str]] = Field(
        default_factory=lambda: {
            "complex": "analytical",
            "analytical": "simple",
            "simple": "micro",
            "micro": None
        },
        description="Tier fallback chain mapping"
    )

class ResearchConfig(BaseSettings):
    """Main research agent configuration"""
    
    model_config = SettingsConfigDict(
        extra="allow",  # Allow fields not explicitly defined (required for rate limiting config)
        env_prefix="AGENT_",
        env_nested_delimiter="__"
    )
        
    # Core models (only the ones actually used)
    models: Dict[str, ModelConfig] = Field(default_factory=lambda: {
        "default": ModelConfig(),
        "web_research": ModelConfig(),
        "synthesis": ModelConfig(max_tokens=6000)
    })
    
    # Multi-agent configuration
    multi_agent: Dict[str, bool] = Field(default_factory=lambda: {"enabled": True})
    
    agents: Dict[str, Any] = Field(default_factory=lambda: {
        "coordinator": CoordinatorConfig().dict(),
        "planner": PlannerConfig().dict(),
        "researcher": ResearcherConfig().dict(),
        "fact_checker": FactCheckerConfig().dict(),
        "reporter": ReporterConfig().dict()
    })
    
    # Feature configurations
    search: SearchConfig = Field(default_factory=SearchConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    reflexion: ReflexionConfig = Field(default_factory=ReflexionConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    citations: CitationsConfig = Field(default_factory=CitationsConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    quality_enhancement: QualityEnhancementConfig = Field(default_factory=QualityEnhancementConfig)
    adaptive_structure: AdaptiveStructureConfig = Field(default_factory=AdaptiveStructureConfig)
    entity_validation: EntityValidationConfig = Field(default_factory=EntityValidationConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tier_fallback: TierFallbackConfig = Field(default_factory=TierFallbackConfig)

    # System settings
    recursion_limit: int = Field(default=100, ge=10)
    
    @validator("models")
    def validate_models_not_empty(cls, v):
        if not v:
            raise ValueError("Models configuration cannot be empty")
        if "default" not in v:
            raise ValueError("Default model configuration is required")
        return v
    
    @validator("agents")
    def validate_agents_config(cls, v):
        required_agents = ["coordinator", "planner", "researcher", "fact_checker", "reporter"]
        for agent in required_agents:
            if agent not in v:
                raise ValueError(f"Agent '{agent}' configuration is required")
        return v
    
    def get_model_config(self, model_type: str = "default") -> ModelConfig:
        """Get model configuration for specific type"""
        return self.models.get(model_type, self.models["default"])
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get agent configuration by name"""
        return self.agents.get(agent_name, {})
    
    def is_agent_enabled(self, agent_name: str) -> bool:
        """Check if an agent is enabled"""
        agent_config = self.get_agent_config(agent_name)
        return agent_config.get("enabled", False)