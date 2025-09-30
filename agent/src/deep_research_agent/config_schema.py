"""
Pydantic configuration schema for the research agent.

This module defines the complete configuration structure with type safety,
validation, and environment variable support.
"""

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
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
    """Model endpoint configuration"""
    endpoint: str = Field(default="databricks-gpt-oss-120b")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4000, gt=0, le=32000)
    # Reasoning-specific configuration for models like GPT-OSS and Claude
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

class ReporterConfig(BaseSettings):
    """Reporter agent configuration"""
    enabled: bool = True
    model: str = "synthesis"
    default_style: ReportStyle = ReportStyle.DEFAULT
    citation_style: str = "APA"
    enable_grounding_markers: bool = True
    use_semantic_extraction: bool = True

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

class RateLimitingConfig(BaseSettings):
    """Rate limiting configuration"""
    max_parallel_requests: int = Field(default=10, ge=1)
    max_requests_per_minute: int = Field(default=60, ge=1)
    max_tokens_per_minute: int = Field(default=100000, ge=1000)

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
    max_observations_per_step: int = Field(default=10, ge=5, le=100, description="Maximum observations per research step")
    max_search_results: int = Field(default=100, ge=10, le=1000, description="Maximum search results to accumulate")

class ResearchConfig(BaseSettings):
    """Main research agent configuration"""
    
    class Config:
        env_prefix = "AGENT_"
        env_nested_delimiter = "__"
        extra = "forbid"  # Reject unknown fields
        
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