"""Framework integration adapters.

These adapters bridge the existing Deep Research application with the
standalone ``databricks-deep-research`` framework.

Adapters:
    llm_adapter: Wraps app LLM client → FrameworkLLMClient
    tool_adapter: Creates framework tools from app config
    domain_context: Event forwarding and persistence delta tracking
    config_translator: OrchestrationConfig → WorkflowDefinition
    checkpoint_adapter: DB persistence → CheckpointHandler protocol
"""
