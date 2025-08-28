# Agent authoring package for Databricks deployment

try:
    # Import from research_agent_refactored.py (Phase 2 implementation)
    from research_agent_refactored import RefactoredResearchAgent
    from tools_tavily import TavilySearchTool, create_tavily_tool
    from tools_vector_search import setup_vector_search_retriever, create_vector_search_tool
    
    # Backward compatibility aliases
    DatabricksResearchAgent = RefactoredResearchAgent
    
    # Define constants locally to avoid backend dependencies
    UC_TOOL_NAMES = ["system.ai.python_exec"]
    AGENT = None  # Will be initialized when needed
    
    __all__ = [
        "RefactoredResearchAgent",
        "DatabricksResearchAgent",  # Backward compatibility
        "TavilySearchTool",
        "create_tavily_tool",
        "setup_vector_search_retriever",
        "create_vector_search_tool",
        "UC_TOOL_NAMES"
    ]
except ImportError as e:
    # Handle import errors gracefully for development environments
    print(f"Warning: Some imports failed during package initialization: {e}")
    __all__ = []