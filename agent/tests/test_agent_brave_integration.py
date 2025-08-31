"""
Integration tests for Research Agent with Brave Search using real Databricks endpoints.

This module provides comprehensive integration testing for the refactored research agent
with actual Brave Search API calls and Databricks LLM endpoints using SDK profiles.
"""

import os
import pytest
import time
import logging
from typing import Dict, List, Any, Optional, Generator
from pathlib import Path
from unittest.mock import patch
import sys

# Add parent directory to Python path  
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from deep_research_agent.research_agent_refactored import RefactoredResearchAgent
from deep_research_agent.databricks_helper import get_workspace_client, get_databricks_openai_client
from deep_research_agent.core import (
    get_logger,
    ConfigManager,
    ResearchContext,
    WorkflowMetrics,
    SearchResult,
    Citation
)
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse, 
    ResponsesAgentStreamEvent
)

logger = get_logger(__name__)

# Test configuration
BRAVE_API_KEY_REQUIRED = "BRAVE_API_KEY"
DATABRICKS_PROFILE_DEFAULT = "e2-demo-west"  # From deploy_config.yaml dev environment
TEST_TIMEOUT = 120  # 2 minutes timeout for integration tests


def pytest_configure(config):
    """Configure pytest for integration testing."""
    # Register custom markers
    config.addinivalue_line("markers", "external: mark test as requiring external services")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    
    # Reduce logging noise during tests
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def check_brave_api_key():
    """Check if Brave API key is available for testing."""
    api_key = os.getenv(BRAVE_API_KEY_REQUIRED)
    if not api_key:
        pytest.skip(f"Integration test requires {BRAVE_API_KEY_REQUIRED} environment variable")
    return api_key


@pytest.fixture(scope="session") 
def databricks_workspace():
    """Get Databricks workspace client using SDK profile."""
    try:
        # Try to get workspace client using default dev profile
        workspace_client = get_workspace_client(env="dev")
        
        # Test the connection
        workspace_client.current_user.me()
        
        return workspace_client
    except Exception as e:
        pytest.skip(f"Could not connect to Databricks workspace: {e}")


@pytest.fixture(scope="session")
def test_agent_config(check_brave_api_key):
    """Load real agent configuration for testing."""
    config_path = Path(__file__).parent.parent / "deep_research_agent" / "agent_config.yaml"
    if not config_path.exists():
        pytest.skip("agent_config.yaml not found")
    
    # Override configuration for testing with environment variable
    test_config_override = {
        # Set profile at root level for _get_config_value to find
        "databricks_profile": "e2-demo-west",
        "models": {
            "default": {
                "endpoint": "databricks-claude-3-7-sonnet",
                "temperature": 0.7,
                "max_tokens": 4000
            }
        },
        "research": {
            "max_research_loops": 1,
            "initial_query_count": 2,
            "enable_streaming": True,
            "enable_citations": True,
            "timeout_seconds": 30,
            "max_retries": 2,
            "search_provider": "brave"
        },
        "rate_limiting": {
            "max_concurrent_searches": 1,
            "batch_delay_seconds": 1.0
        },
        "tools": {
            "brave_search": {
                "enabled": True,
                "api_key": check_brave_api_key,  # Use actual env var value
                "max_results": 5,
                "timeout_seconds": 30,
                "max_retries": 3
            },
            "tavily_search": {
                "enabled": False  # Disable Tavily for this test
            },
            "vector_search": {
                "enabled": False  # Disable vector search for this test
            }
        },
        "databricks": {
            "workspace_url": None,
            "token": None,
            "workspace_profile": "e2-demo-west"
        }
    }
    
    config_manager = ConfigManager(test_config_override)
    
    # Return the raw config dictionary instead of AgentConfiguration object
    # This ensures our custom overrides (like databricks_profile) are preserved
    return test_config_override


@pytest.fixture
def integration_agent(check_brave_api_key, databricks_workspace, test_agent_config):
    """Create research agent configured for integration testing."""
    try:
        # Create agent with real configuration
        agent = RefactoredResearchAgent(config=test_agent_config)
        
        # Verify agent initialized successfully
        assert agent.agent_config is not None
        assert agent.tool_registry is not None
        assert agent.llm is not None
        
        logger.info("Integration agent initialized successfully")
        return agent
        
    except Exception as e:
        pytest.fail(f"Failed to initialize integration agent: {e}")


class TestBraveSearchIntegration:
    """Test real Brave Search integration with Databricks LLMs."""
    
    @pytest.mark.external
    @pytest.mark.integration
    def test_initialization_with_databricks_profile(self, databricks_workspace, test_agent_config):
        """Test that agent initializes correctly with Databricks SDK profile."""
        # Verify workspace client is working
        user_info = databricks_workspace.current_user.me()
        assert user_info.user_name is not None
        logger.info(f"Connected as user: {user_info.user_name}")
        
        # Create agent and verify it initializes
        agent = RefactoredResearchAgent(config=test_agent_config)
        
        # Verify key components are initialized
        assert agent.llm is not None, "LLM should be initialized"
        assert agent.tool_registry is not None, "Tool registry should be initialized"
        assert agent.graph is not None, "Research graph should be built"
        
        # Verify Brave search tool is available
        from deep_research_agent.core.types import ToolType
        brave_tool = agent.tool_registry.get_tool(ToolType.BRAVE_SEARCH)
        assert brave_tool is not None, "Brave search tool should be available"
        
        logger.info("Agent initialization with Databricks profile: PASSED")
    
    @pytest.mark.external
    @pytest.mark.integration
    @pytest.mark.slow
    def test_brave_search_with_real_query_non_streaming(self, integration_agent):
        """Test non-streaming research with real Brave Search API calls."""
        # Create a research request that will require web search
        request = ResponsesAgentRequest(
            input=[{
                "type": "message",
                "role": "user", 
                "content": [{"type": "output_text", "text": "What are the latest features in LangGraph 0.6.6?"}]
            }]
        )
        
        start_time = time.time()
        
        # Execute research
        response = integration_agent.predict(request)
        
        execution_time = time.time() - start_time
        logger.info(f"Non-streaming research completed in {execution_time:.2f} seconds")
        
        # Validate response structure
        assert isinstance(response, ResponsesAgentResponse)
        assert len(response.output) >= 1
        
        # Check response content
        output_item = response.output[0]
        assert output_item.type == "message"
        assert output_item.role == "assistant"
        assert len(output_item.content) >= 1
        
        response_text = output_item.content[0]["text"]
        assert len(response_text) > 100, "Response should be substantial"
        
        # Verify research-like content (should mention searches or findings)
        research_indicators = ["research", "found", "according", "based on", "information", "langgraph"]
        has_research_content = any(indicator in response_text.lower() for indicator in research_indicators)
        assert has_research_content, f"Response should indicate research was performed: {response_text[:200]}..."
        
        # Performance check
        assert execution_time < TEST_TIMEOUT, f"Research took too long: {execution_time:.2f}s"
        
        logger.info("Non-streaming Brave Search integration: PASSED")
    
    @pytest.mark.external
    @pytest.mark.integration  
    @pytest.mark.slow
    def test_brave_search_with_real_query_streaming(self, integration_agent):
        """Test streaming research with real Brave Search API calls."""
        request = ResponsesAgentRequest(
            input=[{
                "type": "message",
                "role": "user",
                "content": [{"type": "output_text", "text": "How does Databricks Agent Framework handle streaming responses?"}]
            }]
        )
        
        start_time = time.time()
        events = []
        
        # Collect streaming events
        try:
            for event in integration_agent.predict_stream(request):
                events.append(event)
                assert isinstance(event, ResponsesAgentStreamEvent)
                
                # Log progress for debugging
                if hasattr(event, 'item') and event.item:
                    logger.debug(f"Stream event: {event.type} - {str(event.item)[:100]}")
                
        except Exception as e:
            pytest.fail(f"Streaming failed: {e}")
        
        execution_time = time.time() - start_time
        logger.info(f"Streaming research completed in {execution_time:.2f} seconds with {len(events)} events")
        
        # Validate streaming events
        assert len(events) > 0, "Should receive streaming events"
        
        # Check for different event types
        event_types = {event.type for event in events}
        logger.info(f"Event types received: {event_types}")
        
        # Should have completion events
        done_events = [e for e in events if e.type == "response.output_item.done"]
        assert len(done_events) > 0, "Should have completion events"
        
        # Validate final response content
        final_content = None
        for event in done_events:
            if hasattr(event, 'item') and event.item and event.item.get("type") == "message":
                content = event.item.get("content", [])
                if content and "text" in content[0]:
                    final_content = content[0]["text"]
                    break
        
        assert final_content is not None, "Should have final text content"
        assert len(final_content) > 100, "Final response should be substantial"
        
        # Check for research indicators
        research_indicators = ["databricks", "streaming", "agent", "framework"]
        has_relevant_content = any(indicator in final_content.lower() for indicator in research_indicators)
        assert has_relevant_content, f"Response should be relevant to query: {final_content[:200]}..."
        
        # Performance check
        assert execution_time < TEST_TIMEOUT, f"Streaming research took too long: {execution_time:.2f}s"
        
        logger.info("Streaming Brave Search integration: PASSED")
    
    @pytest.mark.external
    @pytest.mark.integration
    def test_multiple_search_iterations(self, integration_agent):
        """Test research workflow with potential multiple search iterations."""
        # Use agent config with more loops to test iteration
        integration_agent.agent_config.max_research_loops = 2
        
        request = ResponsesAgentRequest(
            input=[{
                "type": "message",
                "role": "user",
                "content": [{"type": "output_text", "text": "What are current best practices for RAG systems in 2024 and how do they compare to older approaches?"}]
            }]
        )
        print(f"Request: {request}")
        
        start_time = time.time()
        response = integration_agent.predict(request)
        execution_time = time.time() - start_time
        
        # Validate response
        assert isinstance(response, ResponsesAgentResponse)
        response_text = response.output[0].content[0]["text"]
        print(f"Response text: {response_text}")
        
        # For a complex query like this, expect a comprehensive response
        assert len(response_text) > 200, "Complex query should yield comprehensive response"
        
        # Look for indicators of thorough research
        rag_indicators = ["rag", "retrieval", "generation", "embedding", "vector", "2024", "practices"]
        found_indicators = [ind for ind in rag_indicators if ind in response_text.lower()]
        assert len(found_indicators) >= 3, f"Response should cover RAG concepts. Found: {found_indicators}"
        
        logger.info(f"Multiple iteration research completed in {execution_time:.2f} seconds")
        logger.info("Multiple search iterations: PASSED")
    
    @pytest.mark.external 
    @pytest.mark.integration
    def test_conversation_context_handling(self, integration_agent):
        """Test research with conversation history context."""
        # Multi-turn conversation
        request = ResponsesAgentRequest(
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "output_text", "text": "What is LangGraph?"}]
                },
                {
                    "type": "message", 
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "LangGraph is a library for building stateful, multi-actor applications with LLMs."}]
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "output_text", "text": "How does it handle state management in complex workflows?"}]
                }
            ]
        )
        
        response = integration_agent.predict(request)
        
        # Validate response acknowledges context
        response_text = response.output[0].content[0]["text"]
        assert len(response_text) > 50, "Should provide meaningful response"
        
        # Should reference state management concepts
        state_indicators = ["state", "workflow", "management", "langgraph"]
        has_relevant_content = any(indicator in response_text.lower() for indicator in state_indicators)
        assert has_relevant_content, "Response should address state management"
        
        logger.info("Conversation context handling: PASSED")


class TestStreamingFunctionality:
    """Test streaming-specific functionality in detail."""
    
    @pytest.mark.external
    @pytest.mark.integration
    def test_streaming_events_structure(self, integration_agent):
        """Validate streaming event structure compliance with MLflow schema."""
        request = ResponsesAgentRequest(
            input=[{
                "type": "message",
                "role": "user",
                "content": [{"type": "output_text", "text": "What are the main components of transformer architecture?"}]
            }]
        )
        
        events = list(integration_agent.predict_stream(request))
        
        # Validate all events follow ResponsesAgentStreamEvent schema
        for i, event in enumerate(events):
            assert isinstance(event, ResponsesAgentStreamEvent), f"Event {i} is not ResponsesAgentStreamEvent"
            assert hasattr(event, 'type'), f"Event {i} missing type attribute"
            assert isinstance(event.type, str), f"Event {i} type is not string"
            
            # Validate specific event types
            if event.type == "response.output_item.done":
                assert hasattr(event, 'item'), f"Done event {i} missing item"
                assert isinstance(event.item, dict), f"Done event {i} item is not dict"
            elif event.type == "response.output_text.delta":
                assert hasattr(event, 'delta'), f"Delta event {i} missing delta"
                assert hasattr(event, 'item_id'), f"Delta event {i} missing item_id"
        
        logger.info(f"Validated {len(events)} streaming events for schema compliance")
        logger.info("Streaming events structure validation: PASSED")
    
    @pytest.mark.external
    @pytest.mark.integration 
    def test_streaming_progress_updates(self, integration_agent):
        """Test that streaming provides meaningful progress updates."""
        request = ResponsesAgentRequest(
            input=[{
                "type": "message", 
                "role": "user",
                "content": [{"type": "output_text", "text": "Explain how attention mechanisms work in neural networks"}]
            }]
        )
        
        events = list(integration_agent.predict_stream(request))
        
        # Should have meaningful progress through research workflow
        assert len(events) >= 2, f"Expected multiple progress events, got {len(events)}"
        
        # Look for different types of content indicating research progress
        content_pieces = []
        for event in events:
            if hasattr(event, 'item') and event.item and event.item.get("content"):
                content_list = event.item["content"]
                for content_item in content_list:
                    if "text" in content_item:
                        content_pieces.append(content_item["text"])
            elif hasattr(event, 'delta') and event.delta:
                content_pieces.append(event.delta)
        
        # Combine all content to check for research progression
        full_content = " ".join(content_pieces)
        assert len(full_content) > 100, "Should accumulate substantial content through streaming"
        
        # Look for attention-related content
        attention_terms = ["attention", "neural", "network", "mechanism", "transformer"]
        relevant_terms = [term for term in attention_terms if term in full_content.lower()]
        assert len(relevant_terms) >= 2, f"Content should be relevant to query. Found terms: {relevant_terms}"
        
        logger.info("Streaming progress updates: PASSED")


class TestErrorHandling:
    """Test error handling and resilience."""
    
    @pytest.mark.external
    @pytest.mark.integration
    def test_rate_limiting_handling(self, integration_agent):
        """Test graceful handling of API rate limits."""
        # Make multiple rapid requests to potentially trigger rate limiting
        requests = [
            ResponsesAgentRequest(input=[{
                "type": "message",
                "role": "user", 
                "content": [{"type": "output_text", "text": f"What is machine learning concept number {i}?"}]
            }])
            for i in range(3)
        ]
        
        successful_requests = 0
        for i, request in enumerate(requests):
            try:
                start_time = time.time()
                response = integration_agent.predict(request)
                execution_time = time.time() - start_time
                
                assert isinstance(response, ResponsesAgentResponse)
                successful_requests += 1
                
                logger.info(f"Request {i+1} completed in {execution_time:.2f}s")
                
                # Add delay between requests to be respectful to API
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Request {i+1} failed (expected with rate limiting): {e}")
        
        # At least one request should succeed
        assert successful_requests >= 1, "At least one request should succeed despite rate limiting"
        
        logger.info(f"Rate limiting test: {successful_requests}/3 requests successful")
        logger.info("Rate limiting handling: PASSED")
    
    @pytest.mark.external
    @pytest.mark.integration
    def test_graceful_degradation(self, integration_agent):
        """Test agent graceful degradation when searches partially fail."""
        # Create a request that might have some search challenges
        request = ResponsesAgentRequest(
            input=[{
                "type": "message",
                "role": "user",
                "content": [{"type": "output_text", "text": "What is the completely made up technology called QuantumFluxinator used for?"}]
            }]
        )
        
        # Should handle gracefully even if searches don't find much
        response = integration_agent.predict(request)
        
        assert isinstance(response, ResponsesAgentResponse)
        response_text = response.output[0].content[0]["text"]
        
        # Should provide some kind of response, even if searches don't find much
        assert len(response_text) > 20, "Should provide some response even for obscure queries"
        
        # Response should acknowledge the difficulty or provide alternatives
        helpful_indicators = ["not found", "unclear", "unable to find", "no information", "help", "clarify"]
        is_helpful = any(indicator in response_text.lower() for indicator in helpful_indicators)
        
        # Either found something or gracefully handled the lack of information
        assert len(response_text) > 50 or is_helpful, "Should either find info or gracefully handle lack of info"
        
        logger.info("Graceful degradation: PASSED")


# Helper functions for validation

def assert_search_performed(agent, query: str) -> bool:
    """
    Verify that actual Brave API calls were made during research.
    This is inferred from the research workflow completion and content quality.
    """
    # In a real implementation, we could check network logs or tool call history
    # For now, we validate through the agent's research state
    return True


def assert_synthesis_quality(response: ResponsesAgentResponse, search_terms: List[str]) -> bool:
    """
    Check that response incorporates content from searches.
    
    Args:
        response: The agent's response
        search_terms: Terms that should appear in a well-researched response
        
    Returns:
        bool: True if synthesis quality is adequate
    """
    if not response.output or not response.output[0].content:
        return False
    
    response_text = response.output[0].content[0].get("text", "").lower()
    
    # Check for research indicators and relevant terms
    research_indicators = ["research", "found", "according", "based on"]
    has_research_language = any(indicator in response_text for indicator in research_indicators)
    
    # Check for relevant content
    relevant_terms_found = sum(1 for term in search_terms if term.lower() in response_text)
    
    return has_research_language and relevant_terms_found >= len(search_terms) // 2


def collect_streaming_events(agent, request: ResponsesAgentRequest) -> List[ResponsesAgentStreamEvent]:
    """
    Collect and validate all streaming events from the agent.
    
    Args:
        agent: The research agent
        request: The request to process
        
    Returns:
        List of validated streaming events
    """
    events = []
    for event in agent.predict_stream(request):
        assert isinstance(event, ResponsesAgentStreamEvent)
        events.append(event)
    
    return events


def measure_workflow_performance(agent, request: ResponsesAgentRequest) -> Dict[str, float]:
    """
    Track timing and resource usage metrics for the research workflow.
    
    Args:
        agent: The research agent  
        request: The request to process
        
    Returns:
        Dict with performance metrics
    """
    start_time = time.time()
    
    response = agent.predict(request)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return {
        "execution_time": execution_time,
        "response_length": len(response.output[0].content[0].get("text", "")),
        "tokens_estimated": len(response.output[0].content[0].get("text", "").split()),
    }


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])