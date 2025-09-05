"""
Integration tests for intermediate events functionality.

This module tests the full integration of intermediate events with the
research agent, including event emission during actual agent execution.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from deep_research_agent.research_agent_refactored import RefactoredResearchAgent
from deep_research_agent.core.types import IntermediateEventType, ReasoningVisibility
from deep_research_agent.core.event_emitter import get_event_emitter
from deep_research_agent.core.reasoning_tracer import get_reasoning_tracer


class TestIntermediateEventsIntegration:
    """Test integration of intermediate events with research agent."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            "models": {
                "default": {
                    "endpoint": "databricks-gpt-oss-120b",
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
                "max_retries": 3,
                "search_provider": "brave"
            },
            "intermediate_events": {
                "emit_intermediate_events": True,
                "reasoning_visibility": "summarized",
                "thought_snapshot_interval_tokens": 10,
                "thought_snapshot_interval_ms": 100,
                "max_thought_chars_per_step": 500,
                "redact_patterns": [],
                "max_events_per_second": 20,
                "batch_events": True,
                "batch_size": 5,
                "batch_timeout_ms": 50
            },
            "tools": {
                "brave_search": {
                    "enabled": True,
                    "api_key": "test-api-key",
                    "max_results": 3,
                    "timeout_seconds": 30
                }
            }
        }
    
    @pytest.fixture
    def agent_with_events(self, mock_config, tmp_path):
        """Create agent with intermediate events enabled."""
        # Create temporary config file
        config_file = tmp_path / "test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(mock_config, f)
        
        # Create agent with config
        agent = RefactoredResearchAgent(yaml_path=str(config_file))
        return agent
    
    def test_agent_initialization_with_intermediate_events(self, agent_with_events):
        """Test that agent initializes intermediate events correctly."""
        agent = agent_with_events
        
        # Check that event emitter and reasoning tracer are initialized
        assert hasattr(agent, 'event_emitter')
        assert hasattr(agent, 'reasoning_tracer')
        assert agent.event_emitter is not None
        assert agent.reasoning_tracer is not None
        
        # Check configuration
        assert agent.agent_config.emit_intermediate_events is True
        assert agent.agent_config.reasoning_visibility == ReasoningVisibility.SUMMARIZED
    
    def test_agent_initialization_without_intermediate_events(self, mock_config, tmp_path):
        """Test agent initialization with intermediate events disabled."""
        # Disable intermediate events
        mock_config["intermediate_events"]["emit_intermediate_events"] = False
        
        config_file = tmp_path / "test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(mock_config, f)
        
        agent = RefactoredResearchAgent(yaml_path=str(config_file))
        
        # Should not have event emitter/tracer
        assert not hasattr(agent, 'event_emitter') or agent.event_emitter is None
        assert not hasattr(agent, 'reasoning_tracer') or agent.reasoning_tracer is None
    
    @patch('deep_research_agent.core.search_provider.SearchProvider.search')
    @patch('deep_research_agent.research_agent_refactored.RefactoredResearchAgent._build_workflow_graph')
    def test_intermediate_events_during_prediction(self, mock_build_graph, mock_search, agent_with_events):
        """Test intermediate events emission during agent prediction."""
        agent = agent_with_events
        
        # Mock the workflow graph to return a simple response
        mock_graph = Mock()
        mock_stream_result = [
            ("synthesize_answer", {
                "messages": [Mock(content="Test response")],
                "research_context": Mock(synthesis_chunks=["Test response"])
            })
        ]
        mock_graph.stream.return_value = mock_stream_result
        mock_build_graph.return_value = mock_graph
        
        # Mock search results
        mock_search.return_value = []
        
        # Mock the event emitter to capture events
        collected_events = []
        
        def capture_event(event_data):
            collected_events.append(event_data)
        
        agent.event_emitter.stream_emitter = capture_event
        
        # Create test request
        from mlflow.types.responses import ResponsesAgentRequest
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test question"}]
        )
        
        # Execute prediction
        events = list(agent.predict_stream(request))
        
        # Should have collected intermediate events
        assert len(collected_events) > 0
        
        # Verify event structure
        for event_data in collected_events:
            if event_data.get("type") == "intermediate_event":
                assert "event_type" in event_data
                assert "correlation_id" in event_data
                assert "timestamp" in event_data
    
    def test_search_provider_event_emission(self, agent_with_events):
        """Test that search providers emit events correctly."""
        agent = agent_with_events
        
        # Get search provider from agent
        search_provider = None
        for tool in agent.tool_registry.tools.values():
            if hasattr(tool, 'provider'):
                search_provider = tool.provider
                break
        
        if search_provider is None:
            pytest.skip("No search provider found in agent")
        
        # Mock the actual search to avoid external calls
        original_search_async = search_provider.search_async
        
        async def mock_search_async(query, max_results=5, **kwargs):
            # Simulate some results
            from deep_research_agent.core.search_provider import SearchResult
            return [
                SearchResult(
                    title="Test Result",
                    url="https://example.com",
                    content="Test content",
                    score=0.9
                )
            ]
        
        search_provider.search_async = mock_search_async
        
        # Capture emitted events
        collected_events = []
        
        def capture_event(event_data):
            collected_events.append(event_data)
        
        # Set up event capture
        event_emitter = get_event_emitter()
        event_emitter.stream_emitter = capture_event
        
        try:
            # Perform search
            results = search_provider.search(
                "test query",
                max_results=3,
                correlation_id="test-correlation",
                stage_id="test-stage"
            )
            
            # Should have results
            assert len(results) > 0
            
            # Should have emitted events
            assert len(collected_events) > 0
            
            # Check for tool call events
            tool_events = [e for e in collected_events if e.get("type") == "intermediate_event"]
            assert len(tool_events) > 0
            
            # Verify event types
            event_types = [json.loads(e["delta"])["event_type"] for e in tool_events]
            assert IntermediateEventType.TOOL_CALL_START.value in event_types
            assert IntermediateEventType.TOOL_CALL_COMPLETE.value in event_types
            
        finally:
            # Restore original method
            search_provider.search_async = original_search_async
    
    def test_reasoning_tracer_integration(self, agent_with_events):
        """Test reasoning tracer integration with agent."""
        agent = agent_with_events
        
        # Get reasoning tracer
        tracer = agent.reasoning_tracer
        assert tracer is not None
        
        # Capture emitted events
        collected_events = []
        
        def capture_event(event):
            collected_events.append(event)
        
        tracer.event_emitter = capture_event
        
        # Start a reasoning step
        correlation_id = "test-reasoning-123"
        tracer.start_step(correlation_id, "test-stage")
        
        # Add some reasoning
        tracer.add_reasoning_step("Analyze query", "Breaking down the user question")
        tracer.add_decision_point("Use web search", "Need current information")
        tracer.add_thought("This requires comprehensive search strategy", token_count=5)
        tracer.add_thought(" with multiple sources", token_count=5)  # Should trigger emission
        
        # End step
        tracer.end_step()
        
        # Should have emitted thought snapshots
        assert len(collected_events) > 0
        
        # Verify event structure
        for event in collected_events:
            assert event.event_type == IntermediateEventType.THOUGHT_SNAPSHOT
            assert event.correlation_id == correlation_id
            assert "content" in event.data
            assert event.data["content"] != ""
    
    def test_event_redaction_integration(self, agent_with_events):
        """Test that events are properly redacted in integration."""
        agent = agent_with_events
        
        # Capture events
        collected_events = []
        
        def capture_event(event_data):
            collected_events.append(event_data)
        
        agent.event_emitter.stream_emitter = capture_event
        
        # Emit event with sensitive data
        agent.event_emitter.emit_tool_call_start(
            tool_name="test_tool",
            parameters={
                "query": "Find information about user@company.com",
                "api_key": "sk-1234567890abcdef1234567890abcdef",
                "user_email": "admin@service.org"
            },
            correlation_id="test-redaction"
        )
        
        # Flush batch to ensure event is emitted
        agent.event_emitter.flush_batch()
        
        # Should have captured redacted event
        assert len(collected_events) > 0
        
        event_data = collected_events[0]
        
        # Handle both batched and non-batched event structures
        if event_data.get("type") == "event_batch":
            # Batched event structure
            actual_event = event_data["events"][0]
        else:
            # Direct event structure - check if it's wrapped
            if "delta" in event_data:
                event_json = event_data["delta"]
                actual_event = json.loads(event_json)
            else:
                actual_event = event_data
        
        # Verify redaction
        params = actual_event["data"]["parameters"]
        assert "[REDACTED]" in params["query"]
        assert params["api_key"] == "[REDACTED]"
        assert params["user_email"] == "[REDACTED]"
    
    def test_configuration_impact_on_events(self, mock_config, tmp_path):
        """Test that configuration properly affects event behavior."""
        # Test with hidden reasoning visibility
        mock_config["intermediate_events"]["reasoning_visibility"] = "hidden"
        
        config_file = tmp_path / "hidden_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(mock_config, f)
        
        agent = RefactoredResearchAgent(yaml_path=str(config_file))
        
        # Capture events
        collected_events = []
        agent.reasoning_tracer.event_emitter = lambda e: collected_events.append(e)
        
        # Start reasoning
        agent.reasoning_tracer.start_step("test-correlation")
        agent.reasoning_tracer.add_thought("This should be hidden", token_count=10)
        agent.reasoning_tracer.end_step()
        
        # Should not have emitted thought events (hidden visibility)
        assert len(collected_events) == 0
    
    def test_rate_limiting_in_integration(self, mock_config, tmp_path):
        """Test rate limiting behavior in integration scenario."""
        # Set very low rate limit for testing
        mock_config["intermediate_events"]["max_events_per_second"] = 2
        
        config_file = tmp_path / "rate_limit_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(mock_config, f)
        
        agent = RefactoredResearchAgent(yaml_path=str(config_file))
        
        # Track successful emissions
        emission_count = 0
        
        def count_emissions(event_data):
            nonlocal emission_count
            emission_count += 1
        
        agent.event_emitter.stream_emitter = count_emissions
        
        # Emit many events rapidly
        for i in range(10):
            agent.event_emitter.emit_action_progress(
                action=f"action_{i}",
                status="running",
                progress={"step": i}
            )
        
        # Should be rate limited
        assert emission_count <= 2
        assert agent.event_emitter.stats["events_rate_limited"] > 0
    
    def test_event_batching_in_integration(self, mock_config, tmp_path):
        """Test event batching behavior in integration scenario."""
        # Configure batching
        mock_config["intermediate_events"]["batch_events"] = True
        mock_config["intermediate_events"]["batch_size"] = 3
        
        config_file = tmp_path / "batch_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(mock_config, f)
        
        agent = RefactoredResearchAgent(yaml_path=str(config_file))
        
        # Track batch emissions
        batches_received = []
        
        def capture_batch(event_data):
            batches_received.append(event_data)
        
        agent.event_emitter.stream_emitter = capture_batch
        
        # Emit exactly batch_size events to trigger batching
        for i in range(3):
            agent.event_emitter.emit_action_progress(
                action=f"action_{i}",
                status="running",
                progress={"step": i}
            )
        
        # Should have emitted one batch
        assert len(batches_received) == 1
        
        batch_data = batches_received[0]
        assert batch_data["type"] == "event_batch"
        assert batch_data["batch_size"] == 3
        assert len(batch_data["events"]) == 3
    
    def test_error_handling_in_event_emission(self, agent_with_events):
        """Test error handling when event emission fails."""
        agent = agent_with_events
        
        # Disable batching to test immediate emission failure
        agent.event_emitter.batch_events = False
        
        # Mock stream emitter to raise exceptions
        def failing_emitter(event_data):
            raise Exception("Stream emission failed")
        
        agent.event_emitter.stream_emitter = failing_emitter
        
        # Emit event - should not raise exception
        result = agent.event_emitter.emit_action_start(
            action="test_action",
            query="test query"
        )
        
        # Should return False and track error
        assert result is False
        assert agent.event_emitter.stats["events_dropped"] > 0
    
    def test_correlation_id_consistency(self, agent_with_events):
        """Test that correlation IDs are consistent across related events."""
        agent = agent_with_events
        
        # Capture events
        collected_events = []
        
        def capture_event(event_data):
            collected_events.append(event_data)
        
        agent.event_emitter.stream_emitter = capture_event
        
        correlation_id = "workflow-consistency-test"
        
        # Emit a sequence of related events
        agent.event_emitter.emit_action_start(
            action="research_workflow",
            query="test query",
            correlation_id=correlation_id
        )
        
        agent.event_emitter.emit_action_progress(
            action="query_generation",
            status="running",
            correlation_id=correlation_id
        )
        
        agent.event_emitter.emit_tool_call_start(
            tool_name="search",
            parameters={"query": "test"},
            correlation_id=correlation_id
        )
        
        agent.event_emitter.emit_action_complete(
            action="research_workflow",
            result_summary="completed",
            correlation_id=correlation_id
        )
        
        # All events should have the same correlation_id
        for event_data in collected_events:
            if event_data.get("type") == "intermediate_event":
                event_json = event_data["delta"]
                parsed_event = json.loads(event_json)
                assert parsed_event["correlation_id"] == correlation_id
    
    def test_performance_with_events_enabled(self, agent_with_events):
        """Test that intermediate events don't significantly impact performance."""
        agent = agent_with_events
        
        # Measure time with events
        start_time = time.time()
        
        # Emit many events
        for i in range(100):
            agent.event_emitter.emit_action_progress(
                action=f"action_{i}",
                status="running",
                progress={"step": i, "percentage": i}
            )
        
        end_time = time.time()
        
        # Should complete quickly (< 1 second for 100 events)
        elapsed_time = end_time - start_time
        assert elapsed_time < 1.0
        
        # Verify some events were emitted
        stats = agent.event_emitter.get_stats()
        assert stats["events_emitted"] > 0


class TestIntermediateEventsEndToEnd:
    """End-to-end tests for intermediate events functionality."""
    
    @pytest.fixture
    def full_config(self):
        """Full configuration for end-to-end testing."""
        return {
            "models": {
                "default": {
                    "endpoint": "databricks-gpt-oss-120b",
                    "temperature": 0.7,
                    "max_tokens": 4000
                }
            },
            "research": {
                "max_research_loops": 1,
                "initial_query_count": 1,
                "enable_streaming": True,
                "enable_citations": True,
                "search_provider": "brave"
            },
            "intermediate_events": {
                "emit_intermediate_events": True,
                "reasoning_visibility": "summarized",
                "thought_snapshot_interval_tokens": 5,
                "max_thought_chars_per_step": 200,
                "max_events_per_second": 50,
                "batch_events": True,
                "batch_size": 3
            },
            "tools": {
                "brave_search": {
                    "enabled": True,
                    "api_key": "test-key",
                    "max_results": 2
                }
            }
        }
    
    @patch('deep_research_agent.tools_brave_refactored.BraveSearchProvider.search_async')
    def test_complete_workflow_with_events(self, mock_search, full_config, tmp_path):
        """Test complete workflow with intermediate events from start to finish."""
        # Setup config
        config_file = tmp_path / "full_test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(full_config, f)
        
        # Mock search results
        from deep_research_agent.core.search_provider import SearchResult
        mock_search.return_value = [
            SearchResult(
                title="Test Result 1",
                url="https://example1.com",
                content="First test result content",
                score=0.9
            ),
            SearchResult(
                title="Test Result 2", 
                url="https://example2.com",
                content="Second test result content",
                score=0.8
            )
        ]
        
        # Create agent
        agent = RefactoredResearchAgent(yaml_path=str(config_file))
        
        # Collect all events
        all_events = []
        
        def collect_all_events(event_data):
            all_events.append(event_data)
        
        agent.event_emitter.stream_emitter = collect_all_events
        
        # Mock LLM to avoid external calls
        with patch.object(agent, 'llm') as mock_llm:
            mock_response = Mock()
            mock_response.content = "Test analysis based on search results"
            mock_llm.invoke.return_value = mock_response
            mock_llm.stream.return_value = [mock_response]
            
            # Create test request
            from mlflow.types.responses import ResponsesAgentRequest
            request = ResponsesAgentRequest(
                input=[{"role": "user", "content": "What are the latest ML trends?"}]
            )
            
            # Execute complete workflow
            try:
                response_events = list(agent.predict_stream(request))
                
                # Should have response events
                assert len(response_events) > 0
                
                # Should have collected intermediate events
                assert len(all_events) > 0
                
                # Analyze collected events
                intermediate_events = [e for e in all_events if e.get("type") in ["intermediate_event", "event_batch"]]
                assert len(intermediate_events) > 0
                
                # Check for expected event types
                event_types = set()
                for event_data in intermediate_events:
                    if event_data.get("type") == "intermediate_event":
                        event_json = event_data["delta"]
                        parsed_event = json.loads(event_json)
                        event_types.add(parsed_event["event_type"])
                    elif event_data.get("type") == "event_batch":
                        for event in event_data["events"]:
                            event_types.add(event["event_type"])
                
                # Should include workflow events
                expected_types = {
                    IntermediateEventType.ACTION_START.value,
                    IntermediateEventType.ACTION_PROGRESS.value,
                    IntermediateEventType.ACTION_COMPLETE.value
                }
                
                # At least some expected types should be present
                assert len(event_types.intersection(expected_types)) > 0
                
                print(f"Collected {len(all_events)} total events")
                print(f"Event types found: {event_types}")
                
            except Exception as e:
                pytest.fail(f"End-to-end test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
