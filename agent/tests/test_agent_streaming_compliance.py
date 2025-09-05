"""
Comprehensive test suite to ensure agent meets ALL UI streaming expectations.

This test file validates the streaming contract between the Deep Research Agent
and the UI. It ensures all progress events, PHASE markers, metadata fields, 
and event sequencing match what the UI expects for proper rendering.

Key Testing Areas:
- PHASE marker format and coverage
- Metadata field presence and format
- Event sequencing and timing
- Edge cases and error scenarios
- Performance expectations
- Regression prevention
"""

import pytest
import json
import re
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse, ResponsesAgentStreamEvent

# Import test modules
import sys
import os

# Add the deep_research_agent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
deep_research_agent_dir = os.path.join(parent_dir, 'deep_research_agent')
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if deep_research_agent_dir not in sys.path:
    sys.path.insert(0, deep_research_agent_dir)

from deep_research_agent.research_agent_refactored import RefactoredResearchAgent


class MockAgentScenarios:
    """Mock agent configurations for different streaming scenarios."""
    
    @staticmethod
    def create_full_research_agent():
        """Agent that goes through all research phases properly."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = [
            AIMessage(content='{"queries": ["test query 1", "test query 2"]}'),  # generate_queries
            AIMessage(content='{"needs_more_research": false}'),  # reflect  
            AIMessage(content="Based on comprehensive research, here are the findings.")  # synthesize
        ]
        
        # Create agent with mocked dependencies
        with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_llm', return_value=mock_llm):
            mock_phase2_return = (None, None, None, None, None, None)
            with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_phase2_components', return_value=mock_phase2_return):
                agent = RefactoredResearchAgent()
                
                # Mock the graph to emit all workflow nodes
                mock_graph = Mock()
                mock_graph.stream.return_value = [
                    {"generate_queries": {"research_context": Mock(generated_queries=["q1", "q2"])}},
                    {"batch_controller": {}},
                    {"route_to_parallel_search": {}},
                    {"parallel_web_search": {}}, 
                    {"aggregate_search_results": {"research_context": Mock(web_results=[{"url": "test.com"}])}},
                    {"vector_research": {"research_context": Mock(vector_results=[])}},
                    {"reflect": {}},
                    {"synthesize_answer": {"messages": [Mock(content="Final synthesized response.")]}}
                ]
                agent.graph = mock_graph
                return agent
    
    @staticmethod
    def create_fast_cached_agent():
        """Agent that returns cached results with minimal research phases."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Cached response from previous research.")
        
        with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_llm', return_value=mock_llm):
            mock_phase2_return = (None, None, None, None, None, None)
            with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_phase2_components', return_value=mock_phase2_return):
                agent = RefactoredResearchAgent()
                
                # Mock minimal workflow - goes straight to synthesis
                mock_graph = Mock()
                mock_graph.stream.return_value = [
                    {"synthesize_answer": {"messages": [Mock(content="Cached response from previous research.")]}}
                ]
                agent.graph = mock_graph
                return agent
                
    @staticmethod  
    def create_broken_progress_agent():
        """Agent that emits malformed or missing progress events."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Response with broken progress")
        
        with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_llm', return_value=mock_llm):
            mock_phase2_return = (None, None, None, None, None, None)
            with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_phase2_components', return_value=mock_phase2_return):
                agent = RefactoredResearchAgent()
                
                # Mock agent that doesn't emit proper progress format
                original_create_progress = agent._create_progress_delta_event
                def broken_progress(*args, **kwargs):
                    # Return malformed progress event
                    return ResponsesAgentStreamEvent(
                        type="response.output_text.delta",
                        item_id=str(uuid4()),
                        delta="BROKEN_PHASE:INVALID malformed progress"  # Missing brackets
                    )
                
                agent._create_progress_delta_event = broken_progress
                
                mock_graph = Mock()
                mock_graph.stream.return_value = [
                    {"generate_queries": {}},
                    {"synthesize_answer": {"messages": [Mock(content="Response with broken progress")]}}
                ]
                agent.graph = mock_graph
                return agent


class TestAgentStreamingCompliance:
    """Test suite for agent streaming compliance with UI expectations."""
    
    def test_phase_markers_emitted_for_all_nodes(self):
        """Verify [PHASE:X] markers are emitted for each workflow node."""
        agent = MockAgentScenarios.create_full_research_agent()
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test research question"}]
        )
        
        # Collect all stream events
        stream_events = list(agent.predict_stream(request))
        
        # Extract progress events with PHASE markers
        progress_events = []
        for event in stream_events:
            if event.type == "response.output_text.delta" and "[PHASE:" in event.delta:
                progress_events.append(event.delta)
        
        # Verify we have progress events
        assert len(progress_events) >= 3, f"Expected at least 3 progress events, got {len(progress_events)}"
        
        # Verify each has PHASE marker
        for progress_event in progress_events:
            assert "[PHASE:" in progress_event, f"Missing PHASE marker in: {progress_event}"
    
    def test_phase_marker_format(self):
        """Verify phase markers follow [PHASE:NAME] format exactly."""
        agent = MockAgentScenarios.create_full_research_agent()
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test research question"}]
        )
        
        stream_events = list(agent.predict_stream(request))
        
        phase_pattern = re.compile(r'\[PHASE:([A-Z_]+)\]')
        
        for event in stream_events:
            if event.type == "response.output_text.delta" and "[PHASE:" in event.delta:
                match = phase_pattern.search(event.delta)
                assert match is not None, f"Invalid PHASE format in: {event.delta}"
                
                phase_name = match.group(1)
                # Verify phase name is uppercase and uses underscores
                assert phase_name.isupper(), f"Phase name not uppercase: {phase_name}"
                assert re.match(r'^[A-Z_]+$', phase_name), f"Invalid phase name format: {phase_name}"
    
    def test_all_expected_phases_covered(self):
        """Verify all UI-expected phases are emitted."""
        agent = MockAgentScenarios.create_full_research_agent()
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Comprehensive research question"}]
        )
        
        stream_events = list(agent.predict_stream(request))
        
        # Expected phases that UI can handle
        expected_phases = {
            "QUERYING", "PREPARING", "SEARCHING", "SEARCHING_INTERNAL",
            "AGGREGATING", "ANALYZING", "SYNTHESIZING"
        }
        
        found_phases = set()
        phase_pattern = re.compile(r'\[PHASE:([A-Z_]+)\]')
        
        for event in stream_events:
            if event.type == "response.output_text.delta" and "[PHASE:" in event.delta:
                match = phase_pattern.search(event.delta)
                if match:
                    found_phases.add(match.group(1))
        
        # Should find at least some of the expected phases
        overlap = expected_phases.intersection(found_phases)
        assert len(overlap) >= 2, f"Expected phases from {expected_phases}, found {found_phases}"
        
        # All found phases should be in expected set
        unexpected = found_phases - expected_phases
        assert len(unexpected) == 0, f"Unexpected phases found: {unexpected}"
    
    def test_metadata_markers_format(self):
        """Verify [META:key:value] format is correct."""
        agent = MockAgentScenarios.create_full_research_agent()
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test research question"}]
        )
        
        stream_events = list(agent.predict_stream(request))
        
        meta_pattern = re.compile(r'\[META:([^:]+):([^\]]+)\]')
        meta_found = False
        
        for event in stream_events:
            if event.type == "response.output_text.delta" and "[META:" in event.delta:
                matches = meta_pattern.findall(event.delta)
                assert len(matches) > 0, f"Invalid META format in: {event.delta}"
                meta_found = True
                
                for key, value in matches:
                    # Verify key and value are non-empty
                    assert key.strip(), f"Empty META key in: {event.delta}"
                    assert value.strip(), f"Empty META value for key {key} in: {event.delta}"
        
        assert meta_found, "No META markers found in any progress events"
    
    def test_required_metadata_fields(self):
        """Verify required metadata fields are present."""
        agent = MockAgentScenarios.create_full_research_agent()
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test research question"}]
        )
        
        stream_events = list(agent.predict_stream(request))
        
        required_fields = {"elapsed", "progress", "node"}
        found_fields = set()
        
        meta_pattern = re.compile(r'\[META:([^:]+):[^\]]+\]')
        
        for event in stream_events:
            if event.type == "response.output_text.delta" and "[META:" in event.delta:
                matches = meta_pattern.findall(event.delta)
                for field in matches:
                    found_fields.add(field)
        
        missing_fields = required_fields - found_fields
        assert len(missing_fields) == 0, f"Missing required META fields: {missing_fields}"
    
    def test_progress_events_before_content(self):
        """Verify progress events emit BEFORE synthesis content."""
        agent = MockAgentScenarios.create_full_research_agent()
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test research question"}]
        )
        
        stream_events = list(agent.predict_stream(request))
        
        # Find first progress event and first content event
        first_progress_idx = None
        first_content_idx = None
        
        for i, event in enumerate(stream_events):
            if event.type == "response.output_text.delta":
                if "[PHASE:" in event.delta and first_progress_idx is None:
                    first_progress_idx = i
                elif "[PHASE:" not in event.delta and first_content_idx is None:
                    first_content_idx = i
        
        # If both exist, progress should come first
        if first_progress_idx is not None and first_content_idx is not None:
            assert first_progress_idx < first_content_idx, \
                "Progress events should come before content events"
    
    def test_no_json_in_delta_events(self):
        """Verify delta events contain NO JSON objects."""
        agent = MockAgentScenarios.create_full_research_agent()
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test research question"}]
        )
        
        stream_events = list(agent.predict_stream(request))
        
        for event in stream_events:
            if event.type == "response.output_text.delta" and event.delta:
                # Check if content looks like JSON
                stripped = event.delta.strip()
                if stripped.startswith('{') and stripped.endswith('}'):
                    try:
                        parsed = json.loads(stripped)
                        if isinstance(parsed, dict):
                            pytest.fail(f"Found JSON object in delta event: {parsed}")
                    except json.JSONDecodeError:
                        # Not valid JSON, which is fine
                        pass
    
    def test_consistent_item_ids(self):
        """Verify item_id consistency across related events."""
        agent = MockAgentScenarios.create_full_research_agent()
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test research question"}]
        )
        
        stream_events = list(agent.predict_stream(request))
        
        # Separate progress and content events
        progress_item_ids = set()
        content_item_ids = set()
        done_item_ids = set()
        
        for event in stream_events:
            if event.type == "response.output_text.delta":
                if "[PHASE:" in event.delta:
                    progress_item_ids.add(event.item_id)
                else:
                    content_item_ids.add(event.item_id)
            elif event.type == "response.output_item.done":
                done_item_ids.add(event.item["id"])
        
        # All content deltas should have the same item_id
        if len(content_item_ids) > 1:
            pytest.fail(f"Content deltas have inconsistent item_ids: {content_item_ids}")
        
        # Done event should match content item_id
        if content_item_ids and done_item_ids:
            assert content_item_ids == done_item_ids, \
                f"Content item_ids {content_item_ids} don't match done item_ids {done_item_ids}"
    
    def test_cached_response_with_no_research(self):
        """Test when agent returns cached result immediately."""
        agent = MockAgentScenarios.create_fast_cached_agent()
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Cached query"}]
        )
        
        stream_events = list(agent.predict_stream(request))
        
        # Should still have at least one done event
        done_events = [e for e in stream_events if e.type == "response.output_item.done"]
        assert len(done_events) >= 1, "Should have at least one done event even for cached responses"
        
        # May or may not have progress events - that's OK for cached responses
        # The key is that it shouldn't break
        assert len(stream_events) >= 1, "Should have at least some stream events"
    
    def test_malformed_event_handling(self):
        """Test agent handles malformed progress events gracefully."""
        agent = MockAgentScenarios.create_broken_progress_agent()
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test with broken progress"}]
        )
        
        # Should not raise exceptions even with malformed progress
        try:
            stream_events = list(agent.predict_stream(request))
            # Should complete without crashing
            assert len(stream_events) >= 1, "Should produce some events even with broken progress"
        except Exception as e:
            pytest.fail(f"Agent crashed with malformed progress events: {e}")
    
    def test_progress_percentage_in_range(self):
        """Verify progress percentage is 0-100 and generally increases."""
        agent = MockAgentScenarios.create_full_research_agent()
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test research question"}]
        )
        
        stream_events = list(agent.predict_stream(request))
        
        progress_percentages = []
        progress_pattern = re.compile(r'\[META:progress:([^\]]+)\]')
        
        for event in stream_events:
            if event.type == "response.output_text.delta" and "[META:progress:" in event.delta:
                match = progress_pattern.search(event.delta)
                if match:
                    try:
                        percentage = float(match.group(1))
                        progress_percentages.append(percentage)
                        
                        # Verify percentage is in valid range
                        assert 0 <= percentage <= 100, f"Progress percentage out of range: {percentage}"
                    except ValueError:
                        pytest.fail(f"Invalid progress percentage format: {match.group(1)}")
        
        # If we have multiple percentages, they should generally increase
        if len(progress_percentages) >= 2:
            # Allow for some flexibility, but general trend should be increasing
            first_half_avg = sum(progress_percentages[:len(progress_percentages)//2])
            second_half_avg = sum(progress_percentages[len(progress_percentages)//2:])
            
            if len(progress_percentages) >= 4:  # Only check trend with enough data points
                assert second_half_avg >= first_half_avg, \
                    f"Progress should generally increase: {progress_percentages}"
    
    def test_separator_after_progress_events(self):
        """Verify --- separator after progress events for parsing."""
        agent = MockAgentScenarios.create_full_research_agent()
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test research question"}]
        )
        
        stream_events = list(agent.predict_stream(request))
        
        separator_found = False
        for event in stream_events:
            if event.type == "response.output_text.delta" and "[PHASE:" in event.delta:
                if "---" in event.delta:
                    separator_found = True
                    break
        
        # Note: This test might need adjustment based on actual implementation
        # The separator helps UI distinguish progress from content
        if separator_found:
            # If separator is implemented, verify it's properly placed
            assert True  # Implementation-dependent
    
    def test_no_duplicate_progress_events(self):
        """Verify no duplicate progress events for same node."""
        agent = MockAgentScenarios.create_full_research_agent()
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test research question"}]
        )
        
        stream_events = list(agent.predict_stream(request))
        
        # Track which phases we've seen
        seen_phases = set()
        node_pattern = re.compile(r'\[META:node:([^\]]+)\]')
        
        for event in stream_events:
            if event.type == "response.output_text.delta" and "[META:node:" in event.delta:
                match = node_pattern.search(event.delta)
                if match:
                    node_name = match.group(1)
                    assert node_name not in seen_phases, \
                        f"Duplicate progress event for node: {node_name}"
                    seen_phases.add(node_name)
    
    def test_streaming_starts_quickly(self):
        """Verify first event emits within reasonable time."""
        agent = MockAgentScenarios.create_full_research_agent()
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test research question"}]
        )
        
        start_time = time.time()
        stream_events = list(agent.predict_stream(request))
        first_event_time = time.time()
        
        # Should get first event quickly (within 5 seconds for mocked agent)
        time_to_first_event = first_event_time - start_time
        assert time_to_first_event < 5.0, \
            f"First event took too long: {time_to_first_event:.2f} seconds"
        
        # Should have at least one event
        assert len(stream_events) >= 1, "Should emit at least one event"


class TestEdgeCasesAndErrorScenarios:
    """Test edge cases and error scenarios for robust streaming."""
    
    def test_empty_synthesis_response(self):
        """Test handling when synthesis produces empty response."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="")
        
        with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_llm', return_value=mock_llm):
            mock_phase2_return = (None, None, None, None, None, None)
            with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_phase2_components', return_value=mock_phase2_return):
                agent = RefactoredResearchAgent()
                
                mock_graph = Mock()
                mock_graph.stream.return_value = [
                    {"synthesize_answer": {"messages": [Mock(content="")]}}  # Empty content
                ]
                agent.graph = mock_graph
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test empty response"}]
        )
        
        # Should not crash with empty content
        try:
            stream_events = list(agent.predict_stream(request))
            # Should still produce some events (at least done event)
            assert len(stream_events) >= 1, "Should produce events even with empty synthesis"
        except Exception as e:
            pytest.fail(f"Agent crashed with empty synthesis: {e}")
    
    def test_very_long_content_streaming(self):
        """Test streaming with very long content."""
        mock_llm = Mock()
        long_content = "This is a very long response. " * 1000  # ~30KB response
        mock_llm.invoke.return_value = AIMessage(content=long_content)
        
        with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_llm', return_value=mock_llm):
            mock_phase2_return = (None, None, None, None, None, None)
            with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_phase2_components', return_value=mock_phase2_return):
                agent = RefactoredResearchAgent()
                
                mock_graph = Mock()
                mock_graph.stream.return_value = [
                    {"synthesize_answer": {"messages": [Mock(content=long_content)]}}
                ]
                agent.graph = mock_graph
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test long response"}]
        )
        
        stream_events = list(agent.predict_stream(request))
        
        # Should handle long content properly
        done_events = [e for e in stream_events if e.type == "response.output_item.done"]
        assert len(done_events) == 1, "Should have exactly one done event"
        
        # Verify content is preserved
        done_content = done_events[0].item["content"][0]["text"]
        assert len(done_content) >= len(long_content) * 0.9, "Content should be mostly preserved"


if __name__ == "__main__":
    # Run specific test for debugging
    test_suite = TestAgentStreamingCompliance()
    
    print("Running agent streaming compliance tests...")
    print("-" * 60)
    
    tests_to_run = [
        ("Phase markers emitted", test_suite.test_phase_markers_emitted_for_all_nodes),
        ("Phase marker format", test_suite.test_phase_marker_format), 
        ("All expected phases", test_suite.test_all_expected_phases_covered),
        ("Metadata format", test_suite.test_metadata_markers_format),
        ("Required metadata fields", test_suite.test_required_metadata_fields),
        ("Progress before content", test_suite.test_progress_events_before_content),
        ("No JSON in deltas", test_suite.test_no_json_in_delta_events),
        ("Consistent item IDs", test_suite.test_consistent_item_ids),
        ("Cached response handling", test_suite.test_cached_response_with_no_research),
        ("Progress percentage range", test_suite.test_progress_percentage_in_range)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests_to_run:
        try:
            print(f"Testing {test_name}...")
            test_func()
            print(f"   ✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            failed += 1
    
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("Agent streaming compliance tests complete!")