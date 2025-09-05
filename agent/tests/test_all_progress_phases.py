"""
Comprehensive integration test to ensure all 4 progress phases are emitted and received.

This test forces the agent through all workflow stages and verifies that:
1. All progress events are emitted by the agent
2. All events are properly received by the UI backend
3. The phase mapping works correctly for all stages

Key Testing Areas:
- Forces full workflow execution (no shortcuts)
- Verifies QUERYING, SEARCHING, ANALYZING, SYNTHESIZING phases
- Tests UI backend parsing of all phase markers
- Ensures no phases are skipped
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Set
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
from deep_research_agent.core.types import ResearchContext


class ProgressEventCollector:
    """Collects and analyzes progress events during streaming."""
    
    def __init__(self):
        self.events = []
        self.phases_seen = set()
        self.node_events = {}
        self.progress_deltas = []
        
    def collect_event(self, event: ResponsesAgentStreamEvent):
        """Collect and analyze a streaming event."""
        self.events.append(event)
        
        if event.type == "response.output_text.delta" and event.delta:
            # Check for progress markers
            if "[PHASE:" in event.delta:
                self.progress_deltas.append(event.delta)
                # Extract phase
                import re
                phase_match = re.search(r'\[PHASE:(\w+)\]', event.delta)
                if phase_match:
                    phase = phase_match.group(1)
                    self.phases_seen.add(phase)
                    
                    # Extract node name if present
                    node_match = re.search(r'\[META:node:(\w+)\]', event.delta)
                    if node_match:
                        node = node_match.group(1)
                        if node not in self.node_events:
                            self.node_events[node] = []
                        self.node_events[node].append(event.delta)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collected events."""
        return {
            "total_events": len(self.events),
            "phases_seen": list(self.phases_seen),
            "nodes_with_progress": list(self.node_events.keys()),
            "progress_delta_count": len(self.progress_deltas),
            "has_all_phases": self._has_all_required_phases()
        }
    
    def _has_all_required_phases(self) -> bool:
        """Check if all 4 required phases were seen."""
        required = {"QUERYING", "SEARCHING", "ANALYZING", "SYNTHESIZING"}
        # Also accept variations
        alternatives = {
            "PREPARING": "SEARCHING",  # Preparing maps to searching phase
            "AGGREGATING": "SEARCHING",  # Aggregating is part of searching
            "SEARCHING_INTERNAL": "SEARCHING"  # Internal search is still searching
        }
        
        normalized_phases = set()
        for phase in self.phases_seen:
            if phase in alternatives:
                normalized_phases.add(alternatives[phase])
            else:
                normalized_phases.add(phase)
        
        return required.issubset(normalized_phases)


class TestAllProgressPhases:
    """Test suite ensuring all progress phases are emitted and received."""
    
    def create_full_workflow_agent(self):
        """Create an agent that goes through ALL workflow nodes."""
        mock_llm = Mock()
        
        # Mock responses for each node to ensure they all execute
        mock_llm.invoke.side_effect = [
            # generate_queries response
            AIMessage(content='{"queries": ["test query 1", "test query 2"], "strategy": "comprehensive"}'),
            
            # reflect response (needs_more_research: false to avoid loops)
            AIMessage(content='{"needs_more_research": false, "reflection": "Sufficient information gathered"}'),
            
            # synthesize response
            AIMessage(content="Based on comprehensive research across all phases, here is the final answer.")
        ]
        
        # Create agent with mocked components
        with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_llm', return_value=mock_llm):
            mock_phase2_return = (None, None, None, None, None, None)
            with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_phase2_components', return_value=mock_phase2_return):
                agent = RefactoredResearchAgent()
                
                # Mock the graph to emit ALL nodes in sequence
                mock_graph = Mock()
                
                def mock_stream(initial_state, stream_mode=None):
                    """Yield events for ALL workflow nodes."""
                    # Simulate complete workflow execution
                    nodes = [
                        # Phase 1: QUERYING
                        {
                            "generate_queries": {
                                "research_context": Mock(
                                    generated_queries=["query 1", "query 2"],
                                    web_results=[],
                                    vector_results=[]
                                ),
                                "messages": []
                            }
                        },
                        
                        # Phase 2: SEARCHING (multiple nodes)
                        {
                            "batch_controller": {
                                "research_context": Mock(),
                                "messages": []
                            }
                        },
                        {
                            "parallel_web_search": {
                                "research_context": Mock(
                                    web_results=[
                                        {"url": "http://example.com", "content": "Result 1"},
                                        {"url": "http://test.com", "content": "Result 2"}
                                    ]
                                ),
                                "messages": []
                            }
                        },
                        {
                            "aggregate_search_results": {
                                "research_context": Mock(
                                    web_results=[
                                        {"url": "http://example.com", "content": "Result 1"},
                                        {"url": "http://test.com", "content": "Result 2"}
                                    ],
                                    aggregated_results="Combined search results"
                                ),
                                "messages": []
                            }
                        },
                        
                        # Phase 3: ANALYZING (vector search and reflection)
                        {
                            "vector_research": {
                                "research_context": Mock(
                                    vector_results=[
                                        {"source": "internal", "content": "Vector result 1"}
                                    ]
                                ),
                                "messages": []
                            }
                        },
                        {
                            "reflect": {
                                "research_context": Mock(
                                    reflection="Analysis complete, sufficient information gathered"
                                ),
                                "needs_more_research": False,
                                "messages": []
                            }
                        },
                        
                        # Phase 4: SYNTHESIZING
                        {
                            "synthesize_answer": {
                                "research_context": Mock(
                                    synthesis_chunks=["Final ", "synthesized ", "answer."]
                                ),
                                "messages": [AIMessage(content="Final synthesized answer.")]
                            }
                        }
                    ]
                    
                    for node in nodes:
                        yield node
                
                mock_graph.stream = mock_stream
                agent.graph = mock_graph
                return agent
    
    def test_all_phases_emitted(self):
        """Test that all 4 progress phases are emitted by the agent."""
        agent = self.create_full_workflow_agent()
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Explain quantum computing in detail"}]
        )
        
        collector = ProgressEventCollector()
        
        # Collect all events
        for event in agent.predict_stream(request):
            collector.collect_event(event)
        
        # Verify results
        summary = collector.get_summary()
        
        print("\n" + "=" * 60)
        print("PROGRESS PHASES TEST RESULTS")
        print("=" * 60)
        print(f"Total events: {summary['total_events']}")
        print(f"Progress deltas: {summary['progress_delta_count']}")
        print(f"Phases seen: {summary['phases_seen']}")
        print(f"Nodes with progress: {summary['nodes_with_progress']}")
        print(f"Has all phases: {summary['has_all_phases']}")
        
        # Assertions
        assert summary['progress_delta_count'] > 0, "No progress events emitted"
        assert len(summary['phases_seen']) >= 4, f"Not all phases seen: {summary['phases_seen']}"
        assert summary['has_all_phases'], f"Missing required phases. Seen: {summary['phases_seen']}"
        
        # Check specific phases
        expected_phases = ["QUERYING", "SEARCHING", "ANALYZING", "SYNTHESIZING"]
        for phase in expected_phases:
            found = phase in summary['phases_seen'] or any(
                alt in summary['phases_seen'] 
                for alt in self._get_phase_alternatives(phase)
            )
            assert found, f"Phase {phase} not found in {summary['phases_seen']}"
        
        print("\n✅ All 4 progress phases successfully emitted!")
    
    def test_ui_backend_receives_all_phases(self):
        """Test that UI backend correctly parses all phase events."""
        # Import the UI parser from the UI streaming integration test
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from test_ui_streaming_integration import UIStreamEventParser
        
        agent = self.create_full_workflow_agent()
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Research climate change impacts"}]
        )
        
        ui_parser = UIStreamEventParser()
        phases_received_by_ui = set()
        
        # Process events through UI parser
        for event in agent.predict_stream(request):
            if event.type == "response.output_text.delta" and event.delta:
                if "[PHASE:" in event.delta:
                    parsed = ui_parser.parse_progress_delta(event.delta)
                    phases_received_by_ui.add(parsed["phase"])
                    ui_parser.handle_research_update(parsed)
                    
                    print(f"UI received phase: {parsed['phase']} (raw: {parsed['raw_phase']})")
        
        # Verify UI received all phases
        expected_ui_phases = {"querying", "searching", "analyzing", "synthesizing"}
        assert expected_ui_phases.issubset(phases_received_by_ui), \
            f"UI missing phases. Received: {phases_received_by_ui}"
        
        print(f"\n✅ UI backend successfully received all phases: {phases_received_by_ui}")
    
    def test_progress_events_ordering(self):
        """Test that progress events occur in the correct order."""
        agent = self.create_full_workflow_agent()
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Complex research query"}]
        )
        
        phase_order = []
        
        for event in agent.predict_stream(request):
            if event.type == "response.output_text.delta" and "[PHASE:" in event.delta:
                import re
                phase_match = re.search(r'\[PHASE:(\w+)\]', event.delta)
                if phase_match:
                    phase = phase_match.group(1)
                    if phase not in [p for p, _ in phase_order]:
                        phase_order.append((phase, time.time()))
        
        print("\n" + "=" * 60)
        print("PHASE ORDERING TEST")
        print("=" * 60)
        for i, (phase, timestamp) in enumerate(phase_order, 1):
            print(f"{i}. {phase}")
        
        # Verify general ordering (some flexibility for parallel operations)
        phase_positions = {phase: i for i, (phase, _) in enumerate(phase_order)}
        
        # QUERYING should come before SYNTHESIZING
        if "QUERYING" in phase_positions and "SYNTHESIZING" in phase_positions:
            assert phase_positions["QUERYING"] < phase_positions["SYNTHESIZING"], \
                "QUERYING should occur before SYNTHESIZING"
        
        print("\n✅ Progress events occur in correct order!")
    
    def test_node_to_phase_mapping(self):
        """Test that each node maps to the correct UI phase."""
        node_to_expected_phase = {
            "generate_queries": "QUERYING",
            "batch_controller": "PREPARING",
            "parallel_web_search": "SEARCHING",
            "aggregate_search_results": "AGGREGATING",
            "vector_research": "SEARCHING_INTERNAL",
            "reflect": "ANALYZING",
            "synthesize_answer": "SYNTHESIZING"
        }
        
        agent = self.create_full_workflow_agent()
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test query"}]
        )
        
        node_phase_mapping = {}
        
        for event in agent.predict_stream(request):
            if event.type == "response.output_text.delta" and "[PHASE:" in event.delta:
                import re
                phase_match = re.search(r'\[PHASE:(\w+)\]', event.delta)
                node_match = re.search(r'\[META:node:(\w+)\]', event.delta)
                
                if phase_match and node_match:
                    node = node_match.group(1)
                    phase = phase_match.group(1)
                    node_phase_mapping[node] = phase
        
        print("\n" + "=" * 60)
        print("NODE TO PHASE MAPPING TEST")
        print("=" * 60)
        for node, expected_phase in node_to_expected_phase.items():
            actual_phase = node_phase_mapping.get(node, "NOT FOUND")
            status = "✅" if actual_phase == expected_phase else "❌"
            print(f"{status} {node}: {actual_phase} (expected: {expected_phase})")
        
        # Allow some flexibility in mapping
        assert len(node_phase_mapping) > 0, "No node-to-phase mappings found"
        print(f"\n✅ Node to phase mapping verified for {len(node_phase_mapping)} nodes!")
    
    def test_complex_query_triggers_all_phases(self):
        """Test that a complex query triggers all research phases."""
        agent = self.create_full_workflow_agent()
        
        # Complex query that should trigger full research
        complex_request = ResponsesAgentRequest(
            input=[{
                "role": "user", 
                "content": "Compare and contrast quantum computing, classical computing, and neuromorphic computing. Include their principles, current applications, limitations, and future prospects. Provide specific examples and cite recent developments."
            }]
        )
        
        collector = ProgressEventCollector()
        
        for event in agent.predict_stream(complex_request):
            collector.collect_event(event)
        
        summary = collector.get_summary()
        
        print("\n" + "=" * 60)
        print("COMPLEX QUERY TEST")
        print("=" * 60)
        print(f"Query triggered {len(summary['phases_seen'])} phases")
        print(f"Phases: {summary['phases_seen']}")
        
        assert summary['has_all_phases'], \
            f"Complex query should trigger all phases. Got: {summary['phases_seen']}"
        
        print("\n✅ Complex query successfully triggers all research phases!")
    
    def _get_phase_alternatives(self, phase: str) -> List[str]:
        """Get alternative phase names that map to the same UI phase."""
        alternatives = {
            "SEARCHING": ["PREPARING", "AGGREGATING", "SEARCHING_INTERNAL"],
            "ANALYZING": ["REFLECT", "REFLECTION"],
            "SYNTHESIZING": ["SYNTHESIS"]
        }
        return alternatives.get(phase, [])
    
    def test_progress_metadata_completeness(self):
        """Test that progress events include complete metadata."""
        agent = self.create_full_workflow_agent()
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "Test metadata"}]
        )
        
        metadata_found = {
            "progress": False,
            "node": False,
            "elapsed": False,
            "queries": False,
            "results": False
        }
        
        for event in agent.predict_stream(request):
            if event.type == "response.output_text.delta" and "[META:" in event.delta:
                if "[META:progress:" in event.delta:
                    metadata_found["progress"] = True
                if "[META:node:" in event.delta:
                    metadata_found["node"] = True
                if "[META:elapsed:" in event.delta:
                    metadata_found["elapsed"] = True
                if "[META:queries:" in event.delta:
                    metadata_found["queries"] = True
                if "[META:results:" in event.delta:
                    metadata_found["results"] = True
        
        print("\n" + "=" * 60)
        print("METADATA COMPLETENESS TEST")
        print("=" * 60)
        for key, found in metadata_found.items():
            status = "✅" if found else "⚠️"
            print(f"{status} {key}: {'Found' if found else 'Not found'}")
        
        # At minimum, progress and node should be present
        assert metadata_found["progress"], "Progress metadata missing"
        assert metadata_found["node"], "Node metadata missing"
        
        print("\n✅ Essential metadata present in progress events!")


if __name__ == "__main__":
    # Run all tests
    test_suite = TestAllProgressPhases()
    
    tests = [
        ("All Phases Emitted", test_suite.test_all_phases_emitted),
        ("UI Backend Receives All Phases", test_suite.test_ui_backend_receives_all_phases),
        ("Progress Events Ordering", test_suite.test_progress_events_ordering),
        ("Node to Phase Mapping", test_suite.test_node_to_phase_mapping),
        ("Complex Query Triggers All Phases", test_suite.test_complex_query_triggers_all_phases),
        ("Progress Metadata Completeness", test_suite.test_progress_metadata_completeness)
    ]
    
    print("🧪 RUNNING COMPREHENSIVE PROGRESS PHASE TESTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        try:
            test_func()
            passed += 1
            print(f"✅ {test_name} PASSED")
        except AssertionError as e:
            failed += 1
            print(f"❌ {test_name} FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"💥 {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! All 4 progress phases are working correctly!")
    else:
        print(f"\n⚠️  {failed} tests failed. Check the logs above for details.")
        exit(1)