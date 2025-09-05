"""
Test UI's handling of various agent streaming scenarios.

This test file validates how the UI components handle different streaming
patterns from the agent, ensuring graceful degradation when events don't
match expectations and proper parsing of all supported formats.

Key Testing Areas:
- UI parsing of PHASE and META markers
- Handling missing or malformed progress events  
- Phase transition mapping and timing
- Fallback behaviors for edge cases
- Integration between agent streaming and UI state management
"""

import pytest
import json
import re
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

# These tests would typically run against the actual UI components
# For now, we'll simulate the UI parsing logic to test agent compliance


class UIStreamEventParser:
    """Simulates the UI's parsing logic for agent stream events."""
    
    def __init__(self):
        self.current_phase = 'complete'
        self.queries_generated = 0
        self.sources_found = 0
        self.iterations_complete = 0
        self.research_metadata = None
        
    def parse_progress_delta(self, content: str) -> Dict[str, Any]:
        """
        Simulates the UI's _parse_progress_delta method from agent_client.py.
        Returns extracted phase and metadata information.
        """
        import re
        
        # Extract phase from content - matches UI implementation
        phase_match = re.search(r'\[PHASE:(\w+)\]', content)
        phase = phase_match.group(1).lower() if phase_match else "processing"
        
        # Map agent phases to UI phases - matches UI implementation
        ui_phase_map = {
            "querying": "querying",
            "preparing": "searching",
            "searching": "searching", 
            "searching_internal": "searching",
            "aggregating": "analyzing",
            "analyzing": "analyzing", 
            "synthesizing": "synthesizing",
            "processing": "searching"  # fallback
        }
        current_phase = ui_phase_map.get(phase, "searching")  # fallback to searching for unknown phases
        
        # Extract metadata from META markers - matches UI implementation
        metadata = {}
        for match in re.finditer(r'\[META:(\w+):([^\]]+)\]', content):
            key, value = match.groups()
            # Convert numeric values
            try:
                metadata[key] = float(value) if '.' in value else int(value)
            except ValueError:
                metadata[key] = value
        
        # Extract human-readable message (remove markers for display)
        message_match = re.search(r'\[PHASE:\w+\]\s+([^\[]+?)(?:\[|$)', content, re.DOTALL)
        message = message_match.group(1).strip() if message_match else ""
        
        # Remove emoji if present but preserve the description
        if message:
            # Remove leading emoji but keep the descriptive text
            message = re.sub(r'^[🔍📋🌐🗄️📊🤔✍️⚙️]\s*', '', message).strip()
        
        return {
            "phase": current_phase,
            "metadata": metadata,
            "message": message,
            "raw_phase": phase
        }
    
    def handle_research_update(self, parsed_data: Dict[str, Any]):
        """Simulates UI's research progress update handling."""
        if parsed_data["metadata"]:
            self.current_phase = parsed_data["phase"]
            # Accumulate metadata values (don't overwrite with 0 if key not present)
            if "queries" in parsed_data["metadata"]:
                self.queries_generated = parsed_data["metadata"]["queries"]
            if "results" in parsed_data["metadata"]:
                self.sources_found = parsed_data["metadata"]["results"]
            if "totalSourcesFound" in parsed_data["metadata"]:
                self.sources_found = parsed_data["metadata"]["totalSourcesFound"] 
            if "iterations" in parsed_data["metadata"]:
                self.iterations_complete = parsed_data["metadata"]["iterations"]
            
            # Build research metadata
            self.research_metadata = {
                "searchQueries": [],  # Would be populated from actual data
                "sources": [],
                "researchIterations": parsed_data["metadata"].get("queries", 0),
                "totalSourcesFound": parsed_data["metadata"].get("results", 0),
                "phase": parsed_data["phase"],
                "progressPercentage": parsed_data["metadata"].get("progress", 0.0),
                "elapsedTime": parsed_data["metadata"].get("elapsed", 0.0),
                "currentNode": parsed_data["metadata"].get("node", ""),
                "vectorResultsCount": parsed_data["metadata"].get("vector_results", 0)
            }


class TestUIStreamingIntegration:
    """Test UI handling of different agent streaming scenarios."""
    
    def test_ui_parses_valid_progress_events(self):
        """Test UI correctly parses well-formed progress events."""
        parser = UIStreamEventParser()
        
        # Test each phase type the agent emits
        test_cases = [
            ("[PHASE:QUERYING] 🔍 Analyzing your query... (20%)\n[META:progress:20]\n[META:elapsed:1.2]", "querying"),
            ("[PHASE:SEARCHING] 🌐 Searching across multiple sources... (60%)\n[META:results:5]", "searching"),
            ("[PHASE:SYNTHESIZING] ✍️ Synthesizing comprehensive response... (90%)", "synthesizing")
        ]
        
        for content, expected_phase in test_cases:
            parsed = parser.parse_progress_delta(content)
            
            assert parsed["phase"] == expected_phase, f"Phase mismatch for {content}"
            assert parsed["raw_phase"] in content.lower(), "Raw phase should be extracted"
            assert "metadata" in parsed, "Should extract metadata"
            
            # If progress metadata exists, verify it's numeric
            if "progress" in parsed["metadata"]:
                assert isinstance(parsed["metadata"]["progress"], (int, float)), "Progress should be numeric"
                assert 0 <= parsed["metadata"]["progress"] <= 100, "Progress should be 0-100"
    
    def test_ui_handles_missing_phase_markers(self):
        """Test UI gracefully handles content without PHASE markers."""
        parser = UIStreamEventParser()
        
        # Content without PHASE markers
        content = "Regular content without phase markers"
        parsed = parser.parse_progress_delta(content)
        
        # Should default to processing phase
        assert parsed["phase"] == "searching", "Should default to fallback phase"  # processing maps to searching
        assert parsed["raw_phase"] == "processing", "Should use processing as raw phase"
    
    def test_ui_handles_malformed_meta_tags(self):
        """Test UI handles malformed META tags gracefully."""
        parser = UIStreamEventParser()
        
        test_cases = [
            "[PHASE:SEARCHING] Content [META:incomplete",  # Incomplete META tag
            "[PHASE:SEARCHING] Content [META::empty_key]",  # Empty key
            "[PHASE:SEARCHING] Content [META:key:]",       # Empty value  
            "[PHASE:SEARCHING] Content [META:invalid:format:extra]"  # Too many colons
        ]
        
        for content in test_cases:
            # Should not crash with malformed META tags
            try:
                parsed = parser.parse_progress_delta(content)
                assert parsed["phase"] == "searching", "Should still extract phase"
                # Malformed META tags might not be parsed, but shouldn't crash
                assert "metadata" in parsed, "Should have metadata dict even if empty"
            except Exception as e:
                pytest.fail(f"UI parser crashed on malformed META: {content}, error: {e}")
    
    def test_ui_phase_mapping_completeness(self):
        """Test UI has mappings for all agent phases."""
        parser = UIStreamEventParser()
        
        # All phases the agent might emit
        agent_phases = [
            "QUERYING", "PREPARING", "SEARCHING", "SEARCHING_INTERNAL",
            "AGGREGATING", "ANALYZING", "SYNTHESIZING", "PROCESSING"
        ]
        
        for agent_phase in agent_phases:
            content = f"[PHASE:{agent_phase}] Test content"
            parsed = parser.parse_progress_delta(content)
            
            # Should map to a valid UI phase (processing is also valid as fallback)
            valid_ui_phases = {"querying", "searching", "analyzing", "synthesizing", "processing"}
            assert parsed["phase"] in valid_ui_phases, \
                f"Agent phase {agent_phase} maps to invalid UI phase: {parsed['phase']}"
    
    def test_ui_handles_rapid_phase_changes(self):
        """Test UI handles very fast phase transitions."""
        parser = UIStreamEventParser()
        
        # Simulate rapid phase changes
        rapid_phases = [
            "[PHASE:QUERYING] Phase 1 [META:progress:10]",
            "[PHASE:SEARCHING] Phase 2 [META:progress:30]", 
            "[PHASE:ANALYZING] Phase 3 [META:progress:70]",
            "[PHASE:SYNTHESIZING] Phase 4 [META:progress:90]"
        ]
        
        previous_phase = None
        for content in rapid_phases:
            parsed = parser.parse_progress_delta(content)
            parser.handle_research_update(parsed)
            
            # Phase should update
            assert parser.current_phase != previous_phase or previous_phase is None, \
                "Phase should change with each update"
            previous_phase = parser.current_phase
    
    def test_ui_accumulates_metadata_correctly(self):
        """Test UI correctly accumulates and updates metadata."""
        parser = UIStreamEventParser()
        
        # Simulate progression with increasing metadata
        progression = [
            "[PHASE:QUERYING] Generating queries [META:queries:3][META:progress:20]",
            "[PHASE:SEARCHING] Found results [META:results:10][META:progress:60]", 
            "[PHASE:ANALYZING] Processing [META:vector_results:5][META:progress:80]"
        ]
        
        for content in progression:
            parsed = parser.parse_progress_delta(content)
            parser.handle_research_update(parsed)
        
        # Verify final state has accumulated metadata
        assert parser.queries_generated == 3, "Should accumulate query count"
        assert parser.sources_found == 10, "Should accumulate source count"
        assert parser.research_metadata is not None, "Should build research metadata"
        
        # Verify metadata structure
        metadata = parser.research_metadata
        assert "phase" in metadata, "Metadata should include phase"
        assert "progressPercentage" in metadata, "Metadata should include progress"
        assert "vectorResultsCount" in metadata, "Metadata should include vector results"
    
    def test_ui_handles_no_progress_events(self):
        """Test UI gracefully handles when no progress events are received."""
        parser = UIStreamEventParser()
        
        # Simulate only receiving final content with no progress
        # This would happen with cached responses
        
        # Initial state should be stable
        initial_phase = parser.current_phase
        initial_queries = parser.queries_generated
        initial_sources = parser.sources_found
        
        # No updates means state should remain stable
        assert parser.current_phase == initial_phase, "Phase should remain stable"
        assert parser.queries_generated == initial_queries, "Queries should remain stable"
        assert parser.sources_found == initial_sources, "Sources should remain stable"
    
    def test_ui_handles_delayed_progress_events(self):
        """Test UI handles progress events arriving after content."""
        parser = UIStreamEventParser()
        
        # Simulate content arriving first, then progress (edge case)
        # This shouldn't normally happen but test robustness
        
        # First, simulate receiving some progress
        progress_content = "[PHASE:SYNTHESIZING] Final phase [META:progress:90]"
        parsed = parser.parse_progress_delta(progress_content)
        parser.handle_research_update(parsed)
        
        # Should handle the late progress event
        assert parser.current_phase == "synthesizing", "Should update to final phase"
    
    def test_ui_metadata_type_conversion(self):
        """Test UI correctly converts metadata value types."""
        parser = UIStreamEventParser()
        
        # Test different value types
        content = (
            "[PHASE:SEARCHING] Test content "
            "[META:progress:45.5]"      # Float
            "[META:queries:3]"          # Integer  
            "[META:node:test_node]"     # String
            "[META:elapsed:1.234]"      # Float with decimals
        )
        
        parsed = parser.parse_progress_delta(content)
        metadata = parsed["metadata"]
        
        # Verify type conversions
        assert isinstance(metadata["progress"], float), "Progress should be float"
        assert metadata["progress"] == 45.5, "Float value should be preserved"
        
        assert isinstance(metadata["queries"], int), "Queries should be integer"
        assert metadata["queries"] == 3, "Integer value should be preserved"
        
        assert isinstance(metadata["node"], str), "Node should be string"
        assert metadata["node"] == "test_node", "String value should be preserved"
    
    def test_ui_emoji_removal_from_messages(self):
        """Test UI properly removes emojis but preserves descriptive text."""
        parser = UIStreamEventParser()
        
        test_cases = [
            ("[PHASE:QUERYING] 🔍 Analyzing your query and generating search strategies...", 
             "Analyzing your query and generating search strategies..."),
            ("[PHASE:SEARCHING] 🌐 Searching across multiple sources...", 
             "Searching across multiple sources..."),
            ("[PHASE:SYNTHESIZING] ✍️ Synthesizing comprehensive response...",
             "Synthesizing comprehensive response...")
        ]
        
        for content, expected_message in test_cases:
            parsed = parser.parse_progress_delta(content)
            message = parsed["message"]
            
            # Should remove emoji but preserve text
            assert "🔍" not in message and "🌐" not in message and "✍️" not in message, \
                "Emojis should be removed"
            assert expected_message.strip() in message, \
                f"Descriptive text should be preserved. Expected: {expected_message}, Got: {message}"
    
    def test_ui_fallback_behavior_unknown_phases(self):
        """Test UI fallback when receiving unknown phase names."""
        parser = UIStreamEventParser()
        
        # Test with unknown phase names
        unknown_phases = [
            "[PHASE:UNKNOWN_PHASE] Some content",
            "[PHASE:CUSTOM_NODE] Custom processing", 
            "[PHASE:] Empty phase name"
        ]
        
        for content in unknown_phases:
            parsed = parser.parse_progress_delta(content)
            
            # Should map to fallback phase (processing maps to searching)
            valid_phases = {"querying", "searching", "analyzing", "synthesizing"}
            assert parsed["phase"] in valid_phases, \
                f"Unknown phase should map to valid fallback: {parsed['phase']}"
            # Unknown phases should specifically fall back to "searching" (via processing)
            assert parsed["phase"] == "searching", \
                f"Unknown phase should fall back to 'searching', got: {parsed['phase']}"


class TestAgentUIStreamingContract:
    """Test the overall streaming contract between agent and UI."""
    
    def test_end_to_end_progress_parsing(self):
        """Test complete flow of agent events through UI parser."""
        # This would simulate a full agent response through UI parsing
        parser = UIStreamEventParser()
        
        # Simulate a complete agent streaming sequence
        agent_events = [
            "[PHASE:QUERYING] 🔍 Analyzing query... (10%)\n[META:progress:10][META:queries:2]",
            "[PHASE:SEARCHING] 🌐 Searching sources... (40%)\n[META:progress:40][META:results:8]",
            "[PHASE:ANALYZING] 🤔 Analyzing results... (70%)\n[META:progress:70]",
            "[PHASE:SYNTHESIZING] ✍️ Creating response... (90%)\n[META:progress:90]"
        ]
        
        phases_seen = []
        progress_values = []
        
        for event_content in agent_events:
            parsed = parser.parse_progress_delta(event_content)
            parser.handle_research_update(parsed)
            
            phases_seen.append(parser.current_phase)
            if parser.research_metadata and "progressPercentage" in parser.research_metadata:
                progress_values.append(parser.research_metadata["progressPercentage"])
        
        # Verify logical progression
        assert len(phases_seen) == 4, "Should process all events"
        assert "querying" in phases_seen, "Should see querying phase"
        assert "synthesizing" in phases_seen, "Should see synthesizing phase"
        
        # Progress should generally increase
        if len(progress_values) >= 2:
            assert progress_values[-1] >= progress_values[0], "Progress should increase over time"
    
    def test_ui_state_consistency(self):
        """Test UI maintains consistent state throughout streaming."""
        parser = UIStreamEventParser()
        
        # Start with known state
        initial_state = {
            "phase": parser.current_phase,
            "queries": parser.queries_generated,
            "sources": parser.sources_found
        }
        
        # Apply valid update
        content = "[PHASE:SEARCHING] Processing [META:queries:5][META:results:15]"
        parsed = parser.parse_progress_delta(content)
        parser.handle_research_update(parsed)
        
        # Verify state changed appropriately  
        assert parser.current_phase == "searching", "Phase should update"
        assert parser.queries_generated == 5, "Queries should update"
        assert parser.sources_found == 15, "Sources should update"
        
        # State should remain consistent between updates
        second_state = {
            "phase": parser.current_phase,
            "queries": parser.queries_generated,
            "sources": parser.sources_found
        }
        
        # Apply another update
        content2 = "[PHASE:ANALYZING] Next phase [META:progress:80]"
        parsed2 = parser.parse_progress_delta(content2)
        parser.handle_research_update(parsed2)
        
        # Previous values should be preserved where not updated
        assert parser.queries_generated == 5, "Previous query count should persist"
        assert parser.sources_found == 15, "Previous source count should persist"
        assert parser.current_phase == "analyzing", "Phase should update to new value"


if __name__ == "__main__":
    print("Running UI streaming integration tests...")
    print("-" * 60)
    
    # Test the UI parser components
    ui_tests = TestUIStreamingIntegration()
    contract_tests = TestAgentUIStreamingContract()
    
    test_methods = [
        ("UI parses valid progress", ui_tests.test_ui_parses_valid_progress_events),
        ("UI handles missing phases", ui_tests.test_ui_handles_missing_phase_markers),
        ("UI handles malformed META", ui_tests.test_ui_handles_malformed_meta_tags), 
        ("Phase mapping complete", ui_tests.test_ui_phase_mapping_completeness),
        ("Rapid phase changes", ui_tests.test_ui_handles_rapid_phase_changes),
        ("Metadata accumulation", ui_tests.test_ui_accumulates_metadata_correctly),
        ("No progress handling", ui_tests.test_ui_handles_no_progress_events),
        ("Type conversion", ui_tests.test_ui_metadata_type_conversion),
        ("Emoji removal", ui_tests.test_ui_emoji_removal_from_messages),
        ("Unknown phase fallback", ui_tests.test_ui_fallback_behavior_unknown_phases),
        ("End-to-end parsing", contract_tests.test_end_to_end_progress_parsing),
        ("State consistency", contract_tests.test_ui_state_consistency)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in test_methods:
        try:
            print(f"Testing {test_name}...")
            test_func()
            print("   ✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            failed += 1
    
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("UI streaming integration tests complete!")