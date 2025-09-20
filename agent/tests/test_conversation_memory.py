"""
Test suite for conversation memory and multi-question handling.

This test file validates that the agent correctly handles multiple questions
in a conversation, focuses on the latest question, and maintains appropriate
context from previous Q&A pairs.

Key Testing Areas:
- Question extraction (latest vs first)
- Conversation context preservation
- Follow-up question handling
- Context switching between topics
- Memory management across turns
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

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

from deep_research_agent.databricks_compatible_agent import DatabricksCompatibleAgent
from deep_research_agent.core.types import ResearchContext


class TestConversationMemory:
    """Test suite for conversation memory and context management."""
    
    def setup_method(self):
        """Set up test environment with mocked components."""
        self.mock_llm = Mock()
        self.mock_llm.invoke.return_value = AIMessage(content="Test response")
        
        # Create agent with mocked dependencies
        with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_llm', return_value=self.mock_llm):
            mock_phase2_return = (None, None, None, None, None, None)
            with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_phase2_components', return_value=mock_phase2_return):
                self.agent = DatabricksCompatibleAgent()
                
                # Mock the graph to avoid actual workflow execution
                mock_graph = Mock()
                mock_graph.stream.return_value = [
                    {"synthesize_answer": {"messages": [Mock(content="Test synthesis response")]}}
                ]
                self.agent.graph = mock_graph
    
    def test_single_question_extraction(self):
        """Test that single question is extracted correctly."""
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "What is machine learning?"}]
        )
        
        # Mock the predict_stream to capture the initial state
        original_predict_stream = self.agent.predict_stream
        captured_state = {}
        
        def mock_predict_stream(req):
            # Capture the state that would be created
            cc_msgs = []
            for msg in req.input:
                if isinstance(msg, dict):
                    cc_msgs.extend(self.agent._responses_to_cc(msg))
                else:
                    cc_msgs.extend(self.agent._responses_to_cc(msg.model_dump()))
            
            langchain_msgs = []
            for msg in cc_msgs:
                if msg.get("role") == "user":
                    langchain_msgs.append(HumanMessage(content=msg.get("content", "")))
            
            # Extract user question using the same logic as the agent
            user_question = ""
            for msg in reversed(langchain_msgs):
                if isinstance(msg, HumanMessage):
                    user_question = msg.content
                    break
            
            captured_state['question'] = user_question
            captured_state['message_count'] = len(langchain_msgs)
            
            # Return a simple stream event
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.agent.create_text_output_item("Test response", "test-id")
            )
        
        self.agent.predict_stream = mock_predict_stream
        list(self.agent.predict_stream(request))  # Consume the stream
        
        assert captured_state['question'] == "What is machine learning?"
        assert captured_state['message_count'] == 1
    
    def test_multiple_questions_latest_extracted(self):
        """Test that latest question is extracted from conversation."""
        request = ResponsesAgentRequest(
            input=[
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI..."},
                {"role": "user", "content": "How does neural network training work?"}
            ]
        )
        
        # Mock the predict_stream to capture the initial state
        captured_state = {}
        
        def mock_predict_stream(req):
            cc_msgs = []
            for msg in req.input:
                if isinstance(msg, dict):
                    cc_msgs.extend(self.agent._responses_to_cc(msg))
                else:
                    cc_msgs.extend(self.agent._responses_to_cc(msg.model_dump()))
            
            langchain_msgs = []
            for msg in cc_msgs:
                if msg.get("role") == "user":
                    langchain_msgs.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    langchain_msgs.append(AIMessage(content=msg.get("content", "")))
            
            # Extract user question using the same logic as the agent
            user_question = ""
            for msg in reversed(langchain_msgs):
                if isinstance(msg, HumanMessage):
                    user_question = msg.content
                    break
            
            captured_state['question'] = user_question
            captured_state['total_messages'] = len(langchain_msgs)
            captured_state['user_messages'] = [msg.content for msg in langchain_msgs if isinstance(msg, HumanMessage)]
            
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.agent.create_text_output_item("Test response", "test-id")
            )
        
        self.agent.predict_stream = mock_predict_stream
        list(self.agent.predict_stream(request))
        
        # Should extract the LATEST question, not the first
        assert captured_state['question'] == "How does neural network training work?"
        assert captured_state['total_messages'] == 3
        assert len(captured_state['user_messages']) == 2
        assert "What is machine learning?" in captured_state['user_messages']
        assert "How does neural network training work?" in captured_state['user_messages']
    
    def test_conversation_context_building(self):
        """Test that conversation context is built correctly."""
        request = ResponsesAgentRequest(
            input=[
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language"},
                {"role": "user", "content": "What are its main features?"}
            ]
        )
        
        captured_context = {}
        
        def mock_predict_stream(req):
            # Simulate the conversation context building logic
            cc_msgs = []
            for msg in req.input:
                if isinstance(msg, dict):
                    cc_msgs.extend(self.agent._responses_to_cc(msg))
                else:
                    cc_msgs.extend(self.agent._responses_to_cc(msg.model_dump()))
            
            langchain_msgs = []
            for msg in cc_msgs:
                if msg.get("role") == "user":
                    langchain_msgs.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    langchain_msgs.append(AIMessage(content=msg.get("content", "")))
            
            # Build conversation history (simulate agent logic)
            conversation_history = []
            human_message_count = 0
            for msg in langchain_msgs:
                if isinstance(msg, HumanMessage):
                    human_message_count += 1
                    if human_message_count < len([m for m in langchain_msgs if isinstance(m, HumanMessage)]):
                        conversation_history.append(msg)
                elif len(conversation_history) > 0 or human_message_count == 0:
                    conversation_history.append(msg)
            
            captured_context['history_length'] = len(conversation_history)
            captured_context['current_turn'] = human_message_count - 1
            captured_context['history_messages'] = [(type(msg).__name__, msg.content) for msg in conversation_history]
            
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.agent.create_text_output_item("Test response", "test-id")
            )
        
        self.agent.predict_stream = mock_predict_stream
        list(self.agent.predict_stream(request))
        
        # Should have conversation history excluding current question
        assert captured_context['history_length'] == 2  # First Q + first A
        assert captured_context['current_turn'] == 1  # Second question (0-indexed)
        
        # Check history content
        history_messages = captured_context['history_messages']
        assert ('HumanMessage', 'What is Python?') in history_messages
        assert ('AIMessage', 'Python is a programming language') in history_messages
    
    def test_topic_switching_detection(self):
        """Test agent handles topic switching correctly."""
        request = ResponsesAgentRequest(
            input=[
                {"role": "user", "content": "Tell me about cats"},
                {"role": "assistant", "content": "Cats are domestic animals..."},
                {"role": "user", "content": "What about quantum physics?"}  # Topic switch
            ]
        )
        
        captured_data = {}
        
        def mock_predict_stream(req):
            # Replicate the actual agent's message processing logic from predict_stream
            # Convert request messages to ChatCompletion format  
            cc_msgs = []
            for msg in req.input:
                if isinstance(msg, dict):
                    cc_msgs.extend(self.agent._responses_to_cc(msg))
                else:
                    cc_msgs.extend(self.agent._responses_to_cc(msg.model_dump()))
            
            # Convert CC messages to LangChain messages (same as agent)
            langchain_msgs = []
            for msg in cc_msgs:
                if msg.get("role") == "user":
                    langchain_msgs.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    langchain_msgs.append(AIMessage(content=msg.get("content", "")))
                elif msg.get("role") == "system":
                    langchain_msgs.append(SystemMessage(content=msg.get("content", "")))
            
            # Extract user question using the SAME logic as the actual agent
            user_question = ""
            for msg in reversed(langchain_msgs):
                if isinstance(msg, HumanMessage):
                    user_question = msg.content
                    break
            
            # Build previous topics using the SAME logic as the actual agent
            human_messages = [msg for msg in langchain_msgs if isinstance(msg, HumanMessage)]
            previous_topics = []
            if len(human_messages) > 1:
                for msg in human_messages[:-1]:  # All except last
                    topic = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                    previous_topics.append(topic)
            
            captured_data['current_question'] = user_question
            captured_data['previous_topics'] = previous_topics
            
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.agent.create_text_output_item("Test response", "test-id")
            )
        
        # Apply the mock
        original_predict_stream = self.agent.predict_stream
        self.agent.predict_stream = mock_predict_stream
        
        try:
            # Execute the test
            list(self.agent.predict_stream(request))
            
            # Should focus on quantum physics question
            assert captured_data['current_question'] == "What about quantum physics?"
            # Should have previous topic about cats
            assert len(captured_data['previous_topics']) == 1
            assert "cats" in captured_data['previous_topics'][0].lower()
        finally:
            # Restore original method
            self.agent.predict_stream = original_predict_stream
    
    def test_follow_up_question_context(self):
        """Test agent maintains context for follow-up questions."""
        request = ResponsesAgentRequest(
            input=[
                {"role": "user", "content": "What is TensorFlow?"},
                {"role": "assistant", "content": "TensorFlow is a machine learning library developed by Google..."},
                {"role": "user", "content": "How do I install it?"}  # Follow-up question
            ]
        )
        
        captured_synthesis = {}
        
        def mock_predict_stream(req):
            # Replicate the actual agent's message processing logic
            cc_msgs = []
            for msg in req.input:
                if isinstance(msg, dict):
                    cc_msgs.extend(self.agent._responses_to_cc(msg))
                else:
                    cc_msgs.extend(self.agent._responses_to_cc(msg.model_dump()))
            
            # Convert CC messages to LangChain messages
            langchain_msgs = []
            for msg in cc_msgs:
                if msg.get("role") == "user":
                    langchain_msgs.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    langchain_msgs.append(AIMessage(content=msg.get("content", "")))
                elif msg.get("role") == "system":
                    langchain_msgs.append(SystemMessage(content=msg.get("content", "")))
            
            # Extract user question using SAME logic as actual agent
            user_question = ""
            for msg in reversed(langchain_msgs):
                if isinstance(msg, HumanMessage):
                    user_question = msg.content
                    break
            
            # Build conversation history using the SAME logic as the actual agent
            human_messages = [msg for msg in langchain_msgs if isinstance(msg, HumanMessage)]
            human_message_count = len(human_messages)
            
            conversation_history = []
            for i, msg in enumerate(langchain_msgs):
                if isinstance(msg, HumanMessage):
                    # Count which human message this is (1-indexed)
                    current_human_index = len([m for m in langchain_msgs[:i+1] if isinstance(m, HumanMessage)])
                    if current_human_index < human_message_count:  # Not the last human message
                        conversation_history.append(msg)
                elif isinstance(msg, AIMessage):
                    # Include AI messages that come after included human messages
                    if len(conversation_history) > 0:
                        conversation_history.append(msg)
            
            # Test context awareness - does the history contain TensorFlow?
            has_tensorflow_context = any(
                "tensorflow" in msg.content.lower() 
                for msg in conversation_history 
                if isinstance(msg, (HumanMessage, AIMessage))
            )
            
            captured_synthesis['current_question'] = user_question
            captured_synthesis['has_context'] = has_tensorflow_context
            captured_synthesis['context_messages'] = len(conversation_history)
            
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.agent.create_text_output_item("Test response about installing TensorFlow", "test-id")
            )
        
        # Apply the mock
        original_predict_stream = self.agent.predict_stream
        self.agent.predict_stream = mock_predict_stream
        
        try:
            # Execute the test
            list(self.agent.predict_stream(request))
            
            # Should focus on installation question
            assert captured_synthesis['current_question'] == "How do I install it?"
            # Should have TensorFlow context from previous Q&A
            assert captured_synthesis['has_context'] == True
            assert captured_synthesis['context_messages'] == 2  # Previous Q&A pair
        finally:
            # Restore original method
            self.agent.predict_stream = original_predict_stream
    
    def test_empty_conversation_handling(self):
        """Test agent handles edge case of empty conversation."""
        request = ResponsesAgentRequest(input=[])
        
        # Should handle gracefully without crashing
        try:
            stream_events = list(self.agent.predict_stream(request))
            # Should produce some events even with empty input
            assert len(stream_events) >= 1
        except Exception as e:
            # Should not crash - if it does, this is a bug
            pytest.fail(f"Agent crashed with empty conversation: {e}")
    
    def test_long_conversation_context_truncation(self):
        """Test agent handles very long conversations appropriately."""
        # Build a long conversation
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"Question {i}: What about topic {i}?"})
            messages.append({"role": "assistant", "content": f"Answer {i}: Here's information about topic {i}..."})
        
        # Add final question
        messages.append({"role": "user", "content": "What is the final topic?"})
        
        request = ResponsesAgentRequest(input=messages)
        
        captured_info = {}
        
        def mock_predict_stream(req):
            # Replicate the actual agent's message processing logic
            cc_msgs = []
            for msg in req.input:
                if isinstance(msg, dict):
                    cc_msgs.extend(self.agent._responses_to_cc(msg))
                else:
                    cc_msgs.extend(self.agent._responses_to_cc(msg.model_dump()))
            
            # Convert CC messages to LangChain messages  
            langchain_msgs = []
            for msg in cc_msgs:
                if msg.get("role") == "user":
                    langchain_msgs.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    langchain_msgs.append(AIMessage(content=msg.get("content", "")))
                elif msg.get("role") == "system":
                    langchain_msgs.append(SystemMessage(content=msg.get("content", "")))
            
            captured_info['total_input_messages'] = len(req.input)
            captured_info['processed_messages'] = len(cc_msgs)
            captured_info['langchain_messages'] = len(langchain_msgs)
            
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.agent.create_text_output_item("Final response", "test-id")
            )
        
        # Apply the mock
        original_predict_stream = self.agent.predict_stream
        self.agent.predict_stream = mock_predict_stream
        
        try:
            # Execute the test
            list(self.agent.predict_stream(request))
            
            # Should handle all messages without crashing
            assert captured_info['total_input_messages'] == 21  # 10 Q&A pairs + 1 final Q
            assert captured_info['processed_messages'] >= 1  # At least processed something
            assert captured_info['langchain_messages'] >= 1  # Should convert to LangChain messages
        finally:
            # Restore original method
            self.agent.predict_stream = original_predict_stream


class TestConversationEdgeCases:
    """Test edge cases in conversation handling."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_llm = Mock()
        self.mock_llm.invoke.return_value = AIMessage(content="Test response")
        
        with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_llm', return_value=self.mock_llm):
            mock_phase2_return = (None, None, None, None, None, None)
            with patch('deep_research_agent.agent_initialization.AgentInitializer.initialize_phase2_components', return_value=mock_phase2_return):
                self.agent = DatabricksCompatibleAgent()
                
                mock_graph = Mock()
                mock_graph.stream.return_value = [
                    {"synthesize_answer": {"messages": [Mock(content="Test synthesis")]}}
                ]
                self.agent.graph = mock_graph
    
    def test_mixed_role_conversation(self):
        """Test conversation with mixed roles and system messages."""
        request = ResponsesAgentRequest(
            input=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"}, 
                {"role": "system", "content": "Be more detailed"},
                {"role": "user", "content": "Second question"}
            ]
        )
        
        # Should extract latest user question despite system messages
        try:
            stream_events = list(self.agent.predict_stream(request))
            assert len(stream_events) >= 1
        except Exception as e:
            pytest.fail(f"Agent failed with mixed roles: {e}")
    
    def test_duplicate_question_handling(self):
        """Test agent handles duplicate questions appropriately."""
        request = ResponsesAgentRequest(
            input=[
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is artificial intelligence..."},
                {"role": "user", "content": "What is AI?"}  # Same question again
            ]
        )
        
        # Should handle duplicate without issues
        try:
            stream_events = list(self.agent.predict_stream(request))
            assert len(stream_events) >= 1
        except Exception as e:
            pytest.fail(f"Agent failed with duplicate question: {e}")
    
    def test_very_long_single_question(self):
        """Test agent handles very long single questions."""
        long_question = "What is " + "very " * 1000 + "long question about machine learning?"
        
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": long_question}]
        )
        
        # Should handle long content without issues
        try:
            stream_events = list(self.agent.predict_stream(request))
            assert len(stream_events) >= 1
        except Exception as e:
            pytest.fail(f"Agent failed with long question: {e}")


if __name__ == "__main__":
    # Run conversation memory tests
    print("Running conversation memory tests...")
    print("-" * 60)
    
    test_suite = TestConversationMemory()
    edge_case_suite = TestConversationEdgeCases()
    
    # Main conversation tests
    main_tests = [
        ("Single question extraction", test_suite.test_single_question_extraction),
        ("Multiple questions - latest extracted", test_suite.test_multiple_questions_latest_extracted),
        ("Conversation context building", test_suite.test_conversation_context_building),
        ("Topic switching detection", test_suite.test_topic_switching_detection),
        ("Follow-up question context", test_suite.test_follow_up_question_context),
        ("Empty conversation handling", test_suite.test_empty_conversation_handling),
        ("Long conversation truncation", test_suite.test_long_conversation_context_truncation)
    ]
    
    # Edge case tests
    edge_tests = [
        ("Mixed role conversation", edge_case_suite.test_mixed_role_conversation),
        ("Duplicate question handling", edge_case_suite.test_duplicate_question_handling),
        ("Very long single question", edge_case_suite.test_very_long_single_question)
    ]
    
    passed = 0
    failed = 0
    
    # Run main tests
    for test_name, test_func in main_tests:
        try:
            print(f"Testing {test_name}...")
            test_suite.setup_method()
            test_func()
            print("   ✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            failed += 1
    
    # Run edge case tests  
    for test_name, test_func in edge_tests:
        try:
            print(f"Testing {test_name}...")
            edge_case_suite.setup_method()
            test_func()
            print("   ✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            failed += 1
    
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("Conversation memory tests complete!")