"""
Tests for redaction utilities.

This module tests the PII/secret redaction functionality to ensure
sensitive information is properly sanitized from intermediate events.
"""

import pytest
from deep_research_agent.core.redaction_utils import (
    RedactionUtils, 
    get_redactor, 
    redact_text, 
    redact_dict, 
    sanitize_for_ui
)


class TestRedactionUtils:
    """Test redaction utilities functionality."""
    
    def test_default_patterns_initialization(self):
        """Test that default redaction patterns are loaded."""
        redactor = RedactionUtils()
        assert len(redactor.patterns) > 0
        assert len(redactor._compiled_patterns) > 0
        
    def test_custom_patterns_initialization(self):
        """Test initialization with custom patterns."""
        custom_patterns = [r'\btest_secret_\w+\b']
        redactor = RedactionUtils(custom_patterns)
        
        # Should include both default and custom patterns
        assert len(redactor.patterns) > len(RedactionUtils.DEFAULT_PATTERNS)
        assert custom_patterns[0] in redactor.patterns
    
    def test_email_redaction(self):
        """Test email address redaction."""
        redactor = RedactionUtils()
        
        test_cases = [
            ("Contact user@example.com for help", "Contact [REDACTED] for help"),
            ("Email: john.doe+test@company.co.uk", "Email: [REDACTED]"),
            ("Multiple emails: a@b.com and test@example.org", "Multiple emails: [REDACTED] and [REDACTED]"),
            ("No emails here", "No emails here"),
        ]
        
        for input_text, expected in test_cases:
            result = redactor.redact_text(input_text)
            assert result == expected, f"Failed for: {input_text}"
    
    def test_api_key_redaction(self):
        """Test API key redaction."""
        redactor = RedactionUtils()
        
        test_cases = [
            ("API key: sk-1234567890abcdef1234567890abcdef", "API key: [REDACTED]"),
            ("Bearer token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", "Bearer [REDACTED]"),
            ("Long key: abcdefghijklmnopqrstuvwxyz123456", "Long key: [REDACTED]"),
            ("Short key: abc123", "Short key: abc123"),  # Should not be redacted
        ]
        
        for input_text, expected in test_cases:
            result = redactor.redact_text(input_text)
            assert result == expected, f"Failed for: {input_text}"
    
    def test_ssn_redaction(self):
        """Test SSN redaction."""
        redactor = RedactionUtils()
        
        test_cases = [
            ("SSN: 123-45-6789", "SSN: [REDACTED]"),
            ("Social Security: 987654321", "Social Security: 987654321"),  # No dashes
            ("Invalid: 123-456-789", "Invalid: 123-456-789"),  # Wrong format
        ]
        
        for input_text, expected in test_cases:
            result = redactor.redact_text(input_text)
            assert result == expected, f"Failed for: {input_text}"
    
    def test_credit_card_redaction(self):
        """Test credit card redaction."""
        redactor = RedactionUtils()
        
        test_cases = [
            ("Card: 1234 5678 9012 3456", "Card: [REDACTED]"),
            ("Card: 1234-5678-9012-3456", "Card: [REDACTED]"),
            ("Card: 1234567890123456", "Card: [REDACTED]"),
            ("Short: 1234 5678", "Short: 1234 5678"),  # Too short
        ]
        
        for input_text, expected in test_cases:
            result = redactor.redact_text(input_text)
            assert result == expected, f"Failed for: {input_text}"
    
    def test_dict_redaction(self):
        """Test dictionary redaction."""
        redactor = RedactionUtils()
        
        test_dict = {
            "query": "Search for user@example.com",
            "api_key": "sk-1234567890abcdef1234567890abcdef",
            "user_email": "john@company.com",
            "metadata": {
                "contact": "support@service.com",
                "count": 5
            },
            "list_data": [
                "email: test@example.org",
                "safe text",
                {"nested_email": "nested@test.com"}
            ]
        }
        
        result = redactor.redact_dict(test_dict)
        
        # Check that emails and API keys were redacted
        assert result["query"] == "Search for [REDACTED]"
        assert result["api_key"] == "[REDACTED]"
        assert result["user_email"] == "[REDACTED]"
        assert result["metadata"]["contact"] == "[REDACTED]"
        assert result["metadata"]["count"] == 5  # Numbers should be unchanged
        assert result["list_data"][0] == "email: [REDACTED]"
        assert result["list_data"][1] == "safe text"  # Safe text unchanged
        assert result["list_data"][2]["nested_email"] == "[REDACTED]"
    
    def test_list_redaction(self):
        """Test list redaction."""
        redactor = RedactionUtils()
        
        test_list = [
            "Contact user@example.com",
            {"key": "sk-abcdefghijklmnopqrstuvwxyz123456"},
            ["nested", "admin@test.com"],
            42,
            None
        ]
        
        result = redactor.redact_list(test_list)
        
        assert result[0] == "Contact [REDACTED]"
        assert result[1]["key"] == "[REDACTED]"
        assert result[2][0] == "nested"
        assert result[2][1] == "[REDACTED]"
        assert result[3] == 42
        assert result[4] is None
    
    def test_text_truncation(self):
        """Test text truncation functionality."""
        redactor = RedactionUtils()
        
        test_cases = [
            ("Short text", 20, "Short text"),
            ("This is a longer text that should be truncated", 20, "This is a longer..."),
            ("Exactly twenty chars", 20, "Exactly twenty chars"),
            ("ABC", 2, "AB"),  # Very short limit
            ("", 10, ""),  # Empty string
        ]
        
        for text, max_length, expected in test_cases:
            result = redactor.truncate_text(text, max_length)
            assert result == expected, f"Failed for: '{text}' with max_length {max_length}"
    
    def test_sanitize_for_ui(self):
        """Test UI sanitization combining redaction and truncation."""
        redactor = RedactionUtils()
        
        long_text_with_email = "This is a long message containing user@example.com and should be truncated"
        result = redactor.sanitize_for_ui(long_text_with_email, max_length=30)
        
        # Should be redacted and truncated
        assert "[REDACTED]" in result
        assert len(result) <= 30
        assert result.endswith("...")
    
    def test_custom_redaction_patterns(self):
        """Test custom redaction patterns."""
        custom_patterns = [r'\bsecret_\w+\b', r'\bapi_token_\w+\b']
        redactor = RedactionUtils(custom_patterns)
        
        test_text = "Use secret_abc123 and api_token_xyz789 for authentication"
        result = redactor.redact_text(test_text)
        
        assert result == "Use [REDACTED] and [REDACTED] for authentication"
    
    def test_invalid_regex_patterns(self):
        """Test handling of invalid regex patterns."""
        # Invalid regex pattern
        invalid_patterns = [r'[unclosed', r'(?invalid']
        
        # Should not raise exception, just log warning
        redactor = RedactionUtils(invalid_patterns)
        
        # Should still work with valid default patterns
        result = redactor.redact_text("Email: test@example.com")
        assert result == "Email: [REDACTED]"
    
    def test_non_string_input_handling(self):
        """Test handling of non-string inputs."""
        redactor = RedactionUtils()
        
        # Should convert to string and process
        assert redactor.redact_text(123) == "123"
        assert redactor.redact_text(None) == "None"
        assert redactor.redact_text({"key": "value"}) == "{'key': 'value'}"
    
    def test_global_redactor_functions(self):
        """Test global convenience functions."""
        text_with_email = "Contact admin@company.com"
        
        # Test global redact_text function
        result = redact_text(text_with_email)
        assert result == "Contact [REDACTED]"
        
        # Test global redact_dict function
        test_dict = {"email": "user@test.com", "name": "John"}
        result = redact_dict(test_dict)
        assert result["email"] == "[REDACTED]"
        assert result["name"] == "John"
        
        # Test global sanitize_for_ui function
        long_text = "Long text with admin@company.com that needs truncation"
        result = sanitize_for_ui(long_text, max_length=20)
        assert "[REDACTED]" in result
        assert len(result) <= 20
    
    def test_event_data_redaction(self):
        """Test redaction of event data structure."""
        redactor = RedactionUtils()
        
        event_data = {
            "action": "search",
            "query": "Find information about user@company.com",
            "parameters": {
                "api_key": "sk-1234567890abcdef1234567890abcdef",
                "max_results": 5
            },
            "result_summary": "Found 3 results for admin@service.org",
            "metadata": {
                "execution_time": 1.23,
                "bearer_token": "Bearer abc123def456ghi789"
            }
        }
        
        result = redactor.redact_event_data(event_data)
        
        assert result["action"] == "search"  # Should be unchanged
        assert result["query"] == "Find information about [REDACTED]"
        assert result["parameters"]["api_key"] == "[REDACTED]"
        assert result["parameters"]["max_results"] == 5  # Should be unchanged
        assert result["result_summary"] == "Found 3 results for [REDACTED]"
        assert result["metadata"]["execution_time"] == 1.23  # Should be unchanged
        assert result["metadata"]["bearer_token"] == "[REDACTED]"


class TestRedactionIntegration:
    """Integration tests for redaction utilities."""
    
    def test_redaction_with_real_agent_data(self):
        """Test redaction with realistic agent event data."""
        redactor = RedactionUtils()
        
        # Simulate realistic intermediate event data
        event_data = {
            "tool_name": "brave_search",
            "query": "How to contact support@databricks.com for API issues",
            "results": [
                {
                    "title": "Contact Support",
                    "url": "https://databricks.com/contact",
                    "content": "Email us at help@databricks.com or call support"
                }
            ],
            "api_response": {
                "headers": {
                    "authorization": "Bearer sk-proj-abcdefghijklmnopqrstuvwxyz1234567890",
                    "user-agent": "agent/1.0"
                }
            },
            "user_context": {
                "workspace_id": "12345",
                "user_email": "john.doe@company.com"
            }
        }
        
        result = redactor.redact_event_data(event_data)
        
        # Verify all sensitive data is redacted
        assert "[REDACTED]" in result["query"]
        assert "[REDACTED]" in result["results"][0]["content"]
        assert result["api_response"]["headers"]["authorization"] == "[REDACTED]"
        assert result["user_context"]["user_email"] == "[REDACTED]"
        
        # Verify non-sensitive data is preserved
        assert result["tool_name"] == "brave_search"
        assert result["results"][0]["title"] == "Contact Support"
        assert result["user_context"]["workspace_id"] == "12345"
    
    def test_performance_with_large_data(self):
        """Test redaction performance with large data structures."""
        redactor = RedactionUtils()
        
        # Create large data structure
        large_data = {
            "messages": [
                {
                    "id": i,
                    "content": f"Message {i} with email user{i}@test.com and key sk-{i:032d}",
                    "metadata": {
                        "timestamp": f"2024-01-{i:02d}",
                        "user": f"user{i}@company.com"
                    }
                }
                for i in range(100)
            ]
        }
        
        import time
        start_time = time.time()
        result = redactor.redact_dict(large_data)
        end_time = time.time()
        
        # Should complete reasonably quickly (< 1 second)
        assert end_time - start_time < 1.0
        
        # Verify redaction worked
        assert "[REDACTED]" in result["messages"][0]["content"]
        assert result["messages"][0]["metadata"]["user"] == "[REDACTED]"


if __name__ == "__main__":
    pytest.main([__file__])
