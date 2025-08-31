"""
Tests for Brave Search integration.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent.absolute()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from deep_research_agent.tools_brave import BraveSearchTool, create_brave_tool
from deep_research_agent.components.tool_manager import BraveToolFactory
from deep_research_agent.core.types import ToolType, ToolConfiguration
from deep_research_agent.core.exceptions import ToolInitializationError


class TestBraveSearchTool:
    """Test suite for Brave Search Tool."""
    
    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        tool = BraveSearchTool(api_key="test_api_key")
        assert tool.api_key == "test_api_key"
        assert tool.base_url == "https://api.search.brave.com/res/v1"
        assert tool.name == "brave_search"
    
    def test_init_from_env_var(self):
        """Test initialization from environment variable."""
        with patch.dict(os.environ, {"BRAVE_API_KEY": "env_api_key"}):
            tool = BraveSearchTool()
            assert tool.api_key == "env_api_key"
    
    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ToolInitializationError, match="Brave Search API key is required but not provided"):
                BraveSearchTool()
    
    @patch("deep_research_agent.tools_brave.requests.get")
    def test_search_success(self, mock_get):
        """Test successful search operation."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Test Result 1",
                        "url": "https://example.com/1",
                        "description": "Test description 1"
                    },
                    {
                        "title": "Test Result 2",
                        "url": "https://example.com/2",
                        "description": "Test description 2",
                        "relevance_score": 0.95
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        # Create tool and search
        tool = BraveSearchTool(api_key="test_key")
        results = tool.search("test query", max_results=2)
        
        # Verify request
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://api.search.brave.com/res/v1/web/search"
        assert call_args[1]["params"]["q"] == "test query"
        assert call_args[1]["params"]["count"] == 2
        assert call_args[1]["headers"]["X-Subscription-Token"] == "test_key"
        
        # Verify results
        assert len(results) == 2
        assert results[0]["title"] == "Test Result 1"
        assert results[0]["url"] == "https://example.com/1"
        assert results[0]["content"] == "Test description 1"
        assert results[0]["score"] == 0.0  # No relevance_score in mock
        
        assert results[1]["title"] == "Test Result 2"
        assert results[1]["score"] == 0.95  # Has relevance_score
    
    @patch("deep_research_agent.tools_brave.requests.get")
    def test_search_api_error(self, mock_get):
        """Test handling of API errors."""
        mock_get.side_effect = Exception("API Error")
        
        tool = BraveSearchTool(api_key="test_key")
        
        with pytest.raises(Exception, match="Error processing Brave Search response"):
            tool.search("test query")
    
    def test_create_brave_tool_factory(self):
        """Test factory function."""
        tool = create_brave_tool(api_key="factory_key")
        assert isinstance(tool, BraveSearchTool)
        assert tool.api_key == "factory_key"


class TestBraveToolFactory:
    """Test suite for Brave Tool Factory."""
    
    def test_get_tool_type(self):
        """Test that factory returns correct tool type."""
        factory = BraveToolFactory()
        assert factory.get_tool_type() == ToolType.BRAVE_SEARCH
    
    def test_validate_config_valid(self):
        """Test validation of valid configuration."""
        factory = BraveToolFactory()
        config = ToolConfiguration(
            tool_type=ToolType.BRAVE_SEARCH,
            enabled=True,
            config={
                "api_key": "test_key",
                "max_results": 5
            }
        )
        issues = factory.validate_config(config)
        assert len(issues) == 0
    
    def test_validate_config_disabled(self):
        """Test validation skips for disabled tool."""
        factory = BraveToolFactory()
        config = ToolConfiguration(
            tool_type=ToolType.BRAVE_SEARCH,
            enabled=False,
            config={}
        )
        issues = factory.validate_config(config)
        assert len(issues) == 0
    
    def test_validate_config_missing_api_key(self):
        """Test validation detects missing API key."""
        factory = BraveToolFactory()
        config = ToolConfiguration(
            tool_type=ToolType.BRAVE_SEARCH,
            enabled=True,
            config={"max_results": 5}
        )
        issues = factory.validate_config(config)
        assert "Brave API key is required" in issues
    
    def test_validate_config_secret_placeholder(self):
        """Test validation accepts secret placeholders."""
        factory = BraveToolFactory()
        config = ToolConfiguration(
            tool_type=ToolType.BRAVE_SEARCH,
            enabled=True,
            config={
                "api_key": "{{secrets/msh/BRAVE_API_KEY}}",
                "max_results": 5
            }
        )
        issues = factory.validate_config(config)
        assert len(issues) == 0
    
    def test_validate_config_invalid_max_results(self):
        """Test validation detects invalid max_results."""
        factory = BraveToolFactory()
        config = ToolConfiguration(
            tool_type=ToolType.BRAVE_SEARCH,
            enabled=True,
            config={
                "api_key": "test_key",
                "max_results": "not_a_number"
            }
        )
        issues = factory.validate_config(config)
        assert "max_results must be a positive integer" in issues
    
    @patch("deep_research_agent.tools_brave.BraveSearchTool")
    def test_create_tool_success(self, mock_brave_tool):
        """Test successful tool creation."""
        factory = BraveToolFactory()
        config = ToolConfiguration(
            tool_type=ToolType.BRAVE_SEARCH,
            enabled=True,
            config={
                "api_key": "test_key",
                "base_url": "https://api.brave.com"
            }
        )
        
        mock_tool_instance = Mock()
        mock_brave_tool.return_value = mock_tool_instance
        
        # Patch the import inside the factory's create_tool method
        with patch.dict(sys.modules, {'deep_research_agent.tools_brave': Mock(BraveSearchTool=mock_brave_tool)}):
            tool = factory.create_tool(config)
        
        assert tool == mock_tool_instance
        mock_brave_tool.assert_called_once_with(
            api_key="test_key",
            base_url="https://api.brave.com",
            timeout_seconds=30,
            max_retries=3
        )
    
    def test_create_tool_disabled(self):
        """Test that disabled tool returns None."""
        factory = BraveToolFactory()
        config = ToolConfiguration(
            tool_type=ToolType.BRAVE_SEARCH,
            enabled=False,
            config={"api_key": "test_key"}
        )
        
        tool = factory.create_tool(config)
        assert tool is None
    
    def test_create_tool_no_api_key(self):
        """Test that tool creation fails without API key."""
        factory = BraveToolFactory()
        config = ToolConfiguration(
            tool_type=ToolType.BRAVE_SEARCH,
            enabled=True,
            config={}
        )
        
        with pytest.raises(ToolInitializationError, match="Failed to create Brave tool"):
            factory.create_tool(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])