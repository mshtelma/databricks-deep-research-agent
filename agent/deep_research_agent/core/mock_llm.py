"""
Mock LLM for unit tests only.

Provides deterministic, context-aware responses for testing.
Should never be used in production or integration tests.
"""

from typing import List, Dict, Any, Union
from langchain_core.messages import AIMessage, BaseMessage
import json
import hashlib
from deep_research_agent.core import get_logger

logger = get_logger(__name__)


class MockLLM:
    """
    Mock LLM for unit tests only.
    
    Provides context-aware, deterministic responses based on prompt patterns.
    This allows unit tests to run without real LLM endpoints while still
    testing the logic around LLM interactions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.call_history = []
        logger.debug("MockLLM initialized for testing")
    
    def invoke(self, messages: Union[List[BaseMessage], List[Dict]], **kwargs) -> AIMessage:
        """Synchronous invoke with context-aware responses."""
        # Handle both BaseMessage objects and dict formats
        if messages and isinstance(messages[0], dict):
            last_msg = messages[-1].get('content', '') if messages else ""
        else:
            last_msg = messages[-1].content if messages else ""
        
        self.call_history.append(last_msg)
        
        # Generate response based on context
        response = self._generate_response(last_msg)
        return AIMessage(content=response)
    
    async def ainvoke(self, messages: Union[List[BaseMessage], List[Dict]], **kwargs) -> AIMessage:
        """Async version delegates to sync."""
        return self.invoke(messages, **kwargs)
    
    def _generate_response(self, prompt: str) -> str:
        """Generate context-appropriate response."""
        prompt_lower = prompt.lower()
        
        # Coordinator - Classification
        if "classify" in prompt_lower or "what type" in prompt_lower:
            return "RESEARCH"
        
        # Fact Checker - Claim extraction
        if "extract" in prompt_lower and "claim" in prompt_lower:
            return self._mock_claims_extraction()
        
        # Planner - Research planning
        if "plan" in prompt_lower or "steps" in prompt_lower or "research plan" in prompt_lower:
            return self._mock_research_plan()
        
        # Researcher - Query generation
        if "search" in prompt_lower and ("query" in prompt_lower or "queries" in prompt_lower):
            return self._mock_search_queries()
        
        # Reporter - Synthesis/reporting
        if ("synthesize" in prompt_lower or "report" in prompt_lower or 
            "write" in prompt_lower or "compile" in prompt_lower):
            return self._mock_report_synthesis()
        
        # Default response
        return f"Mock response for prompt: {prompt[:100]}"
    
    def _mock_claims_extraction(self) -> str:
        """Mock claims for fact checking."""
        return """1. Bulgaria has a 10% flat tax rate for personal income
2. France's top marginal tax rate reaches 45% for high earners
3. UK income tax bands are 20%, 40%, and 45%
4. Switzerland's Zug canton has approximately 22% effective tax rate
5. Germany's top tax rate is 42% plus solidarity surcharge
6. Poland offers a 19% flat tax option for qualifying individuals
7. Spain's top marginal rate is 47% for incomes over €300,000
8. RSUs are generally taxed as ordinary income at vesting
9. Child benefits are available in all EU countries
10. Daycare costs vary significantly from €200-2000 per month across Europe"""
    
    def _mock_research_plan(self) -> str:
        """Mock research plan for planner."""
        return """Research Plan for Tax Comparison:

Step 1: Research income tax rates and brackets for each country
- Focus: Current 2024/2025 tax year rates
- Include: Marginal rates, effective rates for specified income levels

Step 2: Investigate RSU taxation rules and timing
- Focus: Treatment at vesting vs. sale
- Include: Capital gains implications

Step 3: Gather childcare costs and availability data
- Focus: Full-time daycare for 1 toddler
- Include: Public vs private options, subsidies

Step 4: Research family benefits and allowances
- Focus: Child benefits, tax credits, family quotient systems
- Include: High-income clawbacks

Step 5: Compile rent data for specified cities
- Focus: 2-3 bedroom apartments in upper-middle-class areas
- Include: Recent market rates

Step 6: Calculate net disposable income
- Focus: Take-home pay minus expenses plus benefits
- Include: All three family scenarios"""
    
    def _mock_search_queries(self) -> str:
        """Mock search queries for researcher."""
        return """tax rates Bulgaria France UK Switzerland Germany Poland Spain 2024
RSU taxation rules Europe employment income capital gains
childcare costs daycare Europe 2024 toddler full-time
family benefits child allowance Europe high income
rent prices Madrid Paris London Zug Frankfurt Warsaw Sofia 2024"""
    
    def _mock_report_synthesis(self) -> str:
        """Mock report synthesis for reporter."""
        return """# Tax and Cost of Living Comparison Across Europe

## Executive Summary

Our analysis reveals significant variations in tax burden and disposable income across the seven European countries examined. Bulgaria offers the most favorable tax environment with a 10% flat rate, while Spain and France impose the highest marginal rates at 47% and 45% respectively.

## Key Findings

### Tax Rates
- **Lowest**: Bulgaria (10% flat rate)
- **Highest**: Spain (47% top rate), France (45% top rate)
- **Most Complex**: Germany (42% + solidarity surcharge)

### Family Benefits
- Nordic influence: Higher benefits in countries with higher taxes
- **Most Generous**: France (family quotient system)
- **Least Generous**: Switzerland (minimal federal benefits)

### Childcare Costs
- **Highest**: UK and Switzerland (€1,500-2,000/month)
- **Lowest**: Poland and Bulgaria (€200-500/month)
- **Best Subsidized**: France and Germany

### Rent Variations
- **Most Expensive**: London, Zug, Paris
- **Most Affordable**: Sofia, Warsaw
- **Mid-Range**: Madrid, Frankfurt

## Recommendations

For the specified income levels (€150k + €100k RSUs), the optimal locations considering net disposable income are:
1. Switzerland (Zug) - Despite high costs, low taxes prevail
2. Bulgaria - Lowest tax burden, affordable living costs
3. Germany - Balanced approach with good benefits

*Note: This analysis is based on 2024 tax rules and market data.*"""
    
    def get_call_count(self) -> int:
        """Get number of calls made to this mock."""
        return len(self.call_history)
    
    def get_last_prompt(self) -> str:
        """Get the last prompt sent to this mock."""
        return self.call_history[-1] if self.call_history else ""
    
    def reset_history(self) -> None:
        """Reset call history (useful for test isolation)."""
        self.call_history = []