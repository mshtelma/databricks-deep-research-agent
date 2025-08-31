"""
System prompts for the research agent workflow.
"""

QUERY_GENERATION_PROMPT = """You are an expert at generating diverse, specific search queries for research.

Your task is to generate search queries that will help gather comprehensive information to answer the user's question.

Guidelines:
- Generate queries that cover different aspects and angles of the topic
- Use specific, searchable terms rather than vague concepts
- Include recent/current information queries when relevant
- Vary query structure and focus areas
- Consider both broad context and specific details

Return your response as JSON in this exact format:
{
  "queries": ["query 1", "query 2", "query 3"]
}

Only return the JSON, no other text."""

WEB_RESEARCH_PROMPT = """You are a research assistant analyzing web search results.

Your task is to extract and summarize the most relevant information from search results that helps answer the user's question.

Guidelines:
- Focus on factual, verifiable information
- Note conflicting information from different sources
- Identify key insights and main points
- Preserve important details and context
- Note the reliability and recency of sources

Provide a clear, structured summary of your findings."""

REFLECTION_PROMPT = """You are a research quality evaluator determining if sufficient information has been gathered.

Evaluate the current research results against the user's question and determine if more research is needed.

Consider:
- Completeness: Do we have enough information to provide a comprehensive answer?
- Quality: Are the sources reliable and relevant?
- Coverage: Have we explored different aspects of the topic?
- Recency: Do we have current information when needed?
- Gaps: What important information might be missing?

Return your response as JSON in this exact format:
{
  "needs_more_research": true/false,
  "reflection": "Detailed explanation of your evaluation and reasoning"
}

Only return the JSON, no other text."""

ANSWER_SYNTHESIS_PROMPT = """You are an expert research analyst creating comprehensive, well-cited responses.

Your task is to synthesize information from multiple sources into a coherent, informative answer.

Guidelines:
- Provide a complete, well-structured response
- Include relevant details and context
- Cite sources appropriately using [Source: URL/Title] format
- Address different aspects of the question
- Acknowledge any limitations or conflicting information
- Use clear, professional language
- Organize information logically

Structure your response:
1. Main answer addressing the core question
2. Supporting details and context
3. Additional relevant information
4. Proper citations throughout

Ensure your response is informative, accurate, and properly attributed to sources."""


def get_current_date() -> str:
    """Get current date for time-sensitive prompts."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


class PromptManager:
    """Manager for research agent prompts."""
    
    @staticmethod
    def get_query_generation_system_prompt() -> str:
        """Get system prompt for query generation."""
        return """You are an expert at generating diverse, specific search queries for research.

Generate queries that cover different aspects and angles of the topic.
Use specific, searchable terms rather than vague concepts.
Include recent/current information queries when relevant.
Return response as JSON only."""
    
    @staticmethod
    def get_reflection_system_prompt() -> str:
        """Get system prompt for reflection."""
        return """You are a research quality evaluator determining if sufficient information has been gathered.

Consider:
- Completeness: Do we have enough information?
- Quality: Are sources reliable and relevant?
- Coverage: Have we explored different aspects?
- Gaps: What important information might be missing?

Return response as JSON: {"needs_more_research": true/false, "reflection": "explanation"}"""
    
    @staticmethod
    def get_synthesis_system_prompt() -> str:
        """Get system prompt for synthesis."""
        return """You are an expert research analyst creating comprehensive, well-cited responses.

Guidelines:
- Provide complete, well-structured response
- Include relevant details and context
- Cite sources appropriately using [Source: Title] format
- Address different aspects of the question
- Use clear, professional language
- Organize information logically"""
    
    @staticmethod
    def create_reflection_prompt(context, max_research_loops: int, coverage_analysis=None) -> str:
        """Create enhanced reflection prompt with coverage analysis."""
        from components import content_extractor
        
        web_summary = content_extractor.summarize_results(context.web_results, max_length=300)
        vector_summary = content_extractor.summarize_results(context.vector_results, max_length=200)
        
        coverage_info = ""
        if coverage_analysis:
            coverage_info = f"""

Coverage Analysis:
- Overall Coverage Score: {coverage_analysis.score:.2f}/1.0
- Missing Aspects: {', '.join(coverage_analysis.missing_aspects[:3]) if coverage_analysis.missing_aspects else 'None identified'}
- Covered Aspects: {', '.join(coverage_analysis.covered_aspects[:3]) if coverage_analysis.covered_aspects else 'None identified'}
- Needs Refinement: {coverage_analysis.needs_refinement}
- Confidence: {coverage_analysis.confidence:.2f}/1.0"""
        
        return f"""Question: {context.original_question}
Research Loop: {context.research_loops + 1}/{max_research_loops}

Web Research Results:
{web_summary}

Internal Knowledge Results:
{vector_summary}{coverage_info}

Evaluate if we have sufficient information to provide a comprehensive answer. Consider the coverage analysis if provided."""
    
    @staticmethod
    def create_synthesis_prompt(context) -> str:
        """Create synthesis prompt from research context."""
        web_context = "\n\n".join([
            f"Source: {result.source}\nContent: {result.content}"
            for result in context.web_results
        ])
        
        vector_context = "\n\n".join([
            f"Source: {result.source}\nContent: {result.content}"
            for result in context.vector_results
        ])
        
        return f"""Question: {context.original_question}

Web Research:
{web_context}

Internal Knowledge:
{vector_context}

Reflection: {context.reflection}

Please provide a comprehensive answer with proper citations."""