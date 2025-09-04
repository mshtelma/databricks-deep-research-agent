"""
Workflow node implementations for the research agent.

This module contains all the LangGraph workflow nodes that execute
the research process.
"""

import time
from typing import Dict, Any, List, Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Send

from deep_research_agent.core import (
    get_logger,
    format_duration,
    ResearchContext,
    ResearchQuery,
    SearchResult,
    SearchResultType,
    SearchTaskState,
    WorkflowNodeType,
    URLResolver,
    ToolType,
    QueryComplexity,
    CoverageAnalysis,
    ModelRole,
    safe_json_loads,
    log_execution_time,
    log_function_calls,
    retry_with_exponential_backoff
)
from deep_research_agent.core.error_handler import retry, safe_call
from deep_research_agent.core.schema_converter import global_schema_converter
from deep_research_agent.core.search_provider import SearchProvider, UnifiedSearchManager
from deep_research_agent.core.state_manager import ResearchState
from deep_research_agent.core.cache_manager import global_cache_manager


def extract_text_from_llm_response(content):
    """
    Extract text from LLM response content that may be string or structured list.
    
    Args:
        content: LLM response content (string or list of structured items)
        
    Returns:
        str: Extracted text content
    """
    if isinstance(content, list):
        # Extract text from structured response
        extracted_text = ""
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text' and 'text' in item:
                    extracted_text += item['text'] + "\n"
                elif 'text' in item:
                    extracted_text += str(item['text']) + "\n"
            else:
                extracted_text += str(item) + "\n"
        return extracted_text.strip()
    else:
        return str(content)
from deep_research_agent.components import content_extractor, message_converter
from deep_research_agent.state_management import RefactoredResearchAgentState
from deep_research_agent.prompts import PromptManager

try:
    from databricks_langchain import ChatDatabricks
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False
    ChatDatabricks = None

logger = get_logger(__name__)


class WorkflowNodes:
    """Container for workflow node implementations."""
    
    def __init__(self, agent):
        """Initialize with reference to main agent."""
        self.agent = agent
        self.config_manager = agent.config_manager
        self.agent_config = agent.agent_config
        self.tool_registry = agent.tool_registry
        self.llm = agent.llm
        self.search_semaphore = agent.search_semaphore
        
        # Phase 2 components
        self.model_manager = agent.model_manager
        self.deduplicator = agent.deduplicator
        self.query_analyzer = agent.query_analyzer
        self.result_evaluator = agent.result_evaluator
        self.adaptive_generator = agent.adaptive_generator
    
    @retry("generate_queries", "llm_call")
    def generate_queries_node(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Generate search queries based on user question with adaptive generation."""
        state = RefactoredResearchAgentState.from_dict(state_dict)
        state.current_node = WorkflowNodeType.QUERY_GENERATION
        
        try:
            # Extract user question
            user_question = self._extract_user_question(state.messages)
            if not user_question:
                logger.warning("No user question found in messages")
                return state.to_dict()
            
            # Update research context
            if not state.research_context.original_question:
                state.research_context.original_question = user_question
            
            # Analyze query complexity for adaptive generation
            query_complexity = None
            if self.query_analyzer:
                query_complexity = self.query_analyzer.analyze_query(user_question)
                logger.info(
                    "Query analysis completed",
                    complexity=query_complexity.complexity_level,
                    intent=query_complexity.intent_type,
                    entities_count=len(query_complexity.entities)
                )
            
            # Check cache for similar queries first
            cache_key = f"queries_{hash(user_question)}"
            cached_queries = global_cache_manager.get(cache_key, "query_generation")
            
            if cached_queries is not None:
                queries = cached_queries
                logger.info("Using cached query generation results")
            else:
                queries = self._generate_search_queries(user_question, query_complexity)
                # Cache for 1 hour
                global_cache_manager.set(cache_key, queries, "query_generation", ttl=3600)
            
            # Generate adaptive queries if this is a subsequent loop
            if (state.research_context.research_loops > 0 and 
                self.adaptive_generator and 
                state.research_context.web_results):
                
                # Evaluate current coverage
                coverage_analysis = None
                if self.result_evaluator:
                    coverage_analysis = self.result_evaluator.evaluate_coverage(
                        user_question,
                        state.research_context.web_results,
                        query_complexity
                    )
                
                # Generate adaptive queries
                adaptive_queries = self.adaptive_generator.generate_adaptive_queries(
                    user_question,
                    state.research_context.web_results,
                    coverage_analysis,
                    query_complexity
                )
                
                if adaptive_queries:
                    queries.extend(adaptive_queries)
                    logger.info(
                        f"Generated {len(adaptive_queries)} adaptive queries",
                        adaptive_queries=adaptive_queries
                    )
            
            # Create ResearchQuery objects
            research_queries = [
                ResearchQuery(query=query, priority=i+1)
                for i, query in enumerate(queries)
            ]
            
            state.research_context.generated_queries.extend(research_queries)
            state.update_metrics(total_queries_generated=len(research_queries))
            
            logger.info(
                f"Generated {len(queries)} total search queries",
                queries=queries[:3],  # Log first 3 queries
                total_queries=len(queries)
            )
            
            return state.to_dict()
        except Exception as e:
            logger.error("Query generation failed", error=e)
            state.update_metrics(error_count=state.workflow_metrics.error_count + 1)
            return state.to_dict()
    
    @safe_call("batch_controller", fallback={})
    def batch_controller_node(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Control batch execution of search queries to respect rate limits.
        
        This node manages the sequential execution of query batches, ensuring
        we don't exceed API rate limits by executing too many searches simultaneously.
        """
        state = RefactoredResearchAgentState.from_dict(state_dict)
        state.current_node = WorkflowNodeType.WEB_RESEARCH
        
        try:
            # Get queries to process
            research_queries = state.research_context.generated_queries[:self.agent_config.initial_query_count]
            batch_size = self.agent_config.max_concurrent_searches
            batch_delay = self.agent_config.batch_delay_seconds
            
            logger.info(
                f"Starting batch controller with {len(research_queries)} queries, "
                f"batch size: {batch_size}, delay: {batch_delay}s"
            )
            
            # Create batches
            batches = []
            for i in range(0, len(research_queries), batch_size):
                batch = research_queries[i:i + batch_size]
                batches.append(batch)
            
            logger.info(f"Created {len(batches)} batches for sequential execution")
            
            # Store batch information in state for the routing function
            state.batch_info = {
                "current_batch_index": 0,
                "total_batches": len(batches),
                "batches": batches,
                "batch_size": batch_size,
                "batch_delay": batch_delay
            }
            
            # Update research context
            state.research_context.pending_searches = len(research_queries)
            state.research_context.completed_searches = 0
            
            return state.to_dict()
            
        except Exception as e:
            logger.error("Batch controller failed", error=e)
            state.update_metrics(error_count=state.workflow_metrics.error_count + 1)
            return state.to_dict()
    
    @safe_call("route_parallel_search", fallback=[])
    def route_to_parallel_search(self, state_dict: Dict[str, Any]) -> List[Send]:
        """Route queries to parallel search execution with batch processing."""
        state = RefactoredResearchAgentState.from_dict(state_dict)
        
        # Check if we have batch info from the batch controller
        if not state.batch_info:
            logger.error("No batch info found in state")
            return []
        
        batch_info = state.batch_info
        batches = batch_info["batches"]
        batch_delay = batch_info["batch_delay"]
        
        logger.info(f"Processing {len(batches)} batches sequentially")
        
        # Process all batches (delays will be handled in parallel nodes)
        all_sends = []
        for batch_index, batch in enumerate(batches):
            logger.info(
                f"Creating sends for batch {batch_index + 1}/{len(batches)} "
                f"with {len(batch)} queries"
            )
            
            # Create Send objects for current batch
            for query_index, research_query in enumerate(batch):
                global_index = batch_index * self.agent_config.max_concurrent_searches + query_index
                search_id = f"search_{global_index}_{research_query.query_id}"
                
                # Create search task state
                search_task = SearchTaskState(
                    search_id=search_id,
                    query=research_query.query,
                    query_index=global_index,
                    status="pending"
                )
                
                # Store in state (this will be passed to each parallel node)
                state.search_tasks[search_id] = search_task
                
                # Create Send for parallel execution with batch delay info
                send_data = {
                    "search_id": search_id,
                    "query": research_query.query,
                    "query_index": global_index,
                    "batch_index": batch_index,
                    "url_resolver": state.url_resolver,
                    "search_tasks": state.search_tasks
                }
                
                all_sends.append(Send("parallel_web_search", send_data))
        
        logger.info(f"Created {len(all_sends)} total send objects across {len(batches)} batches")
        return all_sends
    
    @retry("parallel_web_search", "search")
    def parallel_web_search_node(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual web search with rate limiting."""
        search_id = state_dict.get("search_id")
        query = state_dict.get("query")
        query_index = state_dict.get("query_index", 0)
        batch_index = state_dict.get("batch_index", 0)
        url_resolver = state_dict.get("url_resolver", URLResolver())
        
        search_start_time = time.time()
        
        # Apply delay between searches for rate limiting
        # When max_concurrent_searches=1, each search needs spacing
        if batch_index > 0 or query_index > 0:
            delay = self.agent_config.batch_delay_seconds
            logger.info(f"Applying rate limit delay of {delay}s for search {search_id}")
            time.sleep(delay)
        
        # Acquire semaphore for rate limiting
        logger.debug(f"Acquiring semaphore for search {search_id}")
        with self.search_semaphore:
            logger.info(f"Starting rate-limited search {search_id} for query: {query}")
            
            try:
                # Determine which search provider to use
                search_provider = self.config_manager.get_search_provider()
                print(f"[WorkflowNodes] Search provider from config: {search_provider}")
                logger.info(f"WorkflowNodes: search provider configured as: {search_provider}")
                
                # Get the appropriate search tool based on configuration
                if search_provider == "brave":
                    print(f"[WorkflowNodes] Attempting to get Brave search tool")
                    search_tool = self.tool_registry.get_tool(ToolType.BRAVE_SEARCH)
                    tool_name = "brave"
                    if not search_tool:
                        print(f"[WorkflowNodes] Brave tool not available, falling back to Tavily")
                        logger.warning("Brave search tool not available, falling back to Tavily")
                        search_tool = self.tool_registry.get_tool(ToolType.TAVILY_SEARCH)
                        tool_name = "tavily"
                    else:
                        print(f"[WorkflowNodes] Successfully got Brave search tool")
                else:
                    print(f"[WorkflowNodes] Attempting to get Tavily search tool")
                    search_tool = self.tool_registry.get_tool(ToolType.TAVILY_SEARCH)
                    tool_name = "tavily"
                    if not search_tool:
                        print(f"[WorkflowNodes] Tavily tool not available, falling back to Brave")
                        logger.warning("Tavily search tool not available, falling back to Brave")
                        search_tool = self.tool_registry.get_tool(ToolType.BRAVE_SEARCH)
                        tool_name = "brave"
                    else:
                        print(f"[WorkflowNodes] Successfully got Tavily search tool")
                
                if not search_tool:
                    logger.warning(f"No search tools available for search {search_id}")
                    return {
                        "parallel_search_results": [{
                            "search_id": search_id,
                            "query": query,
                            "status": "failed",
                            "error": "No search tools available",
                            "results": [],
                            "execution_time": time.time() - search_start_time
                        }]
                    }
                
                # Perform search with caching
                search_cache_key = f"search_{hash(query)}_{tool_name}"
                cached_search_results = global_cache_manager.get(search_cache_key, "search_results")
                
                if cached_search_results is not None:
                    raw_results = cached_search_results
                    logger.info(f"Using cached search results for query: {query}")
                else:
                    raw_results = search_tool.search(query, max_results=5)
                    # Cache for 30 minutes
                    global_cache_manager.set(search_cache_key, raw_results, "search_results", ttl=1800)
                
                # Extract structured results
                structured_results = content_extractor.extract_search_results(
                    raw_results,
                    SearchResultType.WEB
                )
                
                # Extract URLs for shortening
                urls = [result.url for result in structured_results if result.url]
                url_mapping = url_resolver.resolve_urls(urls, prefix_id=f"s{query_index}")
                
                # Apply URL shortening to results
                for result in structured_results:
                    if result.url and result.url in url_mapping:
                        result.metadata["original_url"] = result.url
                        result.url = url_mapping[result.url]
                
                execution_time = time.time() - search_start_time
                
                logger.info(
                    f"Completed rate-limited search {search_id} (batch {batch_index})",
                    query=query,
                    results_count=len(structured_results),
                    execution_time=execution_time
                )
                
                return {
                    "parallel_search_results": [{
                        "search_id": search_id,
                        "query": query,
                        "status": "completed",
                        "results": structured_results,
                        "execution_time": execution_time,
                        "url_mapping": url_mapping
                    }]
                }
                
            except Exception as e:
                execution_time = time.time() - search_start_time
                logger.error(
                    f"Rate-limited search {search_id} failed",
                    query=query,
                    error=e,
                    execution_time=execution_time,
                    batch_index=batch_index
                )
                
                return {
                    "parallel_search_results": [{
                        "search_id": search_id,
                        "query": query,
                        "status": "failed",
                        "error": str(e),
                        "results": [],
                        "execution_time": execution_time
                    }]
                }
    
    @safe_call("aggregate_search_results", fallback={})
    def aggregate_search_results_node(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from parallel searches with deduplication."""
        state = RefactoredResearchAgentState.from_dict(state_dict)
        state.current_node = WorkflowNodeType.WEB_RESEARCH
        
        try:
            # LangGraph automatically accumulates results from parallel nodes
            # Get accumulated parallel search results
            all_results = []
            successful_searches = 0
            total_search_time = 0.0
            search_times = []
            
            # Get search results from the annotated accumulator field
            search_results = state_dict.get("parallel_search_results", [])
            
            # If no results found, check if we have existing web results
            if not search_results and state.research_context.web_results:
                logger.info("Using existing web results from previous search")
                return state.to_dict()
            
            # Process search results
            for search_result in search_results:
                if isinstance(search_result, dict):
                    search_id = search_result.get("search_id", "unknown")
                    status = search_result.get("status", "unknown")
                    results = search_result.get("results", [])
                    execution_time = search_result.get("execution_time", 0.0)
                    
                    search_times.append(execution_time)
                    total_search_time += execution_time
                    
                    if status == "completed":
                        successful_searches += 1
                        if isinstance(results, list):
                            all_results.extend(results)
                        
                        logger.info(
                            f"Aggregated results from {search_id}",
                            status=status,
                            results_count=len(results) if isinstance(results, list) else 0,
                            execution_time=execution_time
                        )
                    else:
                        error = search_result.get("error", "Unknown error")
                        logger.warning(
                            f"Search {search_id} failed during aggregation",
                            error=error,
                            execution_time=execution_time
                        )
            
            # Apply semantic deduplication if available
            if all_results and self.deduplicator:
                try:
                    deduplicated_results, dedup_metrics = self.deduplicator.deduplicate_results(all_results)
                    
                    logger.info(
                        "Applied semantic deduplication",
                        original_count=dedup_metrics.original_count,
                        deduplicated_count=dedup_metrics.deduplicated_count,
                        reduction_percentage=dedup_metrics.reduction_percentage
                    )
                    
                    all_results = deduplicated_results
                    
                except Exception as dedup_error:
                    logger.warning(f"Deduplication failed, using original results: {dedup_error}")
            
            # Update research context with new results
            if all_results:
                state.research_context.web_results.extend(all_results)
                state.research_context.completed_searches = successful_searches
                
                # Update metrics
                state.update_metrics(
                    total_web_results=len(all_results),
                    parallel_search_count=len(search_times),
                    average_search_time=total_search_time / max(1, len(search_times)),
                    fastest_search_time=min(search_times) if search_times else 0.0,
                    slowest_search_time=max(search_times) if search_times else 0.0
                )
                
                logger.info(
                    f"Aggregated parallel search results",
                    total_results=len(all_results),
                    successful_searches=successful_searches,
                    total_searches=len(search_times),
                    average_time=state.workflow_metrics.average_search_time
                )
            else:
                logger.warning("No search results found to aggregate")
            
            return state.to_dict()
            
        except Exception as e:
            logger.error("Search result aggregation failed", error=e)
            state.update_metrics(error_count=state.workflow_metrics.error_count + 1)
            return state.to_dict()
    
    @retry("vector_research", "search")
    def vector_research_node(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Perform vector search on internal knowledge base."""
        state = RefactoredResearchAgentState.from_dict(state_dict)
        state.current_node = WorkflowNodeType.VECTOR_RESEARCH
        
        try:
            # Get vector search tool
            vector_tool = self.tool_registry.get_tool(ToolType.VECTOR_SEARCH)
            if not vector_tool:
                logger.info("Vector search tool not available, skipping")
                return state.to_dict()
            
            # Use original question for vector search
            user_question = state.research_context.original_question
            if not user_question:
                logger.warning("No original question for vector search")
                return state.to_dict()
            
            # Cache vector search results
            vector_cache_key = f"vector_{hash(user_question)}"
            cached_documents = global_cache_manager.get(vector_cache_key, "vector_search")
            
            if cached_documents is not None:
                documents = cached_documents
                logger.info("Using cached vector search results")
            else:
                documents = vector_tool.get_relevant_documents(user_question)
                # Cache for 2 hours
                global_cache_manager.set(vector_cache_key, documents, "vector_search", ttl=7200)
            
            # Convert documents to SearchResult objects
            vector_results = []
            for doc in documents:
                result = SearchResult(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "internal"),
                    title=doc.metadata.get("title"),
                    url=doc.metadata.get("url"),
                    score=doc.metadata.get("score", 0.0),
                    result_type=SearchResultType.VECTOR,
                    metadata=doc.metadata
                )
                vector_results.append(result)
            
            state.research_context.vector_results.extend(vector_results)
            state.update_metrics(total_vector_results=len(vector_results))
            
            logger.info(f"Completed vector research with {len(vector_results)} results")
            return state.to_dict()
        except Exception as e:
            logger.error("Vector research failed", error=e)
            state.update_metrics(error_count=state.workflow_metrics.error_count + 1)
            return state.to_dict()
    
    @retry("reflect", "llm_call")
    def reflect_node(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on research results to determine if more research is needed."""
        state = RefactoredResearchAgentState.from_dict(state_dict)
        state.current_node = WorkflowNodeType.REFLECTION
        
        try:
            # Create reflection prompt
            reflection_prompt = PromptManager.create_reflection_prompt(
                state.research_context,
                self.agent_config.max_research_loops
            )
            
            # Get reflection from LLM with caching
            reflection_cache_key = f"reflection_{hash(reflection_prompt)}"
            cached_reflection = global_cache_manager.get(reflection_cache_key, "reflection")
            
            if cached_reflection is not None:
                reflection_response = cached_reflection
                logger.info("Using cached reflection response")
            else:
                reflection_response = self.llm.invoke([
                    SystemMessage(content=PromptManager.get_reflection_system_prompt()),
                    HumanMessage(content=reflection_prompt)
                ])
                # Cache for 30 minutes
                global_cache_manager.set(reflection_cache_key, reflection_response, "reflection", ttl=1800)
            
            # Handle both string and list content from LLM response
            reflection_content = extract_text_from_llm_response(reflection_response.content)
            
            # Parse reflection response
            reflection_data = safe_json_loads(
                reflection_content,
                {"needs_more_research": False, "reflection": reflection_content}
            )
            
            needs_more = reflection_data.get("needs_more_research", False)
            reflection_text = reflection_data.get("reflection", reflection_content)
            
            # Update research context
            state.research_context.reflection = reflection_text
            state.research_context.research_loops += 1
            
            # Don't continue if we've reached max loops
            if state.research_context.research_loops >= self.agent_config.max_research_loops:
                needs_more = False
            
            # Store decision for conditional edge
            state_dict["needs_more_research"] = needs_more
            
            logger.info(
                f"Reflection completed",
                needs_more_research=needs_more,
                research_loop=state.research_context.research_loops,
                max_loops=self.agent_config.max_research_loops
            )
            
            return state.to_dict()
        except Exception as e:
            logger.error("Reflection failed", error=e)
            state.update_metrics(error_count=state.workflow_metrics.error_count + 1)
            # Default to not continue on error
            state_dict["needs_more_research"] = False
            return state.to_dict()
    
    @retry("synthesize_answer", "llm_call")
    def synthesize_answer_node(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final answer with citations from all research."""
        state = RefactoredResearchAgentState.from_dict(state_dict)
        state.current_node = WorkflowNodeType.SYNTHESIS
        
        try:
            # Create synthesis prompt
            synthesis_prompt = PromptManager.create_synthesis_prompt(state.research_context)
            
            # Use only invoke() - nodes are atomic in LangGraph
            synthesis_cache_key = f"synthesis_{hash(synthesis_prompt)}"
            cached_synthesis = global_cache_manager.get(synthesis_cache_key, "synthesis")
            
            if cached_synthesis is not None:
                synthesis_response = cached_synthesis
                logger.info("Using cached synthesis response")
            else:
                synthesis_response = self.llm.invoke([
                    SystemMessage(content=PromptManager.get_synthesis_system_prompt()),
                    HumanMessage(content=synthesis_prompt)
                ])
                # Cache for 30 minutes
                global_cache_manager.set(synthesis_cache_key, synthesis_response, "synthesis", ttl=1800)
            
            synthesis_content = extract_text_from_llm_response(synthesis_response.content)
            
            # Restore original URLs in the synthesis response
            content_with_urls = state.url_resolver.restore_urls_in_text(synthesis_content)
            
            # Validate tables for severe malformation patterns
            def validate_tables(content: str) -> dict:
                """Detect severe table malformation patterns."""
                issues = []
                lines = content.split('\n')
                
                for i, line in enumerate(lines):
                    trimmed = line.strip()
                    
                    # Check for headers-separators-data on single line
                    if '| --- |' in trimmed and trimmed.count('|') > 6:
                        # Likely has merged content
                        parts = trimmed.split('|')
                        has_separator = any('---' in p for p in parts)
                        has_content = any(p.strip() and '---' not in p for p in parts)
                        if has_separator and has_content:
                            issues.append(f"Line {i+1}: Headers/separators/data merged on single line")
                    
                    # Check for condensed separators
                    if '|---|' in trimmed:
                        issues.append(f"Line {i+1}: Condensed separator pattern |---|")
                    
                    # Check for data starting with separator
                    if trimmed.startswith('| --- | --- |') and len(trimmed) > 20:
                        issues.append(f"Line {i+1}: Data row starts with separator pattern")
                    
                    # Check for trailing separators on content
                    if trimmed.endswith('| --- |') and any(c.isalpha() for c in trimmed):
                        issues.append(f"Line {i+1}: Content row has trailing separator")
                
                return {
                    'has_issues': len(issues) > 0,
                    'issue_count': len(issues),
                    'issues': issues[:10]  # Limit to first 10 issues
                }
            
            # Validate tables before fixing
            validation = validate_tables(content_with_urls)
            if validation['has_issues']:
                logger.warning(
                    f"Detected {validation['issue_count']} table malformation issues",
                    sample_issues=validation['issues']
                )
            
            # Apply comprehensive table preprocessing
            try:
                from .core.table_preprocessor import TablePreprocessor
                
                logger.info("Applying comprehensive table preprocessing to synthesis content")
                
                # Initialize preprocessor
                table_preprocessor = TablePreprocessor()
                
                # Count issues before preprocessing
                initial_double_pipes = content_with_urls.count('||')
                initial_condensed_seps = len(re.findall(r'\|---\|', content_with_urls))
                
                # Apply comprehensive preprocessing
                preprocessed_content = table_preprocessor.preprocess_tables(content_with_urls)
                
                # Get statistics
                stats = table_preprocessor.get_stats()
                
                # Count remaining issues
                final_double_pipes = preprocessed_content.count('||')
                final_condensed_seps = len(re.findall(r'\|---\|', preprocessed_content))
                
                logger.info(
                    "Table preprocessing complete",
                    initial_double_pipes=initial_double_pipes,
                    final_double_pipes=final_double_pipes,
                    initial_condensed_seps=initial_condensed_seps,
                    final_condensed_seps=final_condensed_seps,
                    tables_fixed=stats['tables_fixed'],
                    total_fixes=stats['total_fixes'],
                    patterns_fixed=stats['patterns_fixed']
                )
                
                if final_double_pipes > 0:
                    logger.warning(f"Still {final_double_pipes} double pipes after preprocessing")
                    # Apply final cleanup
                    import re
                    preprocessed_content = re.sub(r'\|\|+', '|', preprocessed_content)
                
                final_content = preprocessed_content
                    
            except ImportError as e:
                logger.warning(f"Could not import table preprocessor: {e}, falling back to basic fixes")
                try:
                    from .core.markdown_utils import extract_and_fix_tables
                    final_content = extract_and_fix_tables(content_with_urls)
                except:
                    final_content = content_with_urls
            except Exception as e:
                logger.error(f"Error applying table preprocessing: {e}, using original content")
                final_content = content_with_urls
            
            # Extract citations
            citations = content_extractor.extract_citations(
                state.research_context.web_results + state.research_context.vector_results
            )
            state.research_context.citations = citations
            
            # Update final metrics
            execution_time = time.time() - state.start_time
            url_stats = state.url_resolver.get_stats()
            
            state.update_metrics(
                execution_time_seconds=execution_time,
                total_research_loops=state.research_context.research_loops,
                success_rate=1.0 - (state.workflow_metrics.error_count / max(1, state.research_context.research_loops))
            )
            
            # Apply one final table fix before storing the message - ULTIMATE PROTECTION
            import re
            absolutely_final_content = re.sub(r'\|\|+', '|', final_content)
            
            # Verify no double pipes remain
            remaining_double_pipes = absolutely_final_content.count('||')
            if remaining_double_pipes > 0:
                logger.warning(f"STILL {remaining_double_pipes} double pipes found even after final cleanup!")
                # Apply even more aggressive cleanup
                absolutely_final_content = re.sub(r'\|\|+', '|', absolutely_final_content)
            else:
                logger.info("Final verification: No double pipes in stored message content")
            
            # Add final response to messages with restored URLs
            final_message = AIMessage(content=absolutely_final_content)
            state.messages.append(final_message)
            
            logger.info(
                "Synthesis completed",
                response_length=len(final_content),
                citations_count=len(citations),
                execution_time=format_duration(execution_time),
                url_stats=url_stats
            )
            
            return state.to_dict()
        except Exception as e:
            logger.error("Synthesis failed", error=e)
            # Create error response
            error_message = f"I encountered an error while synthesizing the final answer: {str(e)}"
            state.messages.append(AIMessage(content=error_message))
            state.update_metrics(error_count=state.workflow_metrics.error_count + 1)
            return state.to_dict()
    
    def should_continue_research(self, state_dict: Dict[str, Any]) -> str:
        """Determine if more research is needed."""
        needs_more = state_dict.get("needs_more_research", False)
        return "continue" if needs_more else "finish"
    
    # Helper methods
    def _extract_user_question(self, messages) -> Optional[str]:
        """Extract the latest user question from messages."""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return message_converter.extract_text_content(msg.content)
        return None
    
    def _generate_search_queries(self, user_question: str, query_complexity: Optional[QueryComplexity] = None) -> List[str]:
        """Generate search queries for the user question using model manager."""
        # Use model manager to get appropriate model for query generation
        model_endpoint = "databricks-claude-3-7-sonnet"  # Default
        model_config = None
        if self.model_manager:
            model_config = self.model_manager.config.get_model_for_role(ModelRole.QUERY_GENERATION)
            model_endpoint = model_config.endpoint
        
        # Enhanced prompt that includes query complexity if available
        complexity_info = ""
        if query_complexity:
            complexity_info = f"""
Query Analysis:
- Complexity: {query_complexity.complexity_level}
- Intent: {query_complexity.intent_type}
- Entities: {', '.join(query_complexity.entities[:3]) if query_complexity.entities else 'None'}
- Requires recent data: {query_complexity.requires_recent_data}
- Expected sources: {query_complexity.expected_sources}

Generate queries that match this complexity and intent."""
        
        prompt = f"""Generate {self.agent_config.initial_query_count} diverse search queries for: {user_question}{complexity_info}

Return response as JSON: {{"queries": ["query1", "query2", "query3"]}}"""
        
        # Use model-specific LLM if model manager is available
        llm_to_use = self.llm
        if self.model_manager and model_config and model_endpoint != self.agent_config.llm_endpoint:
            try:
                llm_to_use = ChatDatabricks(
                    endpoint=model_config.endpoint,
                    temperature=getattr(model_config, 'temperature', self.agent_config.temperature),
                    max_tokens=getattr(model_config, 'max_tokens', self.agent_config.max_tokens)
                )
            except Exception as e:
                logger.warning(f"Failed to use specific model {model_endpoint}, using default: {e}")
        
        response = llm_to_use.invoke([
            SystemMessage(content=PromptManager.get_query_generation_system_prompt()),
            HumanMessage(content=prompt)
        ])
        
        # Handle both string and list content from LLM response
        content = extract_text_from_llm_response(response.content)
        
        # Parse queries from JSON
        queries_data = safe_json_loads(content, {"queries": []})
        queries = queries_data.get("queries", [])
        
        # Fallback parsing if JSON failed
        if not queries:
            lines = content.split('\n')
            queries = [
                line.strip('- ').strip()
                for line in lines
                if line.strip() and not line.startswith('#') and len(line.strip()) > 5
            ][:self.agent_config.initial_query_count]
        
        return queries[:self.agent_config.initial_query_count]