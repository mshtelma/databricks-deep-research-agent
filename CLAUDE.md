# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a **Databricks Deep Research Agent** - a comprehensive LangGraph-based research agent with a modern web UI. The project consists of two main components:

1. **Agent** (`agent/`): A sophisticated research agent built with LangGraph that performs multi-stage web research
2. **UI** (`ui/`): A modern React + FastAPI chat interface for the Deep Research Agent

## Architecture

### Agent Component (`agent/`)
- **Multi-Agent Architecture** with 5 specialized agents (Coordinator, Planner, Researcher, Fact Checker, Reporter)
- **Sophisticated LangGraph StateGraph** with conditional routing and feedback loops
- **Enhanced Planning System** with iterative refinement and quality assessment
- **Grounding Framework** for factual verification and claim validation
- **Professional Report Generation** with 7 distinct styles and citation formats
- **Modular core abstractions** in `deep_research_agent/core/` for error handling, caching, state management, and search providers
- **Multiple search providers** (Brave, Tavily) with automatic fallback
- **Databricks integration** via MLflow ResponsesAgent interface and Unity Catalog
- **Deployment system** for multi-environment deployment (dev/staging/prod)

### UI Component (`ui/`)
- **Modern full-stack Databricks App** with FastAPI backend and React TypeScript frontend
- **Real-time streaming** chat interface with Server-Sent Events (SSE)
- **Research visualization** showing live progress tracking and source citations
- **Dual authentication support** for workspace operations and agent endpoint access
- **Development-first workflow** with hot reloading and auto-generated TypeScript client

## Development Commands

### UI Development (Primary Interface)
```bash
# Setup environment and dependencies
cd ui && ./setup.sh

# Start development servers (REQUIRED - never run manually)
cd ui && nohup ./watch.sh > /tmp/databricks-app-watch.log 2>&1 &

# Stop development servers
kill $(cat /tmp/databricks-app-watch.pid) || pkill -f watch.sh

# Format code
cd ui && ./fix.sh

# Deploy UI to Databricks Apps
cd ui && ./deploy.sh

# Monitor deployment logs
cd ui && uv run python dba_logz.py <app-url> --duration 60
```

### Agent Development and Testing
```bash
cd agent

# Install dependencies
pip install -r requirements.txt -r requirements-test.txt

# Run tests
pytest tests/ -v

# Core agent tests
pytest tests/test_refactored_agent.py -v           # Single-agent core tests
pytest tests/test_multi_agent_integration.py -v    # Multi-agent integration tests

# Individual agent tests
pytest tests/test_coordinator_agent.py -v          # Coordinator agent tests
pytest tests/test_planner_agent.py -v              # Planner agent tests  
pytest tests/test_researcher_agent.py -v           # Researcher agent tests
pytest tests/test_fact_checker_agent.py -v         # Fact checker agent tests
pytest tests/test_reporter_agent.py -v             # Reporter agent tests

# Schema compliance and streaming
pytest tests/test_streaming_schema.py -v           # Schema compliance
pytest tests/test_agent_streaming_compliance.py -v # Multi-agent streaming tests

# Search and infrastructure
pytest tests/test_brave_search.py -v               # Search provider tests

# Deploy agent to Databricks
python databricks_deploy.py --env dev              # Development environment
python databricks_deploy.py --env staging          # Staging environment
```

## Key Development Patterns

### UI Development Rules
- **ALWAYS use `./watch.sh`** - Never run servers manually with uvicorn or npm
- **ALWAYS use `uv run`** for Python commands - Never use `python` directly
- **Use `databricks` CLI directly** - It's installed globally during setup
- **Monitor deployment logs immediately** after deployment with `dba_logz.py`
- **Validate with curl** after adding FastAPI endpoints before proceeding

### Authentication Architecture
The UI supports flexible dual authentication:
- **Workspace Auth**: For app deployment and workspace operations (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_CONFIG_PROFILE`)
- **Agent Endpoint Auth**: For accessing the research agent, can be same or different workspace (`AGENT_*` variables)

### Testing Approach
- **Agent**: Uses pytest with markers (`unit`, `integration`, `external`)
- **UI**: Uses FastAPI test client with real-time validation via Playwright
- **Schema Compliance**: Dedicated tests for MLflow ResponsesAgent interface compliance
- **Multi-Agent**: Specialized testing patterns for agent coordination and state management

### Multi-Agent Development Patterns

#### Agent Testing Strategy
**Critical Pattern**: Multi-agent tests require high-level mocking to avoid mock response ordering issues:

```python
# ✅ Correct: High-level mocking at graph level
with patch.object(enhanced_agent, 'graph') as mock_graph:
    mock_graph.ainvoke = AsyncMock(return_value={
        "final_report": "Report content...",
        "factuality_score": 0.85,
        "citations": [{"source": "...", "title": "..."}],
        "plan": MockFactory.create_mock_plan(num_steps=2),
        # ... complete response
    })
    
# ❌ Incorrect: Low-level LLM mocking causes response order issues
# This fails because agents use different LLM patterns:
# - Coordinator: Pattern matching (no LLM)  
# - Background Investigation: Search tools (no LLM)
# - Planner: LLM calls
with patch.object(ChatDatabricks, 'ainvoke') as mock_llm:
    mock_llm.side_effect = [response1, response2]  # Wrong order!
```

**Key Insight**: Only Planner, Researcher, Fact Checker, and Reporter use LLMs. Coordinator uses pattern matching, Background Investigation uses search tools.

#### State Management Patterns

The multi-agent system uses `EnhancedResearchState` with 100+ fields:

```python
from deep_research_agent.core.multi_agent_state import EnhancedResearchState, StateManager

# Initialize state with configuration
state = StateManager.initialize_state(
    research_topic="Your research question",
    config={
        "enable_iterative_planning": True,
        "max_plan_iterations": 3,
        "enable_grounding": True,
        "verification_level": "strict",
        "default_report_style": "academic",
        "enable_reflexion": True,
        "auto_accept_plan": True
    }
)

# Track agent handoffs
state = StateManager.record_handoff(
    state, 
    from_agent="planner", 
    to_agent="researcher",
    reason="Plan approved, beginning execution"
)

# Add observations during research
state = StateManager.add_observation(
    state, 
    "Found 15 relevant studies on quantum cryptography", 
    step=current_step
)
```

#### Configuration Patterns

Multi-agent configuration supports agent-specific settings:

```yaml
# agent_config_enhanced.yaml
agents:
  coordinator:
    enable_safety_filter: true
    
  planner:
    enable_deep_thinking: false
    max_iterations: 3
    quality_threshold: 0.7
    
  researcher:
    enable_reflexion: true
    max_steps_per_execution: 5
    
  fact_checker:
    verification_level: "moderate"  # strict, moderate, lenient
    enable_contradiction_detection: true
    
  reporter:
    default_style: "professional"
    enable_grounding_markers: true
    citation_style: "APA"

workflow:
  enable_background_investigation: true
  enable_human_feedback: false
  auto_accept_plan: true
```

## Project Structure

```
databricks-deep-research-agent/
├── agent/                                    # Multi-Agent Research System
│   ├── deep_research_agent/                 # Main agent package
│   │   ├── agents/                          # 5 Specialized Agents
│   │   │   ├── coordinator.py              # Request classification & routing
│   │   │   ├── planner.py                  # Research planning & refinement
│   │   │   ├── researcher.py               # Step execution & research
│   │   │   ├── fact_checker.py             # Grounding & factuality
│   │   │   └── reporter.py                 # Report generation & styling
│   │   ├── core/                            # Enhanced Core Abstractions
│   │   │   ├── multi_agent_state.py        # EnhancedResearchState (100+ fields)
│   │   │   ├── plan_models.py              # Planning system models
│   │   │   ├── grounding.py                # Factuality framework
│   │   │   ├── report_styles.py            # 7 report styles system
│   │   │   ├── event_emitter.py            # Progress tracking & events
│   │   │   ├── reasoning_tracer.py         # Reflexion & self-improvement
│   │   │   ├── table_preprocessor.py       # Structured data handling
│   │   │   ├── markdown_utils.py           # Report formatting utilities
│   │   │   └── redaction_utils.py          # Content filtering & safety
│   │   ├── tools_*.py                       # Search provider implementations
│   │   ├── workflow_nodes_enhanced.py       # Multi-agent LangGraph workflow
│   │   ├── enhanced_research_agent.py       # Main multi-agent orchestrator
│   │   ├── research_agent_refactored.py     # Legacy single-agent (deprecated)
│   │   ├── agent_config.yaml               # Single-agent configuration
│   │   └── agent_config_enhanced.yaml      # Multi-agent configuration
│   ├── docs/                                # Architecture Documentation
│   │   ├── ARCHITECTURE.md                 # Complete system architecture
│   │   ├── MULTI_AGENT_ARCHITECTURE.md     # Multi-agent deep dive
│   │   └── TESTING_ARCHITECTURE.md         # Testing patterns & strategies
│   ├── tests/                               # Comprehensive Test Suite
│   │   ├── test_multi_agent_integration.py # End-to-end multi-agent tests
│   │   ├── test_*_agent.py                 # Individual agent tests
│   │   ├── test_agent_streaming_compliance.py # Multi-agent streaming
│   │   └── test_core_components.py         # Core component tests
│   ├── deploy/                              # Deployment system
│   ├── requirements.txt                     # Agent dependencies
│   └── databricks_deploy.py                # Deployment script
│
├── ui/                                      # Chat UI (Databricks App)
│   ├── server/                              # FastAPI backend
│   │   ├── app.py                          # Main FastAPI application
│   │   ├── routers/                        # API endpoints (chat, user)
│   │   └── services/                       # Business logic
│   ├── client/                              # React TypeScript frontend
│   │   ├── src/components/chat/            # Chat interface components
│   │   ├── src/hooks/                      # Custom React hooks
│   │   └── src/fastapi_client/             # Auto-generated API client
│   ├── scripts/                             # Development automation
│   │   ├── watch.sh                        # Development server manager
│   │   ├── fix.sh                          # Code formatting
│   │   └── make_fastapi_client.py          # TypeScript client generator
│   ├── setup.sh                            # Environment setup
│   ├── pyproject.toml                      # Python dependencies
│   └── dba_logz.py                         # Deployment log monitor
```

## Important Implementation Notes

### UI Development Workflow
1. **Always start with `./watch.sh`** - This handles environment setup, authentication, and server management
2. **Use Playwright for validation** - Test UI changes in real browser during development
3. **Monitor logs continuously** - Check `/tmp/databricks-app-watch.log` for development server output
4. **Validate endpoints with curl** - Test FastAPI endpoints before moving to frontend integration

### Agent Integration
- The UI communicates with deployed agents via **Databricks serving endpoints**
- **Development mode** available with simulated agent responses for UI development
- **Flexible authentication** allows UI and agent to be in different Databricks workspaces

### Multi-Agent Workflow Architecture

The enhanced agent implements a sophisticated workflow with 5 specialized agents:

```
User Query → Coordinator (classify & route) → Background Investigation (gather context)
    ↓
Planner (create structured plan) ← [iterative refinement loop]
    ↓
Researcher (execute steps with context accumulation)
    ↓
Fact Checker (verify claims & grounding)
    ↓
Reporter (generate styled report) → Final Report with Grounding Markers
```

#### Agent Specialization:
- **Coordinator**: Pattern-based request classification (research/greeting/inappropriate)
- **Planner**: LLM-based structured planning with quality assessment and iterative refinement  
- **Researcher**: Step-by-step execution with observation accumulation and reflexion
- **Fact Checker**: Multi-layer claim verification with contradiction detection
- **Reporter**: Style-aware report generation with 7 professional formats

#### Key Features:
- **Conditional Routing**: State-based decisions between agents
- **Feedback Loops**: Planning refinement, research iteration, factuality revision
- **Context Accumulation**: Observations passed between research steps  
- **Quality Metrics**: Confidence scoring, factuality assessment, completeness tracking
- **Human Feedback Integration**: Optional plan review and editing interrupts

### Configuration Management
- **UI**: Environment variables in `.env.local`, managed by `./setup.sh`
- **Agent**: YAML-based configuration with environment overrides and MLflow secret integration
- **Deployment**: Multi-environment support with dedicated configuration files

### Technology Stack
- **Backend**: FastAPI + uvicorn, Databricks SDK, MLflow, LangGraph
- **Frontend**: React + TypeScript, Vite, shadcn/ui, Tailwind CSS, React Query
- **Development**: uv (Python), bun (Frontend), Playwright (Testing)
- **Deployment**: Databricks Apps, Unity Catalog, MLflow Model Registry

This architecture provides a complete multi-agent research solution with sophisticated planning, grounding, and professional reporting capabilities, supported by modern development tooling and production-ready deployment capabilities.