# Databricks Deployment Package

This package provides a complete deployment solution for the LangGraph Research Agent using Databricks Command Execution API for synchronous deployment with comprehensive testing and error reporting.

## Overview

The deployment system replaces the previous job-based approach with a Command Execution API-based solution that provides:

- **Synchronous execution** - No more polling, immediate feedback
- **Comprehensive testing** - Unit and integration tests run on Databricks
- **Clear error reporting** - Structured error messages optimized for Claude Code
- **Endpoint management** - Delete, create, and validate endpoints
- **Modular architecture** - Clean separation of concerns

## Quick Start

```bash
# Standard deployment with tests
python -m deploy --env dev

# Or use the backwards-compatible wrapper
python databricks_deploy.py --env dev
```

## Architecture

The deployment package consists of these core modules:

- **`__main__.py`** - Main orchestrator and entry point
- **`cli.py`** - Command-line interface and argument parsing
- **`command_executor.py`** - Command Execution API wrapper
- **`deployment_steps.py`** - Core deployment logic (extracted from notebook)
- **`test_runner.py`** - Test execution on Databricks
- **`workspace_manager.py`** - File sync and workspace operations
- **`endpoint_manager.py`** - Endpoint lifecycle management
- **`validator.py`** - Endpoint validation and testing
- **`error_reporter.py`** - Error formatting for Claude Code

## Configuration

Configuration is stored in `deploy/config.yaml` (moved from root `deploy_config.yaml`):

```yaml
environments:
  dev:
    profile: "e2-demo-west"
    workspace_path: "/Workspace/Users/user@company.com/LLM/langgraph-research-agent"
    
    command_execution:
      cluster_id: "0816-235658-5i3k9jfh"
      timeout: 300
      
    endpoint:
      name: "langgraph-research-agent-dev"
      workload_size: "Small"
      
    model:
      catalog: "main"
      schema: "msh"
      name: "langgraph_research_agent_dev"
```

## Usage Examples

### Basic Deployment

```bash
# Deploy to development environment
python -m deploy --env dev

# Deploy to production
python -m deploy --env prod
```

### Testing Options

```bash
# Skip tests for quick iteration
python -m deploy --env dev --skip-tests

# Run only tests, no deployment
python -m deploy --env dev --test-only

# Run specific test markers
python -m deploy --env dev --test-markers "unit"
python -m deploy --env dev --test-markers "integration"
python -m deploy --env dev --test-markers "not external"
```

### Endpoint Management

```bash
# Deploy with custom endpoint name
python -m deploy --env dev --endpoint-name "my-agent-v2"

# Delete existing endpoint before deployment
python -m deploy --env dev --delete-existing-endpoint

# List all endpoints
python -m deploy --list-endpoints

# Check specific endpoint status
python -m deploy --endpoint-status "my-endpoint"
```

### Workspace Management

```bash
# Clean workspace before deployment
python -m deploy --env dev --clean-workspace

# Sync files only
python -m deploy --env dev --sync-only
```

### Debugging and Development

```bash
# Dry run to see what would be done
python -m deploy --env dev --dry-run

# Verbose output for debugging
python -m deploy --env dev --verbose

# Skip validation for faster testing
python -m deploy --env dev --skip-validation
```

## Deployment Stages

The deployment pipeline consists of these stages:

1. **Workspace Preparation**
   - Optional workspace cleanup
   - File synchronization using databricks CLI

2. **Test Execution** (unless skipped)
   - Unit tests on Databricks cluster
   - Integration tests (if unit tests pass)
   - Structured error reporting for failures

3. **Endpoint Management** (if requested)
   - Delete existing endpoint
   - Wait for deletion to complete

4. **Model Deployment**
   - Install dependencies on cluster
   - Test agent setup and functionality
   - Log model to MLflow with Unity Catalog
   - Register model in Unity Catalog
   - Deploy to serving endpoint using Agent Framework
   - Wait for endpoint to become ready

5. **Validation** (unless skipped)
   - Run validation queries against endpoint
   - Verify response format and performance

## Error Reporting

The system provides detailed error reporting optimized for Claude Code:

- **Test Failures**: File paths, line numbers, and specific error messages
- **Execution Errors**: Full context with suggestions for common issues
- **Deployment Failures**: Stage-specific guidance and recovery steps

Example error output:
```
🔥 TEST FAILURES DETECTED
Failed: 2 | Passed: 5

❌ UNIT PHASE FAILED
   Failed: 2 | Passed: 3

🔍 DETAILED FAILURE ANALYSIS
1. 📍 test_agent_initialization
   File: tests/test_refactored_agent.py:45
   Phase: unit
   Error: ImportError: No module named 'tavily'...
   💡 Suggestion: Check package dependencies in requirements.txt

🚀 CLAUDE CODE ACTION ITEMS
1. Fix unit test failures - these are basic functionality issues
2. Focus on these files: tests/test_refactored_agent.py
3. Run failing tests locally for detailed debugging
```

## Key Improvements Over Job-Based System

1. **Synchronous Execution**: No polling or async complexity
2. **Immediate Feedback**: Direct command execution with real-time results  
3. **Better Error Handling**: Structured error messages with file:line references
4. **Comprehensive Testing**: Pytest runs directly on Databricks
5. **Modular Design**: Clean separation of concerns
6. **Claude Code Integration**: Error messages optimized for debugging

## Migration from Job-Based System

The old job-based system has been replaced. Key changes:

- **Configuration**: Moved from `deploy_config.yaml` to `deploy/config.yaml`
- **Execution**: Command Execution API instead of job execution
- **Notebook**: `log_and_deploy.py` replaced by `deployment_steps.py` 
- **Backwards Compatibility**: `databricks_deploy.py` now delegates to new system

## Troubleshooting

### Common Issues

1. **"No cluster_id found in configuration"**
   - Add `command_execution.cluster_id` to your environment config
   - Ensure the cluster ID exists and is accessible

2. **"Deploy package not found"**
   - Make sure you're running from the repository root
   - Verify all files in `deploy/` directory exist

3. **Authentication errors**
   - Check databricks CLI profile: `databricks auth login`
   - Verify profile matches config: `--profile <profile-name>`

4. **Test failures**
   - Run tests locally first: `pytest tests/`
   - Check test dependencies in `requirements-test.txt`

### Debug Mode

Enable verbose logging for detailed debugging:

```bash
python -m deploy --env dev --verbose --dry-run
```

This shows all API calls, authentication details, and execution steps.

## Development

To extend the deployment system:

1. Add new functionality to appropriate modules
2. Update CLI options in `cli.py` if needed
3. Add error handling in `error_reporter.py`
4. Test with dry run first: `--dry-run`

The modular design makes it easy to add new features or customize deployment behavior for specific needs.