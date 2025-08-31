# Databricks App Template Development Guide

## Project Memory

This is a modern full-stack application template for Databricks Apps, featuring FastAPI backend with React TypeScript frontend and modern development tooling.

## Tech Stack

**Backend:**
- Python with `uv` for package management
- FastAPI for API framework
- Databricks SDK for workspace integration
- OpenAPI automatic client generation

**Frontend:**
- TypeScript with React
- Vite for fast development and hot reloading
- shadcn/ui components with Tailwind CSS
- React Query for API state management
- Bun for package management

## Development Workflow

### Package Management
- Use `uv add/remove` for Python dependencies, not manual edits to pyproject.toml
- Use `bun add/remove` for frontend dependencies, not manual package.json edits
- Always check if dependencies exist in the project before adding new ones

### Development Commands
- `./setup.sh` - Interactive environment setup and dependency installation
- `./watch.sh` - Start development servers with hot reloading (frontend:5173, backend:8000)
- `./fix.sh` - Format code (ruff for Python, prettier for TypeScript)
- `./deploy.sh` - Deploy to Databricks Apps

### 🚨 IMPORTANT: NEVER RUN THE SERVER MANUALLY 🚨

**ALWAYS use the watch script with nohup and logging:**

```bash
# Start development servers (REQUIRED COMMAND)
nohup ./watch.sh > /tmp/databricks-app-watch.log 2>&1 &

# Or for production mode
nohup ./watch.sh --prod > /tmp/databricks-app-watch.log 2>&1 &
```

**NEVER run uvicorn or the server directly!** Always use `./watch.sh` as it:
- Configures environment variables properly
- Starts both frontend and backend correctly
- Generates TypeScript client automatically
- Handles authentication setup
- Provides proper logging and error handling

### 🛑 HOW TO KILL THE DEVELOPMENT SERVERS

**To kill the development servers, use ONE of these methods:**

```bash
# Method 1: Using the PID file (preferred)
kill $(cat /tmp/databricks-app-watch.pid)

# Method 2: Kill by process name
pkill -f watch.sh

# Method 3: If you know the watch script PID directly
kill [PID]
```

**How it works (Updated Cleanup System):**
- **Recursive Process Tree Killing**: The cleanup function kills each tracked process (frontend, backend, watcher) and ALL their descendant processes recursively
- **Final Port Cleanup**: After process tree cleanup, any remaining processes on ports 5173 and 8000 are force-killed as a final safety measure
- **Complete Port Liberation**: Both ports 5173 (frontend) and 8000 (backend) are guaranteed to be freed
- **Handles Complex Process Chains**: Works with npm → bun → node process hierarchies that can't be killed with simple parent-child relationships

**Technical Details:**
- Uses `kill_tree()` function that recursively finds and kills all child processes
- Tracks specific PIDs: Frontend (shell wrapper), Backend (uvicorn), Watcher (watchmedo)  
- Final cleanup: `lsof -ti:5173 | xargs kill` and `lsof -ti:8000 | xargs kill`
- **Test Result**: Killing the watch script completely frees both ports with zero orphaned processes

### 🚨 PYTHON EXECUTION RULE 🚨

**NEVER run `python` directly - ALWAYS use `uv run`:**

```bash
# ✅ CORRECT - Always use uv run
uv run python script.py
uv run uvicorn server.app:app
uv run scripts/make_fastapi_client.py

# ❌ WRONG - Never use python directly
python script.py
uvicorn server.app:app
python scripts/make_fastapi_client.py
```

### 🚨 DATABRICKS CLI EXECUTION RULE 🚨

**The Databricks CLI is installed globally and run directly:**

```bash
# ✅ CORRECT - Use databricks CLI directly
databricks current-user me
databricks apps list
databricks workspace list /

# With environment variables when needed:
source .env.local && export DATABRICKS_HOST && export DATABRICKS_TOKEN && databricks current-user me
```

**Installation:**
- The CLI is installed globally during setup via ./setup.sh
- Uses official installation methods (brew on macOS, official installer on Linux)
- Always available in the PATH after installation

**🚨 DATABRICKS APPS COMPATIBILITY 🚨**

If the user runs into issues with the Databricks CLI not recognizing `apps` commands (e.g., "Error: unknown command 'apps'"), they need to upgrade to CLI version 0.265.0+:

```bash
# Check current version
databricks --version

# If version is < 0.265.0, offer to upgrade
# ONLY upgrade if user explicitly confirms - very important!
brew upgrade databricks
```

**When to upgrade:**
- User gets "unknown command 'apps'" errors
- CLI version shows < 0.265.0
- User explicitly requests CLI upgrade

**IMPORTANT**: Always ask for user confirmation before upgrading, as this affects their global CLI installation.

### Claude Natural Language Commands
Claude understands natural language commands for common development tasks:

**Development Lifecycle:**
- "start the devserver" → Runs `./watch.sh` in background with logging
- "kill the devserver" → Stops all background development processes
- "fix the code" → Runs `./fix.sh` to format Python and TypeScript code
- "deploy the app" → Runs `./deploy.sh` to deploy to Databricks Apps

**Development Tasks:**
- "add a new API endpoint" → Creates FastAPI routes with proper patterns
- "create a new React component" → Builds UI components using shadcn/ui
- "debug this error" → Analyzes logs and fixes issues
- "install [package]" → Adds dependencies using uv (Python) or bun (frontend)
- "generate the TypeScript client" → Regenerates API client from OpenAPI spec
- "open the UI in playwright" → Opens the frontend app in Playwright browser for testing
- "open app" → Gets app URL from `./app_status.sh` and opens it with `open {url}`

### Implementation Validation Workflow
**During implementation, ALWAYS:**
1. **Start development server first**: `nohup ./watch.sh > /tmp/databricks-app-watch.log 2>&1 &`
2. **Open app with Playwright** to see current state before changes
3. **After each implementation step:**
   - Check logs: `tail -f /tmp/databricks-app-watch.log`
   - Use Playwright to verify UI changes are working
   - Take snapshots to confirm features render correctly
   - Test user interactions and API calls
4. **🚨 CRITICAL: FastAPI Endpoint Verification**
   - **IMPORTANT: After adding ANY new FastAPI endpoint, MUST curl the endpoint to verify it works**
   - **NEVER move on to the next step until the endpoint is verified with curl**
   - **Example verification commands:**
     ```bash
     # Test GET endpoint
     curl -s http://localhost:8000/api/new-endpoint | jq
     
     # Test POST endpoint
     curl -X POST -H "Content-Type: application/json" -d '{"key":"value"}' http://localhost:8000/api/new-endpoint | jq
     ```
   - **Show the curl response to confirm the endpoint works correctly**
   - **If the endpoint fails, debug and fix it before proceeding**
5. **Install Playwright if needed**: `claude mcp add playwright npx '@playwright/mcp@latest'`
6. **Iterative validation**: Test each feature before moving to next step

**This ensures every implementation step is validated and working before proceeding.**

### Development Server
- **ALWAYS** run `./watch.sh` with nohup in background and log to file for debugging
- Watch script automatically runs in background and logs to `/tmp/databricks-app-watch.log`
- Frontend runs on http://localhost:5173
- Backend runs on http://localhost:8000
- API docs available at http://localhost:8000/docs
- Supports hot reloading for both frontend and backend
- Automatically generates TypeScript client from FastAPI OpenAPI spec
- **Check logs**: `tail -f /tmp/databricks-app-watch.log`
- **Stop processes**: `pkill -f "watch.sh"` or check PID file

### Code Quality
- Use `./fix.sh` for code formatting before commits
- Python: ruff for formatting and linting, ty for type checking
- TypeScript: prettier for formatting, ESLint for linting
- Type checking with TypeScript and ty (Python)

### API Development
- FastAPI automatically generates OpenAPI spec
- TypeScript client is auto-generated from OpenAPI spec
- Test endpoints with curl or FastAPI docs
- Check server logs after requests
- Verify response includes expected fields

### Databricks API Integration
- **ALWAYS** reference `docs/databricks_apis/` documentation when implementing Databricks features
- Use `docs/databricks_apis/databricks_sdk.md` for workspace, cluster, and SQL operations
- Use `docs/databricks_apis/mlflow_genai.md` for AI agent and LLM functionality
- Use `docs/databricks_apis/model_serving.md` for model serving endpoints and inference
- Use `docs/databricks_apis/workspace_apis.md` for file operations and directory management
- Follow the documented patterns and examples for proper API usage
- Check official documentation links in each API guide for latest updates

### Frontend Development
- Use shadcn/ui components for consistent UI
- Follow React Query patterns for API calls
- Use TypeScript strictly - no `any` types
- Import from auto-generated client: `import { apiClient } from '@/fastapi_client'`
- Client uses shadcn/ui components with proper TypeScript configuration
- shadcn components must be added with: npx shadcn@latest add <component-name>

### Testing Methodology
- Test API endpoints using FastAPI docs interface
- Use browser dev tools for frontend debugging
- Check network tab for API request/response inspection
- Verify console for any JavaScript errors

### Deployment
- Use `./deploy.sh` for Databricks Apps deployment
- Automatically builds frontend and generates requirements.txt
- Configures app.yaml with environment variables
- Verifies deployment through Databricks CLI

### 🚨 CRITICAL: Post-Deployment Monitoring Workflow 🚨

**ALWAYS follow this workflow after any deployment:**

1. **Immediately after deployment, MUST run log monitoring:**
   ```bash
   # Monitor deployment logs for 60 seconds to catch installation issues
   uv run python dba_logz.py <app-url> --duration 60
   
   # Or search specifically for uvicorn startup messages:
   uv run python dba_logz.py <app-url> --search "Application startup complete\|Uvicorn running" --duration 60
   ```

2. **Verify successful uvicorn startup:**
   - **REQUIRED**: Look for these specific uvicorn startup messages in logs:
     - `INFO: Application startup complete.`
     - `INFO: Uvicorn running`
   - **REQUIRED**: Look for any Python exceptions, import errors, or dependency issues in the logs
   - **If uvicorn startup messages not seen after reasonable time, run without search filter to see all logs and find errors:**
     ```bash
     # If no startup messages found, check all logs for errors
     uv run python dba_logz.py <app-url> --duration 30
     ```
   - **If ANY exceptions occur during installation or startup, MUST fix and redeploy**

3. **Exception handling protocol:**
   - **If Python exceptions found**: Analyze the error, fix the issue in code, and redeploy immediately
   - **If dependency issues found**: Update requirements, fix dependencies, and redeploy immediately
   - **If uvicorn fails to start**: Debug the FastAPI app, fix the issue, and redeploy immediately
   - **Never leave a deployment in a broken state**

4. **Deployment verification checklist:**
   - ✅ No Python exceptions during installation
   - ✅ All dependencies installed successfully  
   - ✅ Uvicorn server started and listening
   - ✅ FastAPI app accessible at the deployed URL
   - ✅ No critical errors in the log stream

5. **Test deployed endpoints with `dba_client.py`:**
   ```bash
   # Test core endpoints to verify app is functional
   uv run python dba_client.py <app-url> /health
   uv run python dba_client.py <app-url> /docs  
   uv run python dba_client.py <app-url> /api/user/me
   ```

**This monitoring workflow ensures deployments are successful and functional before moving on to other tasks.**

- **IMPORTANT**: Use `dba_logz.py` for real-time log streaming with search capabilities
- App logs are also available at: `https://<app-url>/logz` (visit in browser - requires OAuth authentication)

### Environment Configuration

#### Authentication Structure
The application now supports separate workspace and agent endpoint configuration:

**Databricks Workspace (for app deployment and workspace operations):**
- `DATABRICKS_HOST` - Workspace URL (e.g., https://your-workspace.cloud.databricks.com)
- `DATABRICKS_TOKEN` - Personal Access Token (optional, uses profile if not set)
- `DATABRICKS_CONFIG_PROFILE` - CLI profile name (used if no token provided)

**Agent Endpoint (can be same or different workspace):**
- `AGENT_DATABRICKS_HOST` - Agent workspace URL (optional, uses workspace host if not set)
- `AGENT_DATABRICKS_TOKEN` - Agent PAT (optional, uses profile if not set)
- `AGENT_CONFIG_PROFILE` - Agent CLI profile (optional, falls back to workspace auth)
- `AGENT_ENDPOINT_NAME` - Serving endpoint name (default: "deep-research-agent")

#### Authentication Logic
- If token is provided → uses PAT authentication with host + token
- If no token → uses CLI profile authentication
- Agent endpoint URL constructed as: `{AGENT_DATABRICKS_HOST}/serving-endpoints/{AGENT_ENDPOINT_NAME}/invocations`

#### Configuration Examples

**Same workspace for app and agent (PAT):**
```bash
DATABRICKS_HOST=https://workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi123...
AGENT_ENDPOINT_NAME=deep-research-agent
```

**Different workspaces (agent using profile):**
```bash
# App workspace
DATABRICKS_HOST=https://app-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi456...

# Agent endpoint
AGENT_CONFIG_PROFILE=production-profile
AGENT_ENDPOINT_NAME=deep-research-agent
```

**Mixed authentication:**
```bash
# App workspace using profile
DATABRICKS_CONFIG_PROFILE=dev-profile

# Agent using PAT
AGENT_DATABRICKS_HOST=https://ml-workspace.cloud.databricks.com
AGENT_DATABRICKS_TOKEN=dapi789...
AGENT_ENDPOINT_NAME=research-agent-v2
```

#### Setup Guidelines
- Use `.env.local` for local development configuration
- Never commit `.env.local` to version control
- Use `./setup.sh` to create and update environment configuration with interactive prompts
- The setup script will guide you through both workspace and agent endpoint configuration

### Debugging Tips
- Verify environment variables are set correctly
- Use FastAPI docs for API testing: http://localhost:8000/docs
- Check browser console for frontend errors
- Use React Query DevTools for API state inspection
- **Check watch logs**: `tail -f /tmp/databricks-app-watch.log` for all development server output
- **Check process status**: `ps aux | grep databricks-app` or check PID file at `/tmp/databricks-app-watch.pid`
- **Force stop**: `kill $(cat /tmp/databricks-app-watch.pid)` or `pkill -f watch.sh`

### Key Files
- `server/app.py` - FastAPI application entry point
- `server/routers/` - API endpoint routers
- `client/src/App.tsx` - React application entry point
- `client/src/pages/` - React page components
- `scripts/make_fastapi_client.py` - TypeScript client generator
- `pyproject.toml` - Python dependencies and project configuration
- `client/package.json` - Frontend dependencies and scripts
- `claude_scripts/` - Test scripts created by Claude for testing functionality

### API Documentation
- `docs/databricks_apis/` - Comprehensive API documentation for Databricks integrations
- `docs/databricks_apis/databricks_sdk.md` - Databricks SDK usage patterns
- `docs/databricks_apis/mlflow_genai.md` - MLflow GenAI for AI agents
- `docs/databricks_apis/model_serving.md` - Model serving endpoints and inference
- `docs/databricks_apis/workspace_apis.md` - Workspace file operations

### Documentation Files
- `docs/product.md` - Product requirements document (created during /dba workflow)
- `docs/design.md` - Technical design document (created during /dba workflow)
- These files are generated through iterative collaboration with the user during the /dba command

### Common Issues
- If TypeScript client is not found, run the client generation script manually
- If hot reload not working, restart `./watch.sh`
- If dependencies missing, run `./setup.sh` to reinstall

Remember: This is a development template focused on rapid iteration and modern tooling.