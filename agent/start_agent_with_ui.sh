#!/bin/bash
set -e

# Enhanced startup script for Deep Research Agent with automatic UI setup
# Handles Node.js dependencies, UI building, and server startup automatically

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
AGENT_PORT=8000
PID_FILE="/tmp/deep-research-agent.pid"
LOG_FILE="/tmp/deep-research-agent.log"
BACKGROUND_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --background|-b)
            BACKGROUND_MODE=true
            shift
            ;;
        --port|-p)
            AGENT_PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --background, -b    Run in background mode"
            echo "  --port, -p PORT     Specify port (default: 8000)"
            echo "  --help, -h          Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}✗ Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Helper functions
print_step() {
    echo -e "${BLUE}🔧 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# Show banner
echo -e "${CYAN}"
echo "🚀 Deep Research Agent - Enhanced Startup Script"
echo "==============================================="
echo -e "${NC}"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "Please run this script from the agent/ directory"
    echo "   cd agent && ./start_agent_with_ui.sh"
    exit 1
fi

# Stop any running instance first
print_step "Ensuring any previous agent instance is stopped..."
set +e
./stop_agent.sh >/dev/null 2>&1
STOP_STATUS=$?
set -e
if [ $STOP_STATUS -eq 0 ]; then
    print_success "Previous agent instance stopped"
else
    print_warning "stop_agent.sh reported issues stopping the previous instance"
fi

# Double-check for lingering process
if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
    print_error "Existing agent process (PID: $(cat $PID_FILE)) is still running"
    print_info "Please stop it manually and try again"
    exit 1
fi

# 1. Environment Checks
print_step "Checking system requirements..."

# Check uv
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed"
    echo "   Install from: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
print_success "uv found"

# Check Node.js
if ! command -v node &> /dev/null; then
    print_warning "Node.js not found"
    print_info "Installing Node.js via package manager..."

    if command -v brew &> /dev/null; then
        brew install node
    elif command -v apt &> /dev/null; then
        sudo apt update && sudo apt install -y nodejs npm
    elif command -v yum &> /dev/null; then
        sudo yum install -y nodejs npm
    else
        print_error "Could not install Node.js automatically"
        echo "   Please install Node.js manually: https://nodejs.org/"
        exit 1
    fi
fi
print_success "Node.js found ($(node --version))"

# Check npm
if ! command -v npm &> /dev/null; then
    print_error "npm not found. Please install npm"
    exit 1
fi
print_success "npm found ($(npm --version))"

# 2. UI Setup
print_step "Setting up UI dependencies and build..."

if [ -d "ui" ]; then
    pushd ui >/dev/null

    # Always clean previous build to ensure fresh compilation
    print_info "Cleaning previous UI build..."
    rm -rf static/ dist/ build/ 2>/dev/null || true
    print_success "Previous UI build cleaned"

    print_info "Installing UI dependencies..."
    npm install >/dev/null 2>&1 && print_success "UI dependencies installed" || {
        print_error "Failed to install UI dependencies"
        exit 1
    }

    print_info "Building UI from scratch (ensuring latest changes)..."
    if npm run build >/dev/null 2>&1; then
        print_success "UI built successfully (output: $(pwd)/static)"
    else
        print_error "UI build failed"
        exit 1
    fi

    popd >/dev/null

    # Ensure static directory exists and clean it first
    print_info "Preparing server static directory..."
    rm -rf static/* 2>/dev/null || true
    mkdir -p static

    # Copy built UI files to server static directory
    print_info "Copying fresh UI build to server static directory..."
    if cp -r ui/static/* static/ 2>/dev/null; then
        print_success "Fresh UI files copied to static/ directory"
    else
        print_warning "No UI build files found to copy (ui/static/* -> static/)"
    fi
else
    print_warning "No ui/ directory found, skipping UI setup"
fi

# 3. Python Setup
print_step "Installing Python dependencies..."
uv sync
print_success "Python dependencies installed"

# 4. Environment Setup
print_step "Setting up environment..."

# Check for .env.local
if [ -f ".env.local" ]; then
    print_success "Found .env.local file"
    # Show which Databricks profile will be used
    if grep -q "DATABRICKS_CONFIG_PROFILE" .env.local; then
        PROFILE=$(grep "DATABRICKS_CONFIG_PROFILE" .env.local | cut -d'=' -f2 | tr -d '"')
        print_info "Using Databricks profile: $PROFILE"
    fi
else
    print_warning "No .env.local file found"
    print_info "Create one based on .env.example for local configuration"
fi

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
print_success "PYTHONPATH configured"

# 5. Server Startup
print_step "Starting Deep Research Agent server..."

echo ""
echo -e "${CYAN}🎯 Server Configuration:${NC}"
echo "   • Agent: Multi-agent research system (5 agents)"
echo "   • UI: React interface $([ -d "ui/dist" ] || [ -d "ui/build" ] || [ -d "ui/static" ] && echo "✅ built" || echo "❌ not built")"
echo "   • Server: FastAPI with static file serving"
echo "   • Port: $AGENT_PORT"
echo "   • Environment: $([ -f ".env.local" ] && echo ".env.local loaded" || echo "system environment")"
echo ""

if [ "$BACKGROUND_MODE" = true ]; then
    print_info "Starting server in background mode..."
    echo "   • Logs: $LOG_FILE"
    echo "   • PID file: $PID_FILE"
    echo "   • Stop with: ./stop_agent.sh"
    echo ""

    # Start in background and save PID
    uv run deep-research-agent --port "$AGENT_PORT" --reload > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo $SERVER_PID > "$PID_FILE"

    # Wait a moment to check if server started successfully
    sleep 3
    if kill -0 $SERVER_PID 2>/dev/null; then
        print_success "Server started successfully in background (PID: $SERVER_PID)"
        echo ""
        echo -e "${GREEN}🌐 Access Points:${NC}"
        echo "   • UI: http://localhost:$AGENT_PORT"
        echo "   • API: http://localhost:$AGENT_PORT/invocations"
        echo "   • Health: http://localhost:$AGENT_PORT/health"
        echo "   • API Docs: http://localhost:$AGENT_PORT/docs"
        echo ""
        print_info "Check logs with: tail -f $LOG_FILE"
        print_info "Stop server with: ./stop_agent.sh"
    else
        print_error "Server failed to start. Check logs: $LOG_FILE"
        exit 1
    fi
else
    print_info "Starting server in foreground mode..."
    echo "   • Press Ctrl+C to stop"
    echo ""
    echo -e "${GREEN}🌐 Access Points:${NC}"
    echo "   • UI: http://localhost:$AGENT_PORT"
    echo "   • API: http://localhost:$AGENT_PORT/invocations"
    echo "   • Health: http://localhost:$AGENT_PORT/health"
    echo "   • API Docs: http://localhost:$AGENT_PORT/docs"
    echo ""

    # Save PID for potential cleanup
    echo $$ > "$PID_FILE"

    # Cleanup function
    cleanup() {
        print_info "Shutting down server..."
        rm -f "$PID_FILE"
        exit 0
    }
    trap cleanup SIGINT SIGTERM

    # Start server in foreground
    exec uv run deep-research-agent --port "$AGENT_PORT" --reload
fi
