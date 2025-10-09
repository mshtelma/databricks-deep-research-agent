#!/bin/bash

# Stop script for Deep Research Agent

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PID_FILE="/tmp/deep-research-agent.pid"
LOG_FILE="/tmp/deep-research-agent.log"
USE_FASTAPI=false
DEFAULT_PORT=8000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fastapi)
            USE_FASTAPI=true
            shift
            ;;
        --port|-p)
            DEFAULT_PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --fastapi           (Informational only - stops all servers regardless)"
            echo "  --port, -p PORT     Specify port to clean up (default: 8000)"
            echo "  --help, -h          Show this help"
            echo ""
            echo "Behavior:"
            echo "  This script ALWAYS stops ALL Deep Research Agent processes:"
            echo "  - MLflow/deep-research-agent processes"
            echo "  - FastAPI/uvicorn processes"
            echo "  - Any processes on the specified port"
            echo ""
            echo "  No need to specify --fastapi - everything is stopped automatically!"
            exit 0
            ;;
        *)
            echo -e "${RED}✗ Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Helper functions
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
    echo -e "${BLUE}ℹ️  $1${NC}"
}

echo -e "${BLUE}🛑 Stopping All Deep Research Agent Processes${NC}"
echo "=============================================="
echo -e "${BLUE}Mode: Comprehensive (stops MLflow + FastAPI + port cleanup)${NC}"
echo ""

# Function to stop all agent processes
stop_all_processes() {
    local stopped=false

    # Stop MLflow/deep-research-agent processes
    if pgrep -f "deep-research-agent" > /dev/null; then
        print_info "Found running deep-research-agent processes, attempting to stop..."
        pkill -f "deep-research-agent"
        sleep 2

        if pgrep -f "deep-research-agent" > /dev/null; then
            print_warning "Some processes still running, force killing..."
            pkill -9 -f "deep-research-agent"
        fi
        print_success "Stopped running deep-research-agent processes"
        stopped=true
    fi

    # Stop FastAPI/uvicorn processes
    if pgrep -f "uvicorn.*fastapi_server" > /dev/null; then
        print_info "Found running FastAPI (uvicorn) processes, attempting to stop..."
        pkill -f "uvicorn.*fastapi_server"
        sleep 2

        if pgrep -f "uvicorn.*fastapi_server" > /dev/null; then
            print_warning "Some processes still running, force killing..."
            pkill -9 -f "uvicorn.*fastapi_server"
        fi
        print_success "Stopped running FastAPI server processes"
        stopped=true
    fi

    if [ "$stopped" = false ]; then
        print_info "No deep-research-agent or FastAPI processes found running"
    fi
}

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    print_warning "No PID file found at $PID_FILE"
    print_info "Agent may not be running or may have been started manually"
    print_info "Checking for any running agent processes..."

    # Always stop all processes (ignore --fastapi flag when no PID file)
    stop_all_processes
    exit 0
fi

# Read PID from file
PID=$(cat "$PID_FILE")

# Check if process is actually running
if ! kill -0 "$PID" 2>/dev/null; then
    print_warning "Process with PID $PID is not running"
    rm -f "$PID_FILE"
    print_success "Cleaned up stale PID file"
    exit 0
fi

print_info "Stopping agent process (PID: $PID)..."

# Try graceful shutdown first
if kill "$PID" 2>/dev/null; then
    # Wait for graceful shutdown
    for i in {1..10}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            break
        fi
        echo -n "."
        sleep 1
    done
    echo ""

    # Check if still running
    if kill -0 "$PID" 2>/dev/null; then
        print_warning "Process still running, force killing..."
        kill -9 "$PID" 2>/dev/null
        sleep 1
    fi
fi

# Verify process is stopped
if ! kill -0 "$PID" 2>/dev/null; then
    print_success "Agent stopped successfully (PID: $PID)"
    rm -f "$PID_FILE"
else
    print_error "Failed to stop agent process"
    exit 1
fi

# Additional comprehensive cleanup - stop ALL agent processes
# This ensures we catch any stray processes not tracked by the PID file
print_info "Performing comprehensive cleanup of all agent processes..."
stop_all_processes

# Additional cleanup - kill any remaining processes on the port
if lsof -ti:$DEFAULT_PORT >/dev/null 2>&1; then
    print_info "Cleaning up remaining processes on port $DEFAULT_PORT..."
    lsof -ti:$DEFAULT_PORT | xargs kill -9 2>/dev/null || true
    print_success "Cleaned up processes on port $DEFAULT_PORT"
fi

# Show log file location if it exists
if [ -f "$LOG_FILE" ]; then
    print_info "Logs available at: $LOG_FILE"
    echo "   View with: tail -f $LOG_FILE"
    echo "   Clear with: rm $LOG_FILE"
fi

echo ""
print_success "All Deep Research Agent processes stopped"