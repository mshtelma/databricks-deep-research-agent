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

echo -e "${BLUE}🛑 Stopping Deep Research Agent${NC}"
echo "================================"

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    print_warning "No PID file found at $PID_FILE"
    print_info "Agent may not be running or may have been started manually"

    # Try to find and kill any running deep-research-agent processes
    if pgrep -f "deep-research-agent" > /dev/null; then
        print_info "Found running deep-research-agent processes, attempting to stop..."
        pkill -f "deep-research-agent"
        sleep 2

        if pgrep -f "deep-research-agent" > /dev/null; then
            print_warning "Some processes still running, force killing..."
            pkill -9 -f "deep-research-agent"
        fi
        print_success "Stopped running deep-research-agent processes"
    else
        print_info "No deep-research-agent processes found running"
    fi
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
    print_success "Agent stopped successfully"
    rm -f "$PID_FILE"
else
    print_error "Failed to stop agent process"
    exit 1
fi

# Additional cleanup - kill any remaining processes on the port
if lsof -ti:8000 >/dev/null 2>&1; then
    print_info "Cleaning up remaining processes on port 8000..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
fi

# Show log file location if it exists
if [ -f "$LOG_FILE" ]; then
    print_info "Logs available at: $LOG_FILE"
    echo "   View with: tail -f $LOG_FILE"
    echo "   Clear with: rm $LOG_FILE"
fi

echo ""
print_success "Deep Research Agent stopped"