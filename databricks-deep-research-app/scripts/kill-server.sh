#!/bin/bash
# Kill any existing server running on port 8000
# Used before E2E tests to ensure a fresh server state

PORT=${1:-8000}

# Find and kill processes listening on the port
if command -v lsof &> /dev/null; then
    # macOS / Linux with lsof
    PIDS=$(lsof -ti tcp:$PORT 2>/dev/null)
    if [ -n "$PIDS" ]; then
        echo "Killing existing processes on port $PORT: $PIDS"
        echo "$PIDS" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
elif command -v ss &> /dev/null; then
    # Linux with ss
    PIDS=$(ss -tlnp "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u)
    if [ -n "$PIDS" ]; then
        echo "Killing existing processes on port $PORT: $PIDS"
        echo "$PIDS" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
elif command -v netstat &> /dev/null; then
    # Fallback to netstat
    PIDS=$(netstat -tlnp 2>/dev/null | grep ":$PORT " | awk '{print $7}' | cut -d'/' -f1 | sort -u)
    if [ -n "$PIDS" ]; then
        echo "Killing existing processes on port $PORT: $PIDS"
        echo "$PIDS" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
fi

# Also kill any uvicorn processes that might be orphaned
pkill -9 -f "uvicorn.*src.main:app.*$PORT" 2>/dev/null || true

# Verify port is free
if command -v lsof &> /dev/null; then
    if lsof -ti tcp:$PORT &>/dev/null; then
        echo "Warning: Port $PORT is still in use"
        exit 1
    fi
fi

echo "Port $PORT is free"
exit 0
