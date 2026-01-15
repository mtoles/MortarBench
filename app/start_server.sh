#!/bin/bash

# Script to start the Flask server, killing any previous instance

PORT=5002
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_FILE="$SCRIPT_DIR/app.py"

# Function to kill existing server on the port
kill_existing_server() {
    echo "Checking for existing server on port $PORT..."
    
    # Find processes using the port
    PID=$(lsof -ti:$PORT 2>/dev/null)
    
    if [ ! -z "$PID" ]; then
        echo "Found existing server with PID $PID, killing it..."
        kill -9 $PID 2>/dev/null
        sleep 1
        
        # Verify it's dead
        if lsof -ti:$PORT >/dev/null 2>&1; then
            echo "Warning: Process still running, trying harder..."
            killall -9 -f "flask.*app.py" 2>/dev/null
            killall -9 -f "python.*app.py" 2>/dev/null
            sleep 1
        fi
    fi
    
    # Also check for processes by name
    PIDS=$(pgrep -f "app/app.py" 2>/dev/null)
    if [ ! -z "$PIDS" ]; then
        echo "Found additional Flask processes, killing them..."
        echo "$PIDS" | xargs kill -9 2>/dev/null
        sleep 1
    fi
}

# Kill any existing server
kill_existing_server

# Check if port is still in use
if lsof -ti:$PORT >/dev/null 2>&1; then
    echo "Error: Port $PORT is still in use. Please free it manually."
    exit 1
fi

# Change to the project root directory
cd "$SCRIPT_DIR/.." || exit 1

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start the server
echo "Starting Flask server on port $PORT..."
echo "Server will be accessible at http://localhost:$PORT"
echo "Press Ctrl+C to stop the server"
echo ""

python3 "$APP_FILE"

