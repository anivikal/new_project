#!/bin/bash
# Battery Smart Voicebot - Startup Script

set -e

# Default values
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8000}
WORKERS=${API_WORKERS:-1}
LOG_LEVEL=${LOG_LEVEL:-info}

echo "Starting Battery Smart Voicebot..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo "Log Level: $LOG_LEVEL"

# Run the application
exec python -m uvicorn src.api.app:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL"
