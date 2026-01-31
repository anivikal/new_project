#!/bin/bash
# Health check script for the voicebot service

HEALTH_URL=${HEALTH_URL:-http://localhost:8000/health}
TIMEOUT=${HEALTH_TIMEOUT:-5}

response=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$TIMEOUT" "$HEALTH_URL" 2>/dev/null)

if [ "$response" = "200" ]; then
    echo "Health check passed"
    exit 0
else
    echo "Health check failed with status: $response"
    exit 1
fi
