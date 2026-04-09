#!/bin/bash
# Gunicorn start with dynamic port

# Find free port
find_free_port() {
    local port=5001
    while lsof -i :$port > /dev/null 2>&1; do
        port=$((port + 1))
    done
    echo $port
}

PORT=$(find_free_port)
export PORT

echo "Starting Gunicorn on port $PORT"
gunicorn --bind 0.0.0.0:$PORT \
         --workers 4 \
         --threads 2 \
         --timeout 120 \
         --access-logfile logs/access.log \
         --error-logfile logs/error.log \
         'dmai_core_complete:app'
