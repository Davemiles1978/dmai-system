#!/bin/bash
# Render.com start script with correct paths

cd /opt/render/project/src || exit 1

# Set Python path
export PYTHONPATH=/opt/render/project/src
export PATH=/opt/render/project/src/venv/bin:$PATH

# Load environment
if [ -f .env.render.fixed ]; then
    export $(cat .env.render.fixed | xargs)
fi

# Start the web UI
exec gunicorn dmai_web_ui:app --bind 0.0.0.0:$PORT --workers 4 --timeout 120
