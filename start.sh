#!/bin/bash
# Start evolution engine in background
python evolution_engine.py &

# Start web server in foreground (Render needs this)
gunicorn app_with_api:app --bind 0.0.0.0:10000
