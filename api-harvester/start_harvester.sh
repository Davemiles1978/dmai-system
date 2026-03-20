#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python3 harvester.py --daemon --port 8081
