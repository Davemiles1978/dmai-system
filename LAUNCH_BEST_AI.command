#!/bin/bash
# LAUNCH BEST EVOLVED AI - Full UI with chat history, projects, archives

cd ~/Desktop/AI-Evolution-System
source venv/bin/activate

# Launch the full AI UI
python3 -c "
from pathlib import Path
import webbrowser
import http.server
import socketserver
import threading
import time

# Start server in background
PORT = 8888
Handler = http.server.SimpleHTTPRequestHandler

def start_server():
    with socketserver.TCPServer(('', PORT), Handler) as httpd:
        print(f'🚀 AI UI running at http://localhost:{PORT}')
        httpd.serve_forever()

threading.Thread(target=start_server, daemon=True).start()
time.sleep(2)
webbrowser.open(f'http://localhost:{PORT}/ai_ui.html')

# Keep running
while True:
    time.sleep(1)
"
