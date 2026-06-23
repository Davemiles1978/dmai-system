"""
Gunicorn configuration for DMAI.
No --preload: module-level code runs inside each worker process (after fork),
so daemon threads started by _start_background_services() survive and persist.
No --max-requests: workers are never recycled, preserving all background threads.
"""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1
threads = 8
worker_class = "gthread"
timeout = 300
keepalive = 5

# Never recycle workers — would kill all background threads
max_requests = 0

accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True

def on_starting(server):
    server.log.info("DMAI gunicorn starting — 1 worker, 8 threads, no recycling")

def worker_exit(server, worker):
    server.log.warning("DMAI worker %s exited — gunicorn will respawn and restart background services", worker.pid)
