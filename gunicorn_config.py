"""
Gunicorn configuration for DMAI.
No --preload: module-level code runs inside each worker process (after fork),
so daemon threads started by _start_background_services() survive and persist.
No --max-requests: workers are never recycled, preserving all background threads.
"""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1
# Phase 12: 16 threads (was 8) for headroom. The hub-wide semaphore in
# AIIntegrationHub caps concurrent provider calls at 4 (AI_HUB_MAX_CONCURRENT),
# so HTTP requests always have 12+ threads available even under provider load.
threads = int(os.environ.get('GUNICORN_THREADS', '16'))
worker_class = "gthread"
timeout = 300
graceful_timeout = 60
keepalive = 5
preload_app = False

# Never recycle workers — would kill all background threads
max_requests = 0

accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True

def on_starting(server):
    server.log.info("DMAI gunicorn starting — 1 worker, %s threads, no recycling, ai-hub semaphore capped", threads)

def worker_exit(server, worker):
    server.log.warning("DMAI worker %s exited — gunicorn will respawn and restart background services", worker.pid)
