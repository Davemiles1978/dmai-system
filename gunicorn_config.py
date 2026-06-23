"""
Gunicorn configuration for DMAI.
Uses post_fork to ensure background services start AFTER the worker process
forks — this prevents the daemon-thread-dies-after-fork problem that occurs
with --preload.
"""
import os

# Bind / workers
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1
threads = 8
worker_class = "gthread"
timeout = 300

# Do NOT recycle workers — that kills background threads
# max_requests = 0  (default = never recycle)

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True

def post_fork(server, worker):
    """
    Called in the WORKER process after fork.
    This is the correct place to start background daemon threads —
    they survive here because we're in the worker, not the pre-fork parent.
    """
    try:
        # Import the already-loaded module (no re-import needed with preload)
        import dmai_core_complete as _dmai
        if not getattr(_dmai, "_background_services_started", False):
            _dmai._start_background_services()
            _dmai._background_services_started = True
            server.log.info("DMAI: background services started in worker %s", worker.pid)
        else:
            server.log.info("DMAI: background services already running in worker %s", worker.pid)
    except Exception as e:
        server.log.error("DMAI: post_fork background service startup failed: %s", e)

def on_starting(server):
    server.log.info("DMAI gunicorn starting — background services will start post-fork")

def worker_exit(server, worker):
    server.log.warning("DMAI worker %s exited — gunicorn will respawn", worker.pid)
