"""
Gunicorn configuration for DMAI.

Design rules (DO NOT change without reading docs/HANDOVER.md §3):
  * 1 worker, 8 gthread threads — all 8 background training loops run as
    daemon threads inside this single worker process.
  * preload_app = False — module-level code (which spawns the background
    services) must run AFTER fork, inside the worker, or the daemon threads
    die with the temporary master process.
  * max_requests = 0 — worker is NEVER recycled. Recycling kills every
    daemon thread (KnowledgeGraph, StageLearner, Kaizen, autonomous
    researcher, web learner, vocab ingest, graph evolution, periodic update).
  * timeout = 300, graceful_timeout = 60 — long enough for slow LLM calls
    without triggering a SIGKILL that would also kill the threads.
"""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1
threads = 8
worker_class = "gthread"
timeout = 300
graceful_timeout = 60
keepalive = 5

# Never recycle workers — recycling kills all 8 background training threads.
max_requests = 0
max_requests_jitter = 0

# Module-level code must run inside the worker (post-fork) so the daemon
# threads it spawns survive. Preloading would run it in the master process
# whose threads are discarded when workers are forked.
preload_app = False

accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True

def on_starting(server):
    server.log.info(
        "DMAI gunicorn starting — 1 worker, 8 gthread threads, timeout=300s, "
        "max_requests=0 (no recycling), preload_app=False (post-fork init)."
    )

def post_fork(server, worker):
    server.log.info(
        "DMAI worker %s forked — background training threads will spawn now.",
        worker.pid,
    )

def worker_exit(server, worker):
    server.log.warning(
        "DMAI worker %s exited — gunicorn will respawn and re-initialise "
        "background services.",
        worker.pid,
    )
