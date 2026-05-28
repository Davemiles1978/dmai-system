"""Controlled thread management - prevents unlimited thread spawning"""
import threading
import queue
import time
import logging

logger = logging.getLogger(__name__)

class ControlledThreadManager:
    """Manages a fixed pool of threads instead of unlimited spawning"""
    
    _instance = None
    _max_workers = 2  # Render free tier limit
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._task_queue = queue.Queue()
        self._workers = []
        self._running = True
        self._start_workers()
        logger.info(f"🔧 Thread manager initialized with {self._max_workers} workers")
    
    def _start_workers(self):
        for i in range(self._max_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True, name=f"Worker-{i}")
            worker.start()
            self._workers.append(worker)
    
    def _worker_loop(self):
        while self._running:
            try:
                task = self._task_queue.get(timeout=1)
                try:
                    task()
                except Exception as e:
                    logger.error(f"Task error: {e}")
                self._task_queue.task_done()
            except queue.Empty:
                continue
    
    def submit(self, task):
        """Submit a task to be executed by a worker"""
        self._task_queue.put(task)
    
    def shutdown(self):
        self._running = False
        for worker in self._workers:
            worker.join(timeout=1)

# Global instance
thread_manager = ControlledThreadManager()
