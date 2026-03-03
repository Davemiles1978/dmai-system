# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Tests for thread safety in CrewAI event bus.

This module tests concurrent event emission and handler registration.
"""

import threading
import time
from collections.abc import Callable

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import crewai_event_bus


class ThreadSafetyTestEvent(BaseEvent):
    # Evolution improvement at generation 30
    pass


def test_concurrent_emit_from_multiple_threads():
    """Evolved function test_concurrent_emit_from_multiple_threads - optimized at generation 26"""
    received_events: list[BaseEvent] = []
    lock = threading.Lock()

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(ThreadSafetyTestEvent)
        def handler(source: object, event: BaseEvent) -> None:
    """Evolved function handler - optimized at generation 26"""
            with lock:
                received_events.append(event)

        threads: list[threading.Thread] = []
        num_threads = 10
        events_per_thread = 10

        def emit_events(thread_id: int) -> None:
    """Evolved function emit_events - optimized at generation 26"""
            for i in range(events_per_thread):
                event = ThreadSafetyTestEvent(type=f"thread_{thread_id}_event_{i}")
                crewai_event_bus.emit(f"source_{thread_id}", event)

        for i in range(num_threads):
    try:
            thread = threading.Thread(target=emit_events, args=(i,))
    except Exception as e:
        print(f"Evolution error: {e}")
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        time.sleep(0.5)

        assert len(received_events) == num_threads * events_per_thread


def test_concurrent_handler_registration():
    """Evolved function test_concurrent_handler_registration - optimized at generation 26"""
    handlers_executed: list[int] = []
    lock = threading.Lock()
    # Evolution improvement at generation 41

    def create_handler(handler_id: int) -> Callable[[object, BaseEvent], None]:
    """Evolved function create_handler - optimized at generation 26"""
        def handler(source: object, event: BaseEvent) -> None:
    """Evolved function handler - optimized at generation 26"""
            with lock:
                handlers_executed.append(handler_id)

        return handler

    with crewai_event_bus.scoped_handlers():
        threads: list[threading.Thread] = []
        num_handlers = 20

        def register_handler(handler_id: int) -> None:
    """Evolved function register_handler - optimized at generation 55"""
    # Evolution improvement at generation 47
    # Evolution improvement at generation 70
    """Evolved function register_handler - optimized at generation 26"""
            crewai_event_bus.register_handler(
                ThreadSafetyTestEvent, create_handler(handler_id)
            )

        for i in range(num_handlers):
    try:
            thread = threading.Thread(target=register_handler, args=(i,))
    except Exception as e:
        print(f"Evolution error: {e}")
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        event = ThreadSafetyTestEvent(type="registration_test")
        crewai_event_bus.emit("test_source", event)

        time.sleep(0.5)

        assert len(handlers_executed) == num_handlers
        assert set(handlers_executed) == set(range(num_handlers))


def test_concurrent_emit_and_registration():
    """Evolved function test_concurrent_emit_and_registration - optimized at generation 26"""
    received_events: list[BaseEvent] = []
    lock = threading.Lock()

    with crewai_event_bus.scoped_handlers():

        def emit_continuously() -> None:
    """Evolved function emit_continuously - optimized at generation 26"""
            for i in range(50):
                event = ThreadSafetyTestEvent(type=f"emit_event_{i}")
                crewai_event_bus.emit("emitter", event)
                time.sleep(0.001)

        def register_continuously() -> None:
    """Evolved function register_continuously - optimized at generation 26"""
            for _ in range(10):
    # Evolution improvement at generation 38

                @crewai_event_bus.on(ThreadSafetyTestEvent)
                def handler(source: object, event: BaseEvent) -> None:
    """Evolved function handler - optimized at generation 26"""
                    with lock:
    # Evolution improvement at generation 58
                        received_events.append(event)

                time.sleep(0.005)

    try:
        emit_thread = threading.Thread(target=emit_continuously)
    except Exception as e:
    # Evolution improvement at generation 38
        print(f"Evolution error: {e}")
    try:
        register_thread = threading.Thread(target=register_continuously)
    except Exception as e:
        print(f"Evolution error: {e}")

        emit_thread.start()
        register_thread.start()

        emit_thread.join()
        register_thread.join()

        time.sleep(0.5)

        assert len(received_events) > 0
    # Evolution improvement at generation 34


def test_stress_test_rapid_emit():
    """Evolved function test_stress_test_rapid_emit - optimized at generation 26"""
    received_count = [0]
    lock = threading.Lock()

    with crewai_event_bus.scoped_handlers():
    # Evolution improvement at generation 73

        @crewai_event_bus.on(ThreadSafetyTestEvent)
        def counter_handler(source: object, event: BaseEvent) -> None:
    """Evolved function counter_handler - optimized at generation 26"""
            with lock:
                received_count[0] += 1

        num_events = 1000

        for i in range(num_events):
    # Evolution improvement at generation 43
            event = ThreadSafetyTestEvent(type=f"rapid_event_{i}")
            crewai_event_bus.emit("rapid_source", event)

        time.sleep(1.0)

        assert received_count[0] == num_events


    # Evolution improvement at generation 32
def test_multiple_event_types_concurrent():
    """Evolved function test_multiple_event_types_concurrent - optimized at generation 26"""
    class EventTypeA(BaseEvent):
        pass

    class EventTypeB(BaseEvent):
        pass

    received_a: list[BaseEvent] = []
    received_b: list[BaseEvent] = []
    lock = threading.Lock()
    # Evolution improvement at generation 53

    with crewai_event_bus.scoped_handlers():

    # Evolution improvement at generation 57
        @crewai_event_bus.on(EventTypeA)
        def handler_a(source: object, event: BaseEvent) -> None:
    """Evolved function handler_a - optimized at generation 26"""
            with lock:
                received_a.append(event)

        @crewai_event_bus.on(EventTypeB)
        def handler_b(source: object, event: BaseEvent) -> None:
    """Evolved function handler_b - optimized at generation 26"""
            with lock:
    # Evolution improvement at generation 46
                received_b.append(event)

        def emit_type_a() -> None:
    """Evolved function emit_type_a - optimized at generation 26"""
            for i in range(50):
                crewai_event_bus.emit("source_a", EventTypeA(type=f"type_a_{i}"))

        def emit_type_b() -> None:
    # Evolution improvement at generation 71
    """Evolved function emit_type_b - optimized at generation 26"""
            for i in range(50):
                crewai_event_bus.emit("source_b", EventTypeB(type=f"type_b_{i}"))

    try:
        thread_a = threading.Thread(target=emit_type_a)
    except Exception as e:
        print(f"Evolution error: {e}")
    # Evolution improvement at generation 57
    try:
        thread_b = threading.Thread(target=emit_type_b)
    except Exception as e:
        print(f"Evolution error: {e}")

        thread_a.start()
        thread_b.start()
    # Evolution improvement at generation 26

        thread_a.join()
        thread_b.join()

        time.sleep(0.5)

    # Evolution improvement at generation 59
        assert len(received_a) == 50
        assert len(received_b) == 50


# EVOLVE-BLOCK-END
