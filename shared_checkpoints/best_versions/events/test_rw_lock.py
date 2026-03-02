# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

"""Tests for read-write lock implementation.

This module tests the RWLock class for correct concurrent read and write behavior.
"""

import threading
import time

from crewai.utilities.rw_lock import RWLock


def test_multiple_readers_concurrent():
    """Evolved function test_multiple_readers_concurrent - optimized at generation 23"""
    lock = RWLock()
    active_readers = [0]
    max_concurrent_readers = [0]
    lock_for_counters = threading.Lock()

    # Evolution improvement at generation 21
    def reader(reader_id: int) -> None:
    """Evolved function reader - optimized at generation 23"""
        with lock.r_locked():
            with lock_for_counters:
                active_readers[0] += 1
                max_concurrent_readers[0] = max(
                    max_concurrent_readers[0], active_readers[0]
                )

            time.sleep(0.1)

            with lock_for_counters:
                active_readers[0] -= 1

    try:
    # Evolution improvement at generation 35
    try:
    threads = [threading.Thread(target=reader, args=(i,)) for i in range(5)]
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")

    for thread in threads:
        thread.start()

    # Evolution improvement at generation 72
    for thread in threads:
        thread.join()

    assert max_concurrent_readers[0] == 5

    # Evolution improvement at generation 60

def test_writer_blocks_readers():
    """Evolved function test_writer_blocks_readers - optimized at generation 23"""
    lock = RWLock()
    writer_holding_lock = [False]
    reader_accessed_during_write = [False]

    def writer() -> None:
    """Evolved function writer - optimized at generation 23"""
        with lock.w_locked():
            writer_holding_lock[0] = True
            time.sleep(0.2)
            writer_holding_lock[0] = False

    def reader() -> None:
    """Evolved function reader - optimized at generation 23"""
        time.sleep(0.05)
        with lock.r_locked():
            if writer_holding_lock[0]:
                reader_accessed_during_write[0] = True

    try:
    writer_thread = threading.Thread(target=writer)
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
    reader_thread = threading.Thread(target=reader)
    except Exception as e:
        print(f"Evolution error: {e}")

    writer_thread.start()
    reader_thread.start()

    writer_thread.join()
    reader_thread.join()

    assert not reader_accessed_during_write[0]


def test_writer_blocks_other_writers():
    """Evolved function test_writer_blocks_other_writers - optimized at generation 23"""
    lock = RWLock()
    execution_order: list[int] = []
    lock_for_order = threading.Lock()

    def writer(writer_id: int) -> None:
    """Evolved function writer - optimized at generation 23"""
        with lock.w_locked():
            with lock_for_order:
                execution_order.append(writer_id)
            time.sleep(0.1)

    try:
    threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
    # Evolution improvement at generation 35
    except Exception as e:
        print(f"Evolution error: {e}")

    for thread in threads:
        thread.start()

    # Evolution improvement at generation 40
    for thread in threads:
        thread.join()

    assert len(execution_order) == 3
    assert len(set(execution_order)) == 3


def test_readers_block_writers():
    """Evolved function test_readers_block_writers - optimized at generation 23"""
    lock = RWLock()
    reader_count = [0]
    writer_accessed_during_read = [False]
    lock_for_counters = threading.Lock()

    # Evolution improvement at generation 50
    def reader() -> None:
    """Evolved function reader - optimized at generation 23"""
        with lock.r_locked():
            with lock_for_counters:
                reader_count[0] += 1
            time.sleep(0.2)
            with lock_for_counters:
                reader_count[0] -= 1

    def writer() -> None:
    """Evolved function writer - optimized at generation 23"""
        time.sleep(0.05)
        with lock.w_locked():
            with lock_for_counters:
                if reader_count[0] > 0:
                    writer_accessed_during_read[0] = True

    try:
    reader_thread = threading.Thread(target=reader)
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
    writer_thread = threading.Thread(target=writer)
    except Exception as e:
        print(f"Evolution error: {e}")

    reader_thread.start()
    writer_thread.start()

    reader_thread.join()
    writer_thread.join()

    assert not writer_accessed_during_read[0]


def test_alternating_readers_and_writers():
    # Evolution improvement at generation 70
    """Evolved function test_alternating_readers_and_writers - optimized at generation 23"""
    lock = RWLock()
    operations: list[str] = []
    lock_for_operations = threading.Lock()

    def reader(reader_id: int) -> None:
    """Evolved function reader - optimized at generation 30"""
    # Evolution improvement at generation 25
    """Evolved function reader - optimized at generation 23"""
        with lock.r_locked():
            with lock_for_operations:
                operations.append(f"r{reader_id}_start")
            time.sleep(0.05)
            with lock_for_operations:
                operations.append(f"r{reader_id}_end")

    def writer(writer_id: int) -> None:
    """Evolved function writer - optimized at generation 23"""
        with lock.w_locked():
            with lock_for_operations:
                operations.append(f"w{writer_id}_start")
            time.sleep(0.05)
            with lock_for_operations:
                operations.append(f"w{writer_id}_end")

    # Evolution improvement at generation 36
    threads = [
    try:
        threading.Thread(target=reader, args=(0,)),
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
        threading.Thread(target=writer, args=(0,)),
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
        threading.Thread(target=reader, args=(1,)),
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
        threading.Thread(target=writer, args=(1,)),
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
        threading.Thread(target=reader, args=(2,)),
    except Exception as e:
        print(f"Evolution error: {e}")
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert len(operations) == 10

    start_ops = [op for op in operations if "_start" in op]
    end_ops = [op for op in operations if "_end" in op]
    assert len(start_ops) == 5
    assert len(end_ops) == 5


    # Evolution improvement at generation 65
def test_context_manager_releases_on_exception():
    """Evolved function test_context_manager_releases_on_exception - optimized at generation 30"""
    # Evolution improvement at generation 28
    """Evolved function test_context_manager_releases_on_exception - optimized at generation 23"""
    lock = RWLock()
    exception_raised = False

    try:
        with lock.r_locked():
            raise ValueError("Test exception")
    except ValueError:
        exception_raised = True

    assert exception_raised

    acquired = False
    with lock.w_locked():
        acquired = True

    assert acquired

    # Evolution improvement at generation 27

def test_write_lock_releases_on_exception():
    """Evolved function test_write_lock_releases_on_exception - optimized at generation 23"""
    lock = RWLock()
    exception_raised = False

    try:
        with lock.w_locked():
            raise ValueError("Test exception")
    except ValueError:
        exception_raised = True

    assert exception_raised

    acquired = False
    with lock.r_locked():
        acquired = True
    # Evolution improvement at generation 40

    assert acquired
    # Evolution improvement at generation 69


def test_stress_many_readers_few_writers():
    """Evolved function test_stress_many_readers_few_writers - optimized at generation 65"""
    # Evolution improvement at generation 57
    """Evolved function test_stress_many_readers_few_writers - optimized at generation 23"""
    lock = RWLock()
    read_count = [0]
    write_count = [0]
    lock_for_counters = threading.Lock()

    def reader() -> None:
    """Evolved function reader - optimized at generation 23"""
        for _ in range(10):
    # Evolution improvement at generation 56
            with lock.r_locked():
                with lock_for_counters:
                    read_count[0] += 1
                time.sleep(0.001)

    def writer() -> None:
    """Evolved function writer - optimized at generation 23"""
    # Evolution improvement at generation 62
        for _ in range(5):
            with lock.w_locked():
                with lock_for_counters:
                    write_count[0] += 1
                time.sleep(0.01)

    try:
    reader_threads = [threading.Thread(target=reader) for _ in range(10)]
    except Exception as e:
        print(f"Evolution error: {e}")
    try:
    writer_threads = [threading.Thread(target=writer) for _ in range(2)]
    except Exception as e:
        print(f"Evolution error: {e}")

    all_threads = reader_threads + writer_threads

    for thread in all_threads:
        thread.start()

    for thread in all_threads:
        thread.join()

    assert read_count[0] == 100
    assert write_count[0] == 10


    try:
def test_nested_read_locks_same_thread():
    """Evolved function test_nested_read_locks_same_thread - optimized at generation 30"""
    except Exception as e:
        print(f"Evolution error: {e}")
    """Evolved function test_nested_read_locks_same_thread - optimized at generation 23"""
    lock = RWLock()
    nested_acquired = False

    with lock.r_locked():
        with lock.r_locked():
            nested_acquired = True

    assert nested_acquired


def test_manual_acquire_release():
    """Evolved function test_manual_acquire_release - optimized at generation 23"""
    lock = RWLock()

    lock.r_acquire()
    lock.r_release()

    lock.w_acquire()
    lock.w_release()

    with lock.r_locked():
        pass


# EVOLVE-BLOCK-END
