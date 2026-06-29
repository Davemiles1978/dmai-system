from components.repair_patterns import PATTERNS


def _get(name: str):
    for p in PATTERNS:
        if p.name == name:
            return p
    raise AssertionError(f"pattern not found: {name}")


def test_detect_startup_errors_swallowed():
    p = _get("startup_errors_swallowed")
    assert p.detect({"kind": "broken_routes", "route": "/api/x", "error": "503 Service Unavailable auth"})
    assert not p.detect({"kind": "broken_routes", "route": "/api/x", "error": "404 not found"})


def test_detect_safe_open_kdb_check_same_thread_kwarg():
    p = _get("safe_open_kdb_check_same_thread_kwarg")
    assert p.detect({"error": "TypeError: connect() got an unexpected keyword argument 'check_same_thread'"})


def test_detect_dead_thread_false_positive():
    p = _get("dead_thread_false_positive")
    assert p.detect({"kind": "dead_threads", "thread": "kaizen", "detail": "thread still running (false positive)"})


def test_detect_bytes_affinity_keyerror():
    p = _get("bytes_affinity_keyerror")
    assert p.detect({"error": "KeyError: TEXT bytes affinity"})


def test_detect_bytes_json_serialization_typeerror():
    p = _get("bytes_json_serialization_typeerror")
    assert p.detect({"error": "TypeError: Object of type bytes is not JSON serializable"})
