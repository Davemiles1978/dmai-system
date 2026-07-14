"""Tests for components.generated._smoke_writer."""
from __future__ import annotations

from components.generated._smoke_writer import write_smoke_test


def test_writes_test_file(tmp_path):
    p = write_smoke_test(
        tests_dir=tmp_path,
        slug="my_cap",
        module_dotted="components.generated.staging.my_cap",
        happy_kwargs={"a": 1, "b": [1, 2, 3]},
    )
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    assert "def test_module_imports_cleanly" in body
    assert "def test_module_has_docstring" in body
    assert "def test_run_is_callable" in body
    assert "def test_happy_path_returns_within_2s" in body
    # kwargs preserved
    assert "'a': 1" in body
    assert "'b': [1, 2, 3]" in body


def test_falls_back_to_empty_kwargs_when_unserialisable(tmp_path):
    class NotJson:
        pass
    p = write_smoke_test(
        tests_dir=tmp_path,
        slug="fallback",
        module_dotted="x.y",
        happy_kwargs={"weird": NotJson()},
    )
    body = p.read_text(encoding="utf-8")
    # Fall-back is the empty dict literal
    assert "HAPPY_KWARGS = {}" in body


def test_module_dotted_baked_into_test(tmp_path):
    p = write_smoke_test(
        tests_dir=tmp_path,
        slug="abc",
        module_dotted="components.generated.staging.abc",
        happy_kwargs={},
    )
    body = p.read_text(encoding="utf-8")
    assert "components.generated.staging.abc" in body
