"""Tests for the tier-1 local code-authoring path.

Verifies every recognised capability_type can be synthesised locally,
executes cleanly with the same happy_kwargs the materialiser will use,
and returns the expected {'ok': True, 'source': 'local_template.*'}
contract.
"""
from __future__ import annotations

import ast

import pytest

from components.local_codegen import (
    LOCAL_CAPABILITY_TYPES,
    can_template,
    generate_from_template,
)
from components.capability_materialiser import _happy_kwargs_for


def _kwargs_for(cap_type: str) -> dict:
    """Drop db_path — templates are pure functions, no DB access."""
    kw = _happy_kwargs_for(cap_type)
    return {k: v for k, v in kw.items() if k != "db_path"}


class TestTemplateCoverage:
    def test_all_known_shapes_can_template(self):
        for ct in LOCAL_CAPABILITY_TYPES:
            assert can_template(ct), f"{ct} should be templatable"

    def test_unknown_shapes_cannot_template(self):
        assert not can_template("brand_new_thing")
        assert not can_template("")
        assert not can_template(None)


class TestTemplateSynthesis:
    @pytest.mark.parametrize("cap_type", sorted(LOCAL_CAPABILITY_TYPES))
    def test_every_shape_generates_valid_python(self, cap_type):
        r = generate_from_template(
            concept=f"test {cap_type}",
            insight="smoke insight",
            capability_type=cap_type,
            happy_kwargs=_kwargs_for(cap_type),
        )
        assert r.ok, f"{cap_type} failed: {r.reason}"
        # Must be syntactically valid.
        ast.parse(r.source)
        # Must expose run().
        assert "def run(**kwargs):" in r.source
        # Must have module docstring.
        assert r.source.lstrip().startswith('"""')

    @pytest.mark.parametrize("cap_type", sorted(LOCAL_CAPABILITY_TYPES))
    def test_every_shape_executes_cleanly(self, cap_type):
        r = generate_from_template(
            concept="x", insight="y",
            capability_type=cap_type,
            happy_kwargs=_kwargs_for(cap_type),
        )
        ns: dict = {}
        exec(r.source, ns)
        out = ns["run"](**_kwargs_for(cap_type))
        assert isinstance(out, dict)
        assert out["ok"] is True
        assert out["source"] == f"local_template.{cap_type}"


class TestPRAAA2NewShapes:
    """PR AAA-2: explicit smoke tests for the 6 new templated shapes.
    Each shape must (a) declare templatable, (b) generate valid Python,
    and (c) return the expected `source: local_template.<type>`
    contract when executed with realistic kwargs."""

    NEW_TYPES = ["monitor", "infrastructure", "analyser",
                 "training", "api_wrapper", "testing"]

    @pytest.mark.parametrize("cap_type", NEW_TYPES)
    def test_new_type_is_templatable(self, cap_type):
        assert can_template(cap_type) is True
        assert cap_type in LOCAL_CAPABILITY_TYPES

    def test_monitor_flags_threshold_breach(self):
        r = generate_from_template(
            concept="cpu monitor", insight="alert on cpu > 0.9",
            capability_type="monitor", happy_kwargs=_kwargs_for("monitor"),
        )
        assert r.ok
        ns = {}
        exec(r.source, ns)
        out = ns["run"](
            samples=[{"cpu": 0.5}, {"cpu": 0.95}, {"cpu": 0.3}],
            thresholds={"cpu": 0.9},
        )
        assert out["ok"] and out["total"] == 3
        assert out["alerting"] == 1
        assert out["healthy"] == 2
        assert out["breach_count"] == 1

    def test_infrastructure_produces_stable_digest(self):
        r = generate_from_template(
            concept="topology", insight="",
            capability_type="infrastructure",
            happy_kwargs=_kwargs_for("infrastructure"),
        )
        ns = {}; exec(r.source, ns)
        a = ns["run"](resources={"services": ["web", "db"]})
        b = ns["run"](resources={"services": ["db", "web"]})  # sorted ⇒ same
        assert a["topology_digest"] == b["topology_digest"]
        c = ns["run"](resources={"services": ["web"]})
        assert a["topology_digest"] != c["topology_digest"]

    def test_analyser_counts_distribution(self):
        r = generate_from_template(
            concept="kinds", insight="",
            capability_type="analyser", happy_kwargs=_kwargs_for("analyser"),
        )
        ns = {}; exec(r.source, ns)
        out = ns["run"](
            records=[{"kind": "a"}, {"kind": "a"}, {"kind": "b"}],
            group_by="kind",
        )
        assert out["count"] == 3
        assert out["distribution"] == {"a": 2, "b": 1}
        assert out["top"][0] == {"key": "a", "count": 2}

    def test_training_reports_label_balance(self):
        r = generate_from_template(
            concept="balance", insight="",
            capability_type="training", happy_kwargs=_kwargs_for("training"),
        )
        ns = {}; exec(r.source, ns)
        # Balanced
        out = ns["run"](samples=[{"label": "a"}, {"label": "b"}])
        assert out["balanced"] is True
        # Imbalanced
        out = ns["run"](samples=[{"label": "a"}, {"label": "a"},
                                  {"label": "b"}])
        assert out["balanced"] is False
        assert out["label_count"] == 2

    def test_api_wrapper_never_makes_http_call(self):
        """Critical: template must NOT import urllib/requests."""
        r = generate_from_template(
            concept="echo", insight="",
            capability_type="api_wrapper",
            happy_kwargs=_kwargs_for("api_wrapper"),
        )
        assert r.ok
        assert "urllib" not in r.source
        assert "import requests" not in r.source
        assert "urlopen" not in r.source
        ns = {}; exec(r.source, ns)
        out = ns["run"](request={"method": "post", "path": "/x",
                                  "params": {"a": 1}})
        assert out["method"] == "POST"
        assert out["param_count"] == 1

    def test_testing_runs_assertion_cases(self):
        r = generate_from_template(
            concept="assert", insight="",
            capability_type="testing", happy_kwargs=_kwargs_for("testing"),
        )
        ns = {}; exec(r.source, ns)
        out = ns["run"](target={"a": 1, "b": 2},
                        cases=[{"key": "a", "expected": 1},
                               {"key": "b", "expected": 99}])
        assert out["passed"] == 1
        assert out["failed"] == 1
        assert out["pass_rate"] == 0.5


class TestTemplateResilience:
    def test_utility_handles_empty_values(self):
        r = generate_from_template(
            concept="x", insight="y", capability_type="utility",
            happy_kwargs={"values": []},
        )
        ns: dict = {}; exec(r.source, ns)
        out = ns["run"](values=[])
        assert out["ok"] and out["count"] == 0

    def test_utility_handles_non_numeric(self):
        r = generate_from_template(
            concept="x", insight="y", capability_type="utility",
            happy_kwargs={"values": ["a", "b"]},
        )
        ns: dict = {}; exec(r.source, ns)
        out = ns["run"](values=["a", "b"])
        assert out["ok"] and out["count"] == 0

    def test_trading_zero_first_price_no_div_zero(self):
        r = generate_from_template(
            concept="x", insight="y", capability_type="trading",
            happy_kwargs={"prices": [0.0, 5.0, 10.0]},
        )
        ns: dict = {}; exec(r.source, ns)
        out = ns["run"](prices=[0.0, 5.0, 10.0])
        assert out["ok"] and out["return_pct"] == 0.0

    def test_configuration_rejects_non_dict(self):
        r = generate_from_template(
            concept="x", insight="y", capability_type="configuration",
            happy_kwargs={"config": {}},
        )
        ns: dict = {}; exec(r.source, ns)
        out = ns["run"](config="not a dict")
        assert out["ok"] is False

    def test_frontier_deterministic_under_seed(self):
        r = generate_from_template(
            concept="x", insight="y", capability_type="frontier",
            happy_kwargs={"seed": 42},
        )
        ns: dict = {}; exec(r.source, ns)
        a = ns["run"](seed=42)
        b = ns["run"](seed=42)
        assert a["value"] == b["value"] and a["picks"] == b["picks"]


class TestCodegenClientIntegration:
    """The request_code entrypoint should prefer templates when possible."""

    def test_templated_shape_bypasses_llm(self, monkeypatch):
        # If the LLM path is invoked, blow up loudly.
        from components.generated import _codegen_client as cc

        def _boom(*a, **kw):
            raise AssertionError(
                "external LLM was called for a templatable shape",
            )
        monkeypatch.setattr(cc, "_post_openrouter", _boom)

        att = cc.request_code(
            concept="local test",
            insight="never call LLM for utility",
            capability_type="utility",
            happy_kwargs={"values": [1, 2, 3]},
        )
        assert att.ok
        assert att.model.startswith("local_template.")
        assert "def run" in att.source
        assert att.usage.get("local") is True

    def test_retry_escalates_past_local_templates(self, monkeypatch):
        """On retry, skip the local path and go straight to LLM."""
        from components.generated import _codegen_client as cc

        calls = []

        def _stub(*a, **kw):
            calls.append(kw.get("max_tokens"))
            return {
                "__error__": "http_401",
                "http_status": 401,
                "body_snippet": "unauthorized",
            }
        # Ensure key is set so we exercise the LLM branch.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-testkey")
        monkeypatch.setattr(cc, "_post_openrouter", _stub)

        att = cc.request_code(
            concept="x", insight="y", capability_type="utility",
            happy_kwargs={"values": [1]},
            retry_reasons=["smoke test failed: ..."],
        )
        # LLM was consulted (calls > 0) and reason surfaces the error.
        assert calls, "expected external LLM to be called on retry"
        assert not att.ok
        assert "status_401" in (att.reason or "")

    def test_unknown_shape_falls_through_to_llm(self, monkeypatch):
        from components.generated import _codegen_client as cc

        called = {"n": 0}

        def _stub(*a, **kw):
            called["n"] += 1
            return {
                "__error__": "http_401",
                "http_status": 401,
                "body_snippet": "no key",
            }
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-testkey")
        monkeypatch.setattr(cc, "_post_openrouter", _stub)

        att = cc.request_code(
            concept="x", insight="y",
            capability_type="brand_new_shape_dmai_never_saw",
            happy_kwargs={"input": None},
        )
        assert called["n"] == 1
        assert not att.ok
