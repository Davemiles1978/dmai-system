"""DMAI local code-authoring (Band 1: template synthesis).

Generates working Python capability modules locally, without any
external LLM call, for the well-understood capability shapes:

    - utility            -> pure function over list values
    - configuration      -> dict merge / validate
    - data_structure     -> list transform / stats
    - trading            -> price-series compute
    - blockchain         -> payload-hash summariser
    - interface          -> request-echo / normaliser
    - research           -> query-echo bundler
    - integration        -> payload passthrough
    - composite          -> merge of a + b
    - frontier           -> seeded RNG scalar
    - diversity_nudge    -> seeded RNG scalar (labelled)
    - ai_provider_update -> release-note passthrough
    - concept            -> input-echo
    - monitor            -> healthy vs alerting metric samples
    - infrastructure     -> service/env/port topology digest
    - analyser           -> record-list distribution summary
    - training           -> sample-list stats + label balance
    - api_wrapper        -> request-envelope normaliser (no HTTP)
    - testing            -> assertion-case pass/fail runner (pure)

Every generated module:
    - is syntactically valid Python 3.11 (ast.parse in the local
      validator before returning)
    - exposes a `run(**kwargs)` function that accepts the happy_kwargs
      the materialiser will smoke-test with
    - returns a dict with at least {"ok": True, ...}
    - has zero external side effects (no I/O, no network, no sqlite
      writes) so the smoke test can run it safely

This module has no external dependencies and no imports beyond the
stdlib.
"""
from ._templates import (
    LOCAL_CAPABILITY_TYPES,
    can_template,
    generate_from_template,
    TemplateResult,
)

__all__ = [
    "LOCAL_CAPABILITY_TYPES",
    "can_template",
    "generate_from_template",
    "TemplateResult",
]
