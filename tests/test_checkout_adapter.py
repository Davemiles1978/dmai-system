"""Tests for the scaffolded auto-checkout adapters (PR L, PART B).

Core invariant: no adapter can complete a real purchase. Every
``execute_checkout`` raises ``NotImplementedError`` and every
``can_checkout`` returns False.
"""
from __future__ import annotations

import pytest

from components.purchase_gate.checkout_adapter import (
    ADAPTERS,
    IMPLEMENTED,
    AmazonUKAdapter,
    NeweggAdapter,
    ServeTheHomeAdapter,
    adapter_map,
)


def test_adapters_registered():
    assert set(ADAPTERS) == {"amazon_uk", "newegg_us", "serve_the_home"}


@pytest.mark.parametrize("cls", [AmazonUKAdapter, NeweggAdapter,
                                 ServeTheHomeAdapter])
def test_can_checkout_always_false(cls):
    ok, reason = cls().can_checkout({"hw_name": "x"})
    assert ok is False
    assert isinstance(reason, str) and reason


@pytest.mark.parametrize("cls", [AmazonUKAdapter, NeweggAdapter,
                                 ServeTheHomeAdapter])
def test_execute_checkout_raises(cls):
    with pytest.raises(NotImplementedError):
        cls().execute_checkout({"hw_name": "x"}, dry_run=False)


def test_amazon_normalise_url_extracts_asin():
    a = AmazonUKAdapter()
    assert a.normalise_url("https://www.amazon.co.uk/dp/B0ABCDE123/ref=x") \
        == "B0ABCDE123"
    # No ASIN → returns the original string unchanged.
    assert a.normalise_url("https://example.com/thing") \
        == "https://example.com/thing"


def test_adapter_map_all_unimplemented():
    m = adapter_map()
    assert set(m) == set(ADAPTERS)
    assert all(v["is_implemented"] is False for v in m.values())
    assert all(v is False for v in IMPLEMENTED.values())
