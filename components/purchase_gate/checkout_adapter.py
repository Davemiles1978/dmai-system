"""Retailer auto-checkout adapters (PR L) — SCAFFOLD, NOT IMPLEMENTED.

None of these adapters can complete a real purchase. ``execute_checkout``
raises ``NotImplementedError`` everywhere and ``can_checkout`` returns
``False`` with a reason. This layer exists so the auto-checkout gate in
:mod:`components.purchase_gate.monitor` has a stable interface to call once a
real, PCI-reviewed implementation is written — see workspace/pr_l_notes.md.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple


class CheckoutAdapter(ABC):
    @abstractmethod
    def can_checkout(self, proposal: dict) -> Tuple[bool, str]:
        ...

    @abstractmethod
    def execute_checkout(self, proposal: dict, dry_run: bool = True) -> dict:
        ...

    @abstractmethod
    def normalise_url(self, url: str) -> str:
        ...


class AmazonUKAdapter(CheckoutAdapter):
    """Amazon UK checkout adapter — NOT IMPLEMENTED.

    A real implementation would need:
      - Authenticated session (Amazon MFA cookies or SP-API affiliate creds)
      - ASIN parsing from product URL (regex /dp/[A-Z0-9]{10}/)
      - Address book selection (default to primary shipping address)
      - Saved payment method ID (must be pre-registered by operator)
      - Order-review confirmation step with returned order_id
      - PCI compliance review before any live rollout
    """

    def can_checkout(self, proposal: dict) -> Tuple[bool, str]:
        return (False,
                "Amazon UK adapter not implemented — no live checkout path "
                "exists")

    def execute_checkout(self, proposal: dict, dry_run: bool = True) -> dict:
        raise NotImplementedError(
            "Amazon UK checkout adapter body not implemented")

    def normalise_url(self, url: str) -> str:
        import re
        m = re.search(r'/dp/([A-Z0-9]{10})', url or "")
        return m.group(1) if m else url


class NeweggAdapter(CheckoutAdapter):
    """Newegg checkout adapter — NOT IMPLEMENTED. Would need Newegg API
    access + saved payment."""

    def can_checkout(self, proposal: dict) -> Tuple[bool, str]:
        return (False, "Newegg adapter not implemented")

    def execute_checkout(self, proposal: dict, dry_run: bool = True) -> dict:
        raise NotImplementedError(
            "Newegg checkout adapter body not implemented")

    def normalise_url(self, url: str) -> str:
        return url


class ServeTheHomeAdapter(CheckoutAdapter):
    """STH is a review site, not a retailer — cannot ever checkout."""

    def can_checkout(self, proposal: dict) -> Tuple[bool, str]:
        return (False, "STH is a review site, not a retailer")

    def execute_checkout(self, proposal: dict, dry_run: bool = True) -> dict:
        raise NotImplementedError("STH is not a retailer")

    def normalise_url(self, url: str) -> str:
        return url


ADAPTERS = {
    "amazon_uk":      AmazonUKAdapter,
    "newegg_us":      NeweggAdapter,
    "serve_the_home": ServeTheHomeAdapter,
}

# No adapter implements a live checkout path yet (invariant #3).
IMPLEMENTED = {key: False for key in ADAPTERS}


def adapter_map() -> dict:
    """source → {class, is_implemented} — surfaced by the status endpoint."""
    return {
        key: {"class": klass.__name__, "is_implemented": IMPLEMENTED[key]}
        for key, klass in ADAPTERS.items()
    }


__all__ = [
    "CheckoutAdapter",
    "AmazonUKAdapter",
    "NeweggAdapter",
    "ServeTheHomeAdapter",
    "ADAPTERS",
    "IMPLEMENTED",
    "adapter_map",
]
