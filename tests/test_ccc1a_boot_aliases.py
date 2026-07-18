"""PR CCC-1a: verify boot-time import aliases resolve.

The dmai_core_complete module imports:

    from hmac_validator import validate_webhook_signature, require_webhook_hmac
    from chain_logger import ChainLogger, log_chain_step

Both imports had been failing since a refactor renamed the underlying
symbols, producing warnings on every boot and disabling the HMAC +
chain-audit paths. CCC-1a restored the historical names via aliases
(hmac) and a shim (chain_logger.log_chain_step).

These tests lock the aliases so future refactors can't silently break
the same boot path again.
"""
from __future__ import annotations


def test_hmac_validate_alias_matches_verify():
    """validate_webhook_signature must be the same callable as verify_..."""
    import hmac_validator as hv
    assert hasattr(hv, "verify_webhook_signature")
    assert hasattr(hv, "validate_webhook_signature")
    assert hv.validate_webhook_signature is hv.verify_webhook_signature


def test_hmac_require_alias_matches_underlying():
    """require_webhook_hmac must be the same callable as require_webhook_auth."""
    import hmac_validator as hv
    assert hasattr(hv, "require_webhook_auth")
    assert hasattr(hv, "require_webhook_hmac")
    assert hv.require_webhook_hmac is hv.require_webhook_auth


def test_chain_logger_exports_log_chain_step():
    """log_chain_step must be importable and callable."""
    from chain_logger import log_chain_step, ChainLogger
    assert callable(log_chain_step)
    assert isinstance(ChainLogger, type)


def test_log_chain_step_swallows_errors(tmp_path, monkeypatch):
    """Audit path must never raise, even if the underlying logger dies."""
    import chain_logger as cl
    # Point at a path we then break to force an internal error.
    monkeypatch.setattr(cl, "_default_logger", None)
    monkeypatch.chdir(tmp_path)
    # Should not raise even with weird inputs
    cl.log_chain_step("test_chain_1", "step_a", data={"k": "v"})
    cl.log_chain_step("test_chain_2", "step_b", data=None)
    cl.log_chain_step("test_chain_3", "step_c")  # data omitted


def test_boot_import_shape_matches_dmai_core_complete():
    """Sanity: the exact `from ... import ...` line from dmai_core_complete
    must resolve without ImportError."""
    from hmac_validator import validate_webhook_signature, require_webhook_hmac
    from chain_logger import ChainLogger, log_chain_step
    # Force use to sidestep 'unused' linter noise
    assert all([
        validate_webhook_signature is not None,
        require_webhook_hmac is not None,
        ChainLogger is not None,
        log_chain_step is not None,
    ])
