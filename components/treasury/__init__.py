"""DMAI Treasury — banked-revenue ledger and treasury balance.

PR I foundation for the self-hosting funding goal: track every
realised P&L event from ``trades_ledger`` (live-mode only) and
``bets_ledger`` (settled only), plus manual infra-spend entries,
into a single ledger keyed in GBP. The treasury_balance is the
running sum of everything from the ledger install timestamp
forward.
"""
