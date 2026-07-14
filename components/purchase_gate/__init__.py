"""Purchase-approval gate (PR L).

DMAI monitors her treasury balance against the live procurement shortlist
(PR K) and, when she can comfortably afford the top-ranked box, emits a
*purchase proposal* for the operator to approve. Approval auto-debits the
treasury; the operator then checks out manually at the retailer and marks
the proposal purchased, at which point the ledger is reconciled against the
real price paid.

An auto-checkout adapter layer is scaffolded but **feature-flagged OFF** and
has **no working retailer implementation** — see
:mod:`components.purchase_gate.checkout_adapter`.
"""
