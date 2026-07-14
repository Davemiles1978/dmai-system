# PR L — purchase-approval gate + auto-checkout scaffold — discovery notes

## What PR L adds
A treasury-gated purchase-approval workflow. DMAI monitors the live procurement
top-1 shortlist (PR K) against the treasury balance (PR I) and, once
`balance >= 1.2 × capex`, emits an operator **purchase proposal** through three
notification channels. The operator approves (auto-debiting the treasury),
checks out manually, then marks it purchased (the ledger reconciles any price
delta). A retailer **auto-checkout adapter layer** is scaffolded but
**feature-flagged OFF** with no working implementation.

## Package layout (flat-file style, mirrors treasury_ledger.py)
```
components/purchase_gate/
  __init__.py            package doc; notes auto-checkout is OFF
  config.py              constants + defaults + confirm_token()
  schema.py              purchase_proposals + config_kv DDL
  purchase_ledger.py     PurchaseGateStore + treasury-wiring action fns
  checkout_adapter.py    ABC + 3 stub adapters (NOT IMPLEMENTED)
  notifier.py            tri-channel fanout (in-app / Slack / email)
  monitor.py             one check cycle + positive_pnl_streak_days()
  monitor_loop.py        idempotent-bootstrap + cadence-gate loop
```
Chose flat module functions (`approve_proposal`, `mark_purchased`, …) over a
`store.py`/`loop.py` split so the style matches `treasury_ledger.py`, and so
the admin endpoints can call the action functions directly with explicit DB
paths (which is how the endpoint tests drive them).

## State machine (enforced in code)
```
                 approve                  mark-purchased
   pending ───────────────▶ approved ────────────────────▶ purchased
      │                        │
      │ decline                │ cancel
      ▼                        ▼
   declined                 cancelled
```
`VALID_TRANSITIONS` rejects any other move (and unknown ids) with `ValueError`.

## Treasury wiring (accepted drift — read carefully)
The treasury ledger (PR I) has a **CHECK constraint** allowing only five kinds
(`trade_realised`, `bet_settled`, `infra_spend`, `manual_credit`,
`manual_debit`) and has **no `provenance` column**. The spec asked for an
`infra_spend_adjustment` kind and a `provenance` field. Neither can exist
without a treasury migration, which is out of scope for PR L.

**Decision (documented drift):**
- **approve**: `record_manual(kind='infra_spend', amount=-capex, description="purchase_proposal:<id> auto-debit <hw>")`.
- **cancel** (post-approve reversal): `record_manual(kind='manual_credit', amount=+capex, description="purchase_proposal_cancelled:<id> reverse auto-debit")`.
- **mark-purchased reconciliation**: if `actual != capex`, `record_manual(kind='infra_spend', amount=-(actual-capex), description="infra_spend_adjustment purchase_proposal:<id> actual X vs capex Y")`.
- **decline** (from pending): no treasury entry — nothing was ever debited.

So the "adjustment" semantics and provenance live in the entry **description**
(prefixed `infra_spend_adjustment` / `purchase_proposal[_cancelled]:<id>`), and
each proposal row links back to its treasury entries via the
`auto_debit_entry_id` / `cancel_credit_entry_id` / `reconcile_entry_id`
columns. This keeps the treasury schema untouched while remaining auditable.

## Notifications
Tri-channel, best-effort, log-and-continue (a failure on one channel never
blocks proposal creation); `send_new_proposal` returns the list of channels
that actually delivered, stored in `purchase_proposals.channels_notified`.
- **in-app** + **email** go through a Perplexity-harness `send_notification`
  bridge (`_pplx_send_notification`). No such Python symbol exists in-repo, so
  the bridge lazy-imports `pplx_tools` / `components.integrations.pplx` and
  returns `False` if neither resolves. Tests monkeypatch the per-channel
  helpers directly.
- **Slack** reuses the existing `components.monetisation.notifier.SlackNotifier`
  with `mask={"purchase"}` and category `"purchase"` (→ #dmaitalk).

## Positive-P&L streak (auto-checkout invariant #5)
`positive_pnl_streak_days()` counts **consecutive calendar days**, ending at
the most-recent recorded day, whose summed realised-P&L entries
(`kind IN ('trade_realised','bet_settled')`) net **> 0**. A day that is
missing, zero, or negative breaks the streak. Surfaced by the status endpoint.

## Auto-checkout scaffold — 6 invariants (all enforced/verified by tests)
1. `AUTO_CHECKOUT_ENABLED` defaults **False**.
2. `AUTO_CHECKOUT_DRY_RUN` defaults **True** even when enabled.
3. **No adapter implements `execute_checkout`** — every one raises
   `NotImplementedError`, and every `can_checkout` returns `(False, reason)`.
   There is **no live purchase path**.
4. `AUTO_CHECKOUT_MAX_GBP = 750.0` hard cap — proposals above it never attempt.
5. `AUTO_CHECKOUT_REQUIRE_STREAK_DAYS = 30` consecutive net-positive days.
6. Enabling requires `confirm_token = sha256("enable-auto-checkout-"+install_ts)`.

Config precedence: `config_kv` override → `DMAI_AUTO_CHECKOUT_*` env → module
constant. The admin POST endpoint returns **403 + the expected token** when the
token is missing/wrong, and applies changes only when it matches.

The `/admin/purchases` auto-checkout panel is **read-only by design** — there is
no in-page toggle. The banner is red (enabled+live), amber (enabled+dry-run),
or green (disabled), and the page shows the confirm token plus a paragraph
explaining invariant #3.

## Loop
`PurchaseGateMonitorLoop` mirrors `procurement/loop.py`: idempotent
`start_purchase_gate_loop()` bootstrap, monotonic cadence gate
(`RUN_INTERVAL_SECONDS = 1800`), `force_check()` bypass for tests/admin.
Bootstrapped inside `_start_background_services` next to the procurement loop.

## Tests (+37)
`test_purchase_gate_schema.py` (2), `test_purchase_gate_store.py` (6),
`test_checkout_adapter.py` (5), `test_purchase_gate_notifier.py` (3),
`test_purchase_gate_monitor.py` (5), `test_purchase_gate_auto_checkout.py` (6),
`test_purchase_gate_endpoints.py` (7). Full suite: **588 passed** (551 baseline
+ 37). The 32 `test_reasoning.py::TestUT22OutputChaining` failures are
pre-existing and unrelated (they fail identically in isolation on a clean tree).

## Style drift summary
- Treasury `infra_spend_adjustment` kind + `provenance` column requested by the
  spec do not exist in the PR I schema → encoded in entry description + linked
  via `*_entry_id` columns (see above). This is the only accepted drift.
