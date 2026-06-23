# DMAI Ops — Environment Audit Trail

Version-controlled audit trail for all DMAI Render environment variables.

---

## Files

| Path | Purpose |
|---|---|
| `ops/env_audit.py` | Main audit script |
| `ops/snapshots/snapshot_YYYY-MM-DD.json` | Daily snapshots (values redacted) |
| `ops/snapshots/latest.json` | Symlink to most recent snapshot |
| `ops/snapshots/baseline.json` | Last approved state — diffs measured against this |
| `ops/snapshots/diff_YYYY-MM-DD.json` | Change report per run |
| `ops/RECOVERY_PLAN.md` | Auto-generated recovery checklist |

---

## Quickstart

### Run the audit (read-only)
```bash
RENDER_API_KEY=rnd_xxxx python3 ops/env_audit.py
```

### Run without a Render API key (uses local env fallback)
```bash
python3 ops/env_audit.py --dry-run
```

### After adding new keys — approve current state as baseline
```bash
RENDER_API_KEY=rnd_xxxx python3 ops/env_audit.py --approve
```

### Print the full recovery plan
```bash
python3 ops/env_audit.py --recover
```

### Dry run (no git commit)
```bash
python3 ops/env_audit.py --dry-run
```

---

## How Redaction Works

Values are **never stored**. Each variable is recorded as:

- `status`: `SET` or `UNSET`  
- `fingerprint`: `{first 2 chars}****[{sha1 prefix}]` — e.g. `sk****[a3f9c12b]`

This lets you detect whether a value **changed** (fingerprint differs) without ever exposing the real value.

---

## Change Detection

The diff engine compares each snapshot against `baseline.json` and flags:

| Category | Meaning |
|---|---|
| `added` | New variable present that wasn't in the baseline |
| `removed` | Variable in baseline no longer present |
| `changed` | Variable present in both but fingerprint or status differs |

Unapproved changes cause a **non-zero exit code** — suitable for CI/CD alerts.

---

## Required Render API Key

Get yours at: https://dashboard.render.com/u/settings#api-keys

Set it as `RENDER_API_KEY` in your local environment or as a Render env var on a separate ops service.

---

## Automated Weekly Snapshots

A scheduled task runs every Monday at 08:00 UTC via Perplexity Computer to:
1. Pull current Render env state
2. Diff against baseline
3. Commit snapshot to this repo
4. Send an in-app alert if unapproved changes are detected
