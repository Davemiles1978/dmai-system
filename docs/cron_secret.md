# Cron Secret Auth

## Purpose

Scheduled tasks (crons) need to hit protected endpoints without embedding the
master password in the cron task-text, where it would sit in plaintext in the
scheduler config and logs. The `X-Cron-Secret` header authenticates scheduled
callers against a dedicated `CRON_SECRET` env var instead.

This path is **separate** from interactive/admin auth: the `/api/cron/*`
endpoints accept **only** the `X-Cron-Secret` header — never a JWT and never the
master password. The comparison is constant-time (`hmac.compare_digest`) and
**fails closed**: if `CRON_SECRET` is unset or empty, every `/api/cron/*` call
is rejected and a WARNING is logged once at startup.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/api/cron/status`                 | Trivial healthcheck for the auth path (`{"ok":true,"auth":"cron"}`) |
| POST | `/api/cron/integrity/run`          | Runs the knowledge integrity check in the background |
| POST | `/api/cron/providers/health-check` | Provider health diagnostic (active / pending-key / invalid counts) |
| GET  | `/api/cron/self-evolution/gaps`    | Cron-auth mirror of `/api/self-evolution/gaps` (`?fresh=1` re-scans) |

On auth failure each returns `401` with:

```json
{"error": "cron auth required", "hint": "set X-Cron-Secret header"}
```

Every successful cron-auth call is logged: `cron-auth call: <METHOD> <PATH>`.

## Setup (Render)

Generate a random secret:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Set it on the `dmai-web` service:

```bash
render env set CRON_SECRET "<random-32-char-string>"
```

## Verify

```bash
curl -X POST -H "X-Cron-Secret: $CRON_SECRET" \
  https://dmai-web.onrender.com/api/cron/status
```

Expected: `{"ok":true,"auth":"cron"}`. A `401` means `CRON_SECRET` is unset on
the service or the header value does not match.
