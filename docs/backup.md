# Nightly R2 backup

Automated off-site backup of the persistent disk and the attached Postgres to
Cloudflare R2, with generational rotation.

## What is backed up

Each run produces a single `dmai-backup-YYYY-MM-DD-HHMMSS.tar.gz` containing:

- **Every `*.db` in `DATA_PATH`** — copied with the SQLite *online backup API*
  (`sqlite3.Connection.backup`), not a raw file copy, so a database that is
  being written during the backup is still captured consistently.
- **`api_registry.json`** — plain copy, if present.
- **Postgres** — when `DATABASE_URL` is set, every table in the public schema is
  dumped to `pg_<table>.json` (`admin_api_keys` first). JSON is used rather than
  `pg_dump` because the binary is not guaranteed to be on the image.
- **`manifest.json`** — records what was captured (files, tables, row counts,
  timestamp, size).

## Cadence & rotation

Runs are intended nightly (see below). Rotation keeps:

| Generation | Kept |
|------------|------|
| Daily      | 7 most-recent snapshots |
| Weekly     | 4 Sunday snapshots (distinct ISO weeks) |
| Monthly    | 12 first-of-month snapshots |

Anything outside the union of those keep-sets is deleted on each run. Tunable
via `ROTATION_DAILY_KEEP` / `ROTATION_WEEKLY_KEEP` / `ROTATION_MONTHLY_KEEP` in
`components/backup/r2_backup.py`.

## Endpoints

### `POST /api/cron/backup/run`

Cron-authenticated (`X-Cron-Secret` header — see [cron_secret.md](cron_secret.md)).
Creates the snapshot, uploads it under `backups/`, applies rotation, and returns:

```json
{
  "ok": true,
  "backup_key": "backups/dmai-backup-2026-07-15-014500.tar.gz",
  "size_bytes": 2489113,
  "sqlite_files": ["dmai.db", "dmai_knowledge.db"],
  "extras": ["api_registry.json"],
  "postgres_tables": ["admin_api_keys"],
  "postgres_rows": 16,
  "rotation": {"deleted": [], "kept": 1, "total": 1},
  "elapsed_sec": 3.14
}
```

### `POST /api/cron/backup/restore-list`

**Master-password gated** (`X-Master-Password`), *not* the cron secret — restore
is destructive, so it is kept on the interactive auth path. Returns all backups
newest-first; it never restores anything automatically:

```json
{"ok": true, "count": 3, "backups": [{"key": "...", "size_bytes": N, "last_modified": "..."}]}
```

Restore procedure (manual): download the chosen `backup_key` from R2, `tar xzf`
it, and copy the `*.db` files back into `DATA_PATH` / re-import the `pg_*.json`
dumps into Postgres. Take the app offline first.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `R2_ENDPOINT_URL` | R2 S3 API endpoint |
| `R2_BUCKET_NAME` | bucket (default `dmai-backups`) |
| `R2_REGION` | region (default `auto`) |
| `R2_ACCESS_KEY_ID` | R2 access key |
| `R2_SECRET_ACCESS_KEY` | R2 secret |
| `CRON_SECRET` | shared secret for `X-Cron-Secret` |
| `DATABASE_URL` | if set, Postgres is included in the dump |

## Scheduling

This PR does **not** create the scheduled task. Schedule it operator-side to hit
`POST /api/cron/backup/run` nightly — **01:45 UK time** recommended (off-peak) —
with the `X-Cron-Secret` header.

## Manual trigger

```bash
curl -X POST -H "X-Cron-Secret: $CRON_SECRET" \
  https://dmai-web.onrender.com/api/cron/backup/run
```
