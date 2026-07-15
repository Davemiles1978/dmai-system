"""Cloudflare R2 backup helper (PR P).

Creates a consistent snapshot of the persistent disk — every ``*.db`` in
DATA_PATH copied via the SQLite *online backup API* (not a raw file copy, so a
concurrently-written DB is captured cleanly), a plain copy of
``api_registry.json`` if present, and a per-table JSON dump of the attached
Postgres (``admin_api_keys`` plus any other public tables) when a DATABASE_URL
is configured. Everything is tarred into
``dmai-backup-YYYY-MM-DD-HHMMSS.tar.gz`` and uploaded to R2 over the boto3 S3
API. A rotation policy prunes old snapshots (7 daily + 4 weekly + 12 monthly).

The module is import-safe with no side effects: boto3 is imported lazily inside
``R2BackupClient`` so the app (and the offline test-suite) load without R2
credentials present.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import logging
import os
import shutil
import sqlite3
import tarfile
import tempfile
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("dmai.backup")

# ── Config ──────────────────────────────────────────────────────────────────
R2_BACKUP_PREFIX = "backups/"
ROTATION_DAILY_KEEP = 7      # keep the 7 most-recent daily snapshots
ROTATION_WEEKLY_KEEP = 4     # + 4 Sunday snapshots (weekly)
ROTATION_MONTHLY_KEEP = 12   # + 12 first-of-month snapshots (monthly)

# Postgres tables worth dumping first; anything else found in the public schema
# is appended after these so the important table is always captured.
_PG_PRIORITY_TABLES = ["admin_api_keys"]

_BACKUP_STEM = "dmai-backup-"
_BACKUP_TS_FMT = "%Y-%m-%d-%H%M%S"


# ── R2 client ─────────────────────────────────────────────────────────────
class R2BackupClient:
    """Thin boto3 S3 wrapper targeting a Cloudflare R2 bucket.

    Reads connection settings from the environment (R2_ENDPOINT_URL,
    R2_BUCKET_NAME, R2_REGION, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY). boto3 is
    imported lazily so importing this module never requires the dependency or
    credentials to be present.
    """

    def __init__(self, client=None, bucket: Optional[str] = None):
        self.bucket = bucket or os.environ.get("R2_BACKUP_BUCKET") \
            or os.environ.get("R2_BUCKET_NAME", "dmai-backups")
        if client is not None:
            self._client = client
            return
        import boto3  # lazy: keeps import-time dependency-free
        self._client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("R2_REGION", "auto"),
        )

    def upload_file(self, local_path: str, key: str) -> None:
        self._client.upload_file(local_path, self.bucket, key)

    def list_objects(self, prefix: str = R2_BACKUP_PREFIX) -> List[Dict[str, Any]]:
        """Return [{key, size, last_modified}] for every object under prefix."""
        out: List[Dict[str, Any]] = []
        token: Optional[str] = None
        while True:
            kw: Dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = self._client.list_objects_v2(**kw)
            for obj in resp.get("Contents", []):
                out.append({
                    "key": obj["Key"],
                    "size": obj.get("Size", 0),
                    "last_modified": obj.get("LastModified"),
                })
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
        return out

    def delete_object(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)


# ── Snapshot creation ───────────────────────────────────────────────────────
def _copy_sqlite_consistent(src_path: str, dst_path: str) -> None:
    """Copy a SQLite DB via the online backup API (safe while it's written)."""
    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    dst = sqlite3.connect(dst_path)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def _dump_postgres(db_url: str, out_dir: str) -> Dict[str, int]:
    """Dump each public Postgres table to <out_dir>/pg_<table>.json.

    Returns {table_name: row_count}. Uses the app's PGStorage so we share the
    connection pool and DSN handling; failures per-table are swallowed (logged)
    so one bad table never aborts the whole backup.
    """
    tables_dumped: Dict[str, int] = {}
    try:
        from components.pg_storage import PGStorage
    except Exception as e:  # pragma: no cover - import guarded
        logger.warning("postgres dump skipped, PGStorage import failed: %s", e)
        return tables_dumped

    try:
        pg = PGStorage()
        if not pg.is_available():
            logger.warning("postgres dump skipped, backend not available")
            return tables_dumped
    except Exception as e:
        logger.warning("postgres dump skipped, PGStorage init failed: %s", e)
        return tables_dumped

    # Discover public tables, priority ones first.
    try:
        rows = pg._exec(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' ORDER BY table_name",
            fetch="all",
        ) or []
        discovered = [r["table_name"] for r in rows]
    except Exception as e:
        logger.warning("postgres table discovery failed: %s", e)
        discovered = list(_PG_PRIORITY_TABLES)

    ordered = [t for t in _PG_PRIORITY_TABLES if t in discovered]
    ordered += [t for t in discovered if t not in _PG_PRIORITY_TABLES]
    if not ordered:
        ordered = list(_PG_PRIORITY_TABLES)

    for table in ordered:
        try:
            data = pg._exec(f'SELECT * FROM "{table}"', fetch="all") or []
            out_path = os.path.join(out_dir, f"pg_{table}.json")
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, default=str, indent=2)
            tables_dumped[table] = len(data)
        except Exception as e:
            logger.warning("postgres dump of %s failed: %s", table, e)
    return tables_dumped


def create_snapshot(data_path: str, db_url: Optional[str],
                    now: Optional[_dt.datetime] = None) -> Tuple[str, Dict[str, Any]]:
    """Build a backup tarball and return (tar_path, manifest).

    The tarball lands in a fresh temp dir; the caller is responsible for
    uploading and cleaning it up. The manifest records what was captured.
    """
    now = now or _dt.datetime.utcnow()
    ts = now.strftime(_BACKUP_TS_FMT)
    work = tempfile.mkdtemp(prefix="dmai-backup-")
    staging = os.path.join(work, "snapshot")
    os.makedirs(staging, exist_ok=True)

    manifest: Dict[str, Any] = {
        "created_utc": now.isoformat() + "Z",
        "sqlite_files": [],
        "extras": [],
        "postgres_tables": {},
    }

    # 1) SQLite DBs via the online backup API.
    for db_file in sorted(glob.glob(os.path.join(data_path, "*.db"))):
        name = os.path.basename(db_file)
        try:
            _copy_sqlite_consistent(db_file, os.path.join(staging, name))
            manifest["sqlite_files"].append(name)
        except Exception as e:
            logger.warning("sqlite backup of %s failed: %s", name, e)

    # 2) api_registry.json plain copy (if present).
    reg = os.path.join(data_path, "api_registry.json")
    if os.path.exists(reg):
        shutil.copy2(reg, os.path.join(staging, "api_registry.json"))
        manifest["extras"].append("api_registry.json")

    # 3) Postgres per-table JSON dump.
    if db_url:
        manifest["postgres_tables"] = _dump_postgres(db_url, staging)

    # 4) Write the manifest itself into the snapshot, then tar it up.
    with open(os.path.join(staging, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    tar_name = f"{_BACKUP_STEM}{ts}.tar.gz"
    tar_path = os.path.join(work, tar_name)
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(staging, arcname=os.path.splitext(os.path.splitext(tar_name)[0])[0])

    manifest["tar_name"] = tar_name
    manifest["size_bytes"] = os.path.getsize(tar_path)
    return tar_path, manifest


# ── Rotation ─────────────────────────────────────────────────────────────────
def _parse_backup_dt(key: str) -> Optional[_dt.datetime]:
    """Extract the timestamp from a backups/dmai-backup-<ts>.tar.gz key."""
    base = os.path.basename(key)
    if not base.startswith(_BACKUP_STEM):
        return None
    stem = base[len(_BACKUP_STEM):]
    for suffix in (".tar.gz", ".tgz"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    try:
        return _dt.datetime.strptime(stem, _BACKUP_TS_FMT)
    except ValueError:
        return None


def apply_rotation(client: R2BackupClient, prefix: str,
                   now: Optional[_dt.datetime] = None) -> Dict[str, Any]:
    """Prune old snapshots under prefix per the daily/weekly/monthly policy.

    Retention: the ``ROTATION_DAILY_KEEP`` most-recent snapshots, plus Sunday
    snapshots up to ``ROTATION_WEEKLY_KEEP`` distinct weeks, plus first-of-month
    snapshots up to ``ROTATION_MONTHLY_KEEP`` distinct months. Anything not in
    the union of those keep-sets is deleted.
    """
    now = now or _dt.datetime.utcnow()

    catalogue = []
    for obj in client.list_objects(prefix):
        dt = _parse_backup_dt(obj["key"])
        if dt is not None:
            catalogue.append((dt, obj["key"]))
    catalogue.sort(key=lambda t: t[0], reverse=True)  # newest first

    keep: set = set()

    # Daily: most-recent N regardless of weekday.
    for _dt_, key in catalogue[:ROTATION_DAILY_KEEP]:
        keep.add(key)

    # Weekly: newest Sunday snapshot per ISO week, up to N weeks.
    seen_weeks: set = set()
    for dt, key in catalogue:
        if dt.weekday() == 6:  # Sunday
            wk = (dt.isocalendar()[0], dt.isocalendar()[1])
            if wk not in seen_weeks:
                seen_weeks.add(wk)
                keep.add(key)
                if len(seen_weeks) >= ROTATION_WEEKLY_KEEP:
                    break

    # Monthly: newest first-of-month snapshot per month, up to N months.
    seen_months: set = set()
    for dt, key in catalogue:
        if dt.day == 1:
            mo = (dt.year, dt.month)
            if mo not in seen_months:
                seen_months.add(mo)
                keep.add(key)
                if len(seen_months) >= ROTATION_MONTHLY_KEEP:
                    break

    deleted: List[str] = []
    for _dt_, key in catalogue:
        if key not in keep:
            try:
                client.delete_object(key)
                deleted.append(key)
            except Exception as e:
                logger.warning("rotation delete of %s failed: %s", key, e)

    return {"deleted": deleted, "kept": len(keep), "total": len(catalogue)}
