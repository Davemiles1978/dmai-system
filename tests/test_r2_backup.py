"""Tests for the nightly R2 backup (PR P).

Covers the backup helper (components/backup/r2_backup.py) and the two Flask
endpoints. No network and no real R2: boto3 is replaced by an in-memory fake S3
client, so everything runs offline. moto is listed in requirements-dev.txt for
parity but is not required here.

DATA_PATH is pointed at a temp dir *before* importing the app so boot side
effects stay isolated. Env/component overrides go through monkeypatch only, so
nothing leaks into the wider pytest session (the PR-N leak class of bug).
"""
from __future__ import annotations

import datetime as dt
import os
import re
import sqlite3
import tarfile
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="r2_backup_")
os.environ["DATA_PATH"] = _TMP

import dmai_core_complete  # noqa: E402
from dmai_core_complete import app  # noqa: E402
from components.backup import r2_backup as r2  # noqa: E402

_MASTER_PW = "test-master-pw"
_CRON_SECRET = "test-cron-secret"


class _FakeS3:
    """In-memory stand-in for a boto3 S3 client (only what we call)."""

    def __init__(self):
        self.objects: dict = {}       # key -> {size, last_modified}
        self.uploads: list = []       # (local_path, key)

    def upload_file(self, local_path, bucket, key):
        self.uploads.append((local_path, key))
        self.objects[key] = {
            "size": os.path.getsize(local_path),
            "last_modified": dt.datetime(2026, 7, 15, 1, 45),
        }

    def list_objects_v2(self, **kw):
        prefix = kw.get("Prefix", "")
        contents = [
            {"Key": k, "Size": v["size"], "LastModified": v["last_modified"]}
            for k, v in self.objects.items() if k.startswith(prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)


class _FakeR2:
    """Duck-typed R2BackupClient for rotation/list tests."""

    def __init__(self, catalogue):
        # catalogue: list of (key, size, last_modified)
        self._objs = [{"key": k, "size": s, "last_modified": m}
                      for k, s, m in catalogue]
        self.deleted: list = []

    def list_objects(self, prefix=r2.R2_BACKUP_PREFIX):
        return [o for o in self._objs if o["key"].startswith(prefix)]

    def delete_object(self, key):
        self.deleted.append(key)
        self._objs = [o for o in self._objs if o["key"] != key]


def _make_sqlite(path, rows):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t (id, v) VALUES (?,?)", rows)
    conn.commit()
    conn.close()


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    return app.test_client()


# ── 1. auth ───────────────────────────────────────────────────────────────
def test_backup_endpoint_requires_cron_auth(client, monkeypatch):
    monkeypatch.setenv("CRON_SECRET", _CRON_SECRET)
    resp = client.post("/api/cron/backup/run")  # no X-Cron-Secret header
    assert resp.status_code == 401


# ── 2. snapshot tarball ─────────────────────────────────────────────────────
def test_create_snapshot_produces_tarball(tmp_path):
    d = str(tmp_path)
    _make_sqlite(os.path.join(d, "dmai.db"), [(1, "a")])
    _make_sqlite(os.path.join(d, "other.db"), [(1, "b")])

    tar_path, manifest = r2.create_snapshot(d, None)
    try:
        assert os.path.exists(tar_path)
        assert set(manifest["sqlite_files"]) == {"dmai.db", "other.db"}
        with tarfile.open(tar_path) as tf:
            names = tf.getnames()
        assert any(n.endswith("dmai.db") for n in names)
        assert any(n.endswith("other.db") for n in names)
        assert any(n.endswith("manifest.json") for n in names)
    finally:
        import shutil
        shutil.rmtree(os.path.dirname(tar_path), ignore_errors=True)


# ── 3. online backup API used (consistency, not raw copy) ────────────────────
def test_sqlite_backup_api_used(tmp_path):
    d = str(tmp_path)
    src = os.path.join(d, "dmai.db")
    _make_sqlite(src, [(1, "committed")])

    # Open a second connection holding an UNCOMMITTED write. A raw file copy
    # could smear this; the online backup API captures only committed state.
    writer = sqlite3.connect(src)
    writer.execute("BEGIN")
    writer.execute("INSERT INTO t (id, v) VALUES (2, 'uncommitted')")

    tar_path, _ = r2.create_snapshot(d, None)
    try:
        with tarfile.open(tar_path) as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith("dmai.db"))
            extract_dir = tempfile.mkdtemp()
            tf.extract(member, extract_dir)
            copied = os.path.join(extract_dir, member.name)
        conn = sqlite3.connect(copied)
        count = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        conn.close()
        assert count == 1  # only the committed row
    finally:
        writer.rollback()
        writer.close()
        import shutil
        shutil.rmtree(os.path.dirname(tar_path), ignore_errors=True)


# ── 4. upload key format ─────────────────────────────────────────────────────
def test_r2_upload_uses_correct_key_format(client, monkeypatch, tmp_path):
    d = str(tmp_path)
    _make_sqlite(os.path.join(d, "dmai.db"), [(1, "a")])
    monkeypatch.setenv("DATA_PATH", d)
    monkeypatch.setattr(dmai_core_complete, "DATA_PATH", d)
    monkeypatch.setenv("CRON_SECRET", _CRON_SECRET)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    fake = _FakeS3()
    _RealClient = r2.R2BackupClient
    monkeypatch.setattr(r2, "R2BackupClient",
                        lambda *a, **k: _RealClient(client=fake, bucket="dmai-backups"))

    resp = client.post("/api/cron/backup/run",
                       headers={"X-Cron-Secret": _CRON_SECRET})
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body["ok"] is True
    assert len(fake.uploads) == 1
    _, key = fake.uploads[0]
    assert key == body["backup_key"]
    assert re.fullmatch(
        r"backups/dmai-backup-\d{4}-\d{2}-\d{2}-\d{6}\.tar\.gz", key)
    assert "dmai.db" in body["sqlite_files"]


# ── 5. rotation policy ───────────────────────────────────────────────────────
def test_rotation_keeps_recent_deletes_old():
    now = dt.datetime(2026, 7, 15, 12, 0, 0)
    catalogue = []
    for day in range(1, 16):  # Jul 1..15, one daily snapshot each
        ts = dt.datetime(2026, 7, day, 1, 45, 0)
        key = f"backups/dmai-backup-{ts.strftime('%Y-%m-%d-%H%M%S')}.tar.gz"
        catalogue.append((key, 1000, ts))

    fake = _FakeR2(catalogue)
    result = r2.apply_rotation(fake, r2.R2_BACKUP_PREFIX, now=now)

    def _key(day):
        return f"backups/dmai-backup-2026-07-{day:02d}-014500.tar.gz"

    # Kept: 7 most-recent daily (9-15) + Sunday Jul 5 (weekly) + Jul 1 (monthly).
    kept_days = {9, 10, 11, 12, 13, 14, 15, 5, 1}
    deleted_days = {2, 3, 4, 6, 7, 8}
    assert result["kept"] == len(kept_days)
    assert set(result["deleted"]) == {_key(dd) for dd in deleted_days}
    for dd in kept_days:
        assert _key(dd) not in result["deleted"]


# ── 6. restore-list sorted newest-first ──────────────────────────────────────
def test_restore_list_returns_sorted_backups(client, monkeypatch):
    monkeypatch.setenv("MASTER_PASSWORD", _MASTER_PW)
    catalogue = [
        ("backups/dmai-backup-2026-07-13-014500.tar.gz", 100,
         dt.datetime(2026, 7, 13, 1, 45)),
        ("backups/dmai-backup-2026-07-15-014500.tar.gz", 300,
         dt.datetime(2026, 7, 15, 1, 45)),
        ("backups/dmai-backup-2026-07-14-014500.tar.gz", 200,
         dt.datetime(2026, 7, 14, 1, 45)),
    ]
    fake = _FakeR2(catalogue)
    monkeypatch.setattr(r2, "R2BackupClient", lambda *a, **k: fake)

    resp = client.post("/api/cron/backup/restore-list",
                       headers={"X-Master-Password": _MASTER_PW})
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body["count"] == 3
    keys = [b["key"] for b in body["backups"]]
    assert keys == [
        "backups/dmai-backup-2026-07-15-014500.tar.gz",
        "backups/dmai-backup-2026-07-14-014500.tar.gz",
        "backups/dmai-backup-2026-07-13-014500.tar.gz",
    ]
    assert body["backups"][0]["size_bytes"] == 300


def test_restore_list_requires_auth(client, monkeypatch):
    monkeypatch.setenv("MASTER_PASSWORD", _MASTER_PW)
    resp = client.post("/api/cron/backup/restore-list")
    assert resp.status_code == 401
