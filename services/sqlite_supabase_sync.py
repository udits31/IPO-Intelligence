"""Bidirectionally mirror Capstone's SQLite telemetry rows to Supabase.

The SQLite ledgers remain the app's local working stores. The private
Supabase mirror is a durable second copy. Rows are keyed by the source
database, source table, and declared SQLite primary key. Existing rows are
never deleted. Concurrent edits are handled conservatively: if both sides
changed since the last successful sync, SQLite wins and the conflict is
reported rather than silently overwriting a saved prediction.

The backend calls :func:`request_sync` after data collection writes. The first
startup reconciliation can call :func:`sync_and_record_status` directly.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import sqlite3
import tempfile
import threading
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable

from db.supabase_client import get_supabase

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "telemetry"
STATE_DB = DATA_DIR / "supabase_sync_state.db"
STATUS_FILE = DATA_DIR / "supabase_sync_status.json"
STATE_TABLE = "capstone_sync_state"
META_TABLE = "capstone_sync_meta"
REMOTE_TABLE = "capstone_sqlite_mirror"
EXCLUDED_DB_NAMES = {STATE_DB.name}
PAGE_SIZE = 500
UPLOAD_BATCH = 100
logger = logging.getLogger(__name__)
_sync_requested = threading.Event()
_worker_lock = threading.Lock()
_sync_worker: threading.Thread | None = None


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _pack(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__capstone_type__": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported SQLite value type: {type(value).__name__}")


def _unpack(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__capstone_type__") == "bytes":
        return base64.b64decode(value["base64"], validate=True)
    if isinstance(value, dict):
        return {k: _unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack(v) for v in value]
    return value


def encode_row(row: dict[str, Any], primary_key_columns: list[str]) -> tuple[str, str, str]:
    if not primary_key_columns or any(row.get(k) is None for k in primary_key_columns):
        raise ValueError("Every mirrored row must have a non-null declared primary key")
    payload = {k: _pack(v) for k, v in row.items()}
    key = {k: payload[k] for k in primary_key_columns}
    payload_json = _canonical(payload)
    return _canonical(key), payload_json, hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def decode_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: _unpack(v) for k, v in payload.items()}


def _state_connection(path: Path = STATE_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {STATE_TABLE} (
        db_name TEXT NOT NULL, table_name TEXT NOT NULL, row_key TEXT NOT NULL,
        local_sha256 TEXT NOT NULL, remote_version INTEGER,
        PRIMARY KEY (db_name, table_name, row_key))""")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {META_TABLE} (
        key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
    return conn


def _tables(conn: sqlite3.Connection) -> list[tuple[str, list[str], list[str]]]:
    tables = []
    names = conn.execute("""SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'capstone_sync_%'
        ORDER BY name""").fetchall()
    for (name,) in names:
        info = conn.execute(f'PRAGMA table_info("{name.replace(chr(34), chr(34)*2)}")').fetchall()
        columns = [r[1] for r in info]
        pk = [r[1] for r in sorted((r for r in info if r[5]), key=lambda r: r[5])]
        if pk:
            tables.append((name, columns, pk))
        else:
            logger.warning("Skipping table without a primary key: %s", name)
    return tables


def _fetch_local_hashes(state: sqlite3.Connection, db: str, table: str) -> dict[str, str]:
    return {key: digest for key, digest in state.execute(
        f"SELECT row_key, local_sha256 FROM {STATE_TABLE} WHERE db_name=? AND table_name=?",
        (db, table),
    )}


def _update_local_row(conn: sqlite3.Connection, table: str, columns: list[str],
                      pk_columns: list[str], row: dict[str, Any]) -> None:
    quoted = lambda name: '"' + name.replace('"', '""') + '"'
    values = [row.get(c) for c in columns]
    where = " AND ".join(f"{quoted(k)}=?" for k in pk_columns)
    non_pk = [c for c in columns if c not in pk_columns]
    try:
        conn.execute("SAVEPOINT supabase_inbound")
        exists = conn.execute(
            f"SELECT 1 FROM {quoted(table)} WHERE {where}",
            [row[k] for k in pk_columns],
        ).fetchone()
        if exists:
            assignments = ",".join(f"{quoted(c)}=?" for c in non_pk)
            if assignments:
                conn.execute(
                    f"UPDATE {quoted(table)} SET {assignments} WHERE {where}",
                    [row[c] for c in non_pk] + [row[k] for k in pk_columns],
                )
        else:
            cols = ",".join(quoted(c) for c in columns)
            marks = ",".join("?" for _ in columns)
            conn.execute(f"INSERT INTO {quoted(table)} ({cols}) VALUES ({marks})", values)
        conn.execute("RELEASE supabase_inbound")
        conn.commit()
    except sqlite3.Error:
        conn.execute("ROLLBACK TO supabase_inbound")
        conn.execute("RELEASE supabase_inbound")
        conn.rollback()
        raise


def _remote_page(client: Any, version: int, start: int) -> list[dict[str, Any]]:
    result = (client.table(REMOTE_TABLE).select(
        "db_name,table_name,row_key,payload,row_sha256,sync_version"
    ).gt("sync_version", version).order("sync_version").range(start, start + PAGE_SIZE - 1).execute())
    return result.data or []


def _write_state(state: sqlite3.Connection, db: str, table: str, key: str,
                 digest: str, version: int | None) -> None:
    state.execute(f"""INSERT INTO {STATE_TABLE}(db_name,table_name,row_key,local_sha256,remote_version)
        VALUES(?,?,?,?,?) ON CONFLICT(db_name,table_name,row_key) DO UPDATE SET
        local_sha256=excluded.local_sha256, remote_version=COALESCE(excluded.remote_version,
            {STATE_TABLE}.remote_version)""", (db, table, key, digest, version))


def _sync_remote_to_local(client: Any, state: sqlite3.Connection, db_paths: dict[str, Path]) -> tuple[int, int, int]:
    row = state.execute(f"SELECT value FROM {META_TABLE} WHERE key='last_sync_version'").fetchone()
    cursor = int(row[0]) if row else 0
    inserted = updated = conflicts = 0
    while True:
        records = _remote_page(client, cursor, 0)
        if not records:
            break
        for record in records:
            db_name, table = record["db_name"], record["table_name"]
            if db_name=='runtime_assets':
                cursor=max(cursor,int(record['sync_version']))
                continue
            path = db_paths.get(db_name)
            if path is None:
                logger.warning("Supabase mirror references unknown SQLite database %s", db_name)
                cursor = max(cursor, int(record["sync_version"]))
                continue
            with closing(sqlite3.connect(path, timeout=30)) as local:
                local.row_factory = sqlite3.Row
                schemas = {t: (cols, pk) for t, cols, pk in _tables(local)}
                if table not in schemas:
                    logger.warning("Supabase mirror references unknown SQLite table %s.%s", db_name, table)
                    cursor = max(cursor, int(record["sync_version"]))
                    continue
                columns, pk = schemas[table]
                remote_row = decode_row(record["payload"])
                if set(remote_row) != set(columns) or any(k not in remote_row for k in pk):
                    logger.error("Schema mismatch for %s.%s; skipping inbound row", db_name, table)
                    conflicts += 1
                    cursor = max(cursor, int(record["sync_version"]))
                    continue
                # Native PostgreSQL view writes use PostgreSQL's JSONB digest.
                # Compare the decoded contents using the same Python serializer
                # as the local ledger; raw source/prediction hashes are retained.
                if str(record.get('row_sha256','')).startswith('pgjson:'):
                    record = {**record,'row_sha256':encode_row(remote_row,pk)[2]}
                key = record["row_key"]
                local_hashes = _fetch_local_hashes(state, db_name, table)
                prev_hash = local_hashes.get(key)
                quoted = '"' + table.replace('"', '""') + '"'
                where = " AND ".join('"' + k.replace('"', '""') + '"=?' for k in pk)
                local_record = local.execute(
                    f"SELECT * FROM {quoted} WHERE {where}",
                    [remote_row[k] for k in pk],
                ).fetchone()
                if local_record is None:
                    _update_local_row(local, table, columns, pk, remote_row)
                    _write_state(state, db_name, table, key, record["row_sha256"], int(record["sync_version"]))
                    inserted += 1
                else:
                    local_key, _, local_hash = encode_row(dict(local_record), pk)
                    if local_hash == record["row_sha256"]:
                        _write_state(state, db_name, table, key, local_hash, int(record["sync_version"]))
                    elif prev_hash is not None and prev_hash == local_hash:
                        try:
                            _update_local_row(local, table, columns, pk, remote_row)
                            _write_state(state, db_name, table, key, record["row_sha256"], int(record["sync_version"]))
                            updated += 1
                        except sqlite3.Error:
                            logger.exception("SQLite rejected cloud update for %s.%s key %s; preserving local row", db_name, table, key)
                            # Force the local version through the outbound pass;
                            # otherwise the old matching hash would mask this conflict.
                            _write_state(state, db_name, table, key, "__force_local__", int(record["sync_version"]))
                            conflicts += 1
                    else:
                        # Local changed since the last sync (or no common baseline exists).
                        # Leave it untouched; the outbound phase will publish the local version.
                        conflicts += 1
                state.commit()
            cursor = max(cursor, int(record["sync_version"]))
        state.execute(f"INSERT INTO {META_TABLE}(key,value) VALUES('last_sync_version',?) "
                      f"ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(cursor),))
        state.commit()
        if len(records) < PAGE_SIZE:
            break
    return inserted, updated, conflicts


def _sync_local_to_remote(client: Any, state: sqlite3.Connection, db_paths: dict[str, Path]) -> tuple[int, int]:
    changed = 0
    conflicts = 0
    for db_name, path in db_paths.items():
        with closing(sqlite3.connect(path, timeout=30)) as local:
            local.row_factory = sqlite3.Row
            for table, columns, pk in _tables(local):
                prior = _fetch_local_hashes(state, db_name, table)
                records = []
                for local_row in local.execute(f'SELECT * FROM "{table.replace(chr(34), chr(34)*2)}"'):
                    row = dict(local_row)
                    key, payload_json, digest = encode_row(row, pk)
                    if prior.get(key) == digest:
                        continue
                    records.append({"db_name": db_name, "table_name": table,
                                    "row_key": key, "payload": json.loads(payload_json),
                                    "row_sha256": digest})
                    _write_state(state, db_name, table, key, digest, None)
                    if len(records) >= UPLOAD_BATCH:
                        client.table(REMOTE_TABLE).upsert(
                            records, on_conflict="db_name,table_name,row_key"
                        ).execute()
                        changed += len(records)
                        records = []
                if records:
                    client.table(REMOTE_TABLE).upsert(
                        records, on_conflict="db_name,table_name,row_key"
                    ).execute()
                    changed += len(records)
                state.commit()
    # Every outbound upsert is already the local version, so its server-side
    # sync_version does not need to be downloaded and reprocessed next cycle.
    latest = (client.table(REMOTE_TABLE).select("sync_version")
              .order("sync_version", desc=True).limit(1).execute().data or [])
    if latest:
        prior = state.execute(f"SELECT value FROM {META_TABLE} WHERE key='last_sync_version'").fetchone()
        cursor = max(int(prior[0]) if prior else 0, int(latest[0]["sync_version"]))
        state.execute(f"INSERT INTO {META_TABLE}(key,value) VALUES('last_sync_version',?) "
                      f"ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(cursor),))
        state.commit()
    return changed, conflicts


def sync_once(client: Any = None, data_dir: Path = DATA_DIR, state_path: Path = STATE_DB) -> dict[str, Any]:
    """Reconcile all application SQLite rows with the private Supabase mirror."""
    client = client or get_supabase(use_service_role=True)
    paths = {p.name: p for p in sorted(data_dir.glob("*.db")) if p.name not in EXCLUDED_DB_NAMES}
    if not paths:
        raise RuntimeError(f"No application SQLite databases found in {data_dir}")
    with closing(_state_connection(state_path)) as state:
        added, refreshed, inbound_conflicts = _sync_remote_to_local(client, state, paths)
        from db.ledger import remote_collector_active
        # Real SDK clients use the lease. Test/fake mirror clients retain their
        # original adapter contract; no network calls are introduced into them.
        cloud_owner=remote_collector_active(client) if client.__class__.__module__.startswith('supabase') else False
        if cloud_owner:outbound,outbound_conflicts=0,0
        else:outbound, outbound_conflicts = _sync_local_to_remote(client, state, paths)
    return {"status": "ok", "databases": len(paths), "downloaded_rows": added,
            "cloud_updates_applied": refreshed, "uploaded_rows": outbound,
            "conflicts_preserved_local": inbound_conflicts + outbound_conflicts,
            "local_upload_paused_for_cloud_collector":cloud_owner}


def _save_status(result: dict[str, Any]) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="supabase_sync_", suffix=".tmp", dir=STATUS_FILE.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        temp.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temp.replace(STATUS_FILE)
    finally:
        temp.unlink(missing_ok=True)


def sync_and_record_status() -> dict[str, Any]:
    """Reconcile local rows immediately and persist a non-sensitive status."""
    from db.ledger import cloud_enabled
    if cloud_enabled():
        return {'status':'ok','backend':'supabase_api','mode':'native','sqlite_used':False}
    try:
        result = sync_once()
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        logger.exception("Capstone Supabase sync failed")
    _save_status(result)
    return result


def _sync_worker_loop() -> None:
    while _sync_requested.wait():
        _sync_requested.clear()
        result = sync_and_record_status()
        if result.get("status") == "ok":
            logger.info("SQLite/Supabase write-through complete: %s", result)
        else:
            logger.error("SQLite/Supabase write-through failed; local data retained for retry")


def request_sync(reason: str = "data_capture") -> None:
    """Queue an immediate in-process mirror after a committed SQLite write.

    Multiple collector notifications coalesce while a pass is running. This
    keeps Supabase writes out of SQLite transactions while ensuring the next
    pass includes every committed row. A failed pass is retried at the next
    capture or application startup.
    """
    global _sync_worker
    from db.ledger import cloud_enabled
    if cloud_enabled():
        # Native transactions already committed directly to Supabase.
        return
    _sync_requested.set()
    with _worker_lock:
        if _sync_worker is None or not _sync_worker.is_alive():
            _sync_worker = threading.Thread(
                target=_sync_worker_loop,
                name="sqlite-supabase-write-through",
                daemon=True,
            )
            _sync_worker.start()
    logger.debug("Queued Supabase mirror after %s", reason)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run one sync pass and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = sync_and_record_status()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
