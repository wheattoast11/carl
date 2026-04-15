"""Sync protocol for carl-studio CLI — push/pull to/from Supabase.

Content-hash based: same data = no transfer.
Opt-in: user controls when data goes to cloud.

Usage:
    from platform.cli.sync import push, pull

    push()                      # Push all unsynced entities
    push(['runs'])              # Push only runs
    pull()                      # Pull all updates since last pull
    pull(since='2026-04-01')    # Pull updates since specific date
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any, Optional

from carl_studio.db import LocalDB


class SyncError(Exception):
    """Raised when sync fails after retries."""

    pass


def _supabase_request(
    db: LocalDB,
    function: str,
    method: str = "POST",
    body: dict | None = None,
    params: dict | None = None,
) -> dict:
    """Make authenticated request to Supabase Edge Function."""
    jwt = db.get_auth("jwt")
    supabase_url = db.get_config("supabase_url")

    if not jwt or not supabase_url:
        raise SyncError("Not authenticated. Run: carl camp login")

    url = f"{supabase_url}/functions/v1/{function}"
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{query}"

    headers = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
    }

    data = json.dumps(body).encode() if body else None

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else str(e)
        raise SyncError(f"Sync failed ({e.code}): {error_body}")
    except urllib.error.URLError as e:
        raise SyncError(f"Network error: {e.reason}")


# ─── Push ────────────────────────────────────────────────────


def push(
    entity_types: list[str] | None = None,
    db: LocalDB | None = None,
) -> dict[str, int]:
    """Push unsynced local entities to Supabase.

    1. Query SQLite for unsynced entities (synced=0)
    2. Batch POST to Supabase Edge Function
    3. Mark as synced on success
    4. Queue for retry on failure

    Returns: {entity_type: count_synced}
    """
    db = db or LocalDB()
    entity_types = entity_types or ["runs"]
    results: dict[str, int] = {}

    for etype in entity_types:
        unsynced = db.get_unsynced(etype)
        if not unsynced:
            results[etype] = 0
            continue

        # Clean entities for transport (remove SQLite-specific fields)
        entities = []
        for entity in unsynced:
            clean = {k: v for k, v in entity.items() if k not in ("synced", "remote_id")}
            # Parse JSON strings back to dicts for transport
            for json_field in ("config", "result", "data", "criteria", "results"):
                if json_field in clean and isinstance(clean[json_field], str):
                    try:
                        clean[json_field] = json.loads(clean[json_field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            entities.append(clean)

        try:
            response = _supabase_request(
                db,
                "sync-push",
                body={
                    "entity_type": etype,
                    "entities": entities,
                },
            )

            synced_ids = response.get("ids", [])
            for entity, remote_id in zip(unsynced, synced_ids):
                db.mark_synced(entity["id"], remote_id)

            results[etype] = response.get("synced", 0)

        except SyncError:
            # Queue for retry
            for entity in unsynced:
                db.enqueue_sync("push", etype, entity["id"], entity)
            results[etype] = 0

    return results


# ─── Pull ────────────────────────────────────────────────────


def pull(
    since: str | None = None,
    entity_types: list[str] | None = None,
    db: LocalDB | None = None,
) -> dict[str, int]:
    """Pull updates from Supabase to local SQLite.

    1. Query Supabase for entities updated since last pull
    2. Compare content_hash to detect actual changes
    3. Upsert into local SQLite
    4. Update last_pull timestamp

    Returns: {entity_type: count_pulled}
    """
    db = db or LocalDB()
    since = since or db.get_config("last_pull_at") or "1970-01-01T00:00:00Z"

    params: dict[str, str] = {"since": since}
    if entity_types:
        params["types"] = ",".join(entity_types)

    response = _supabase_request(db, "sync-pull", method="GET", params=params)

    results: dict[str, int] = {}
    entities = response.get("entities", {})

    for etype, items in entities.items():
        count = 0
        for item in items:
            # Check if we already have this with same content_hash
            local = db.get_run(item["id"]) if etype == "runs" else None
            if local and local.get("content_hash") == item.get("content_hash"):
                continue

            # Upsert into local DB
            if etype == "runs":
                existing = db.get_run(item["id"])
                if existing:
                    db.update_run(item["id"], item)
                else:
                    item["synced"] = 1
                    item["remote_id"] = item["id"]
                    db.insert_run(item)
                count += 1

        results[etype] = count

    # Update last pull timestamp
    pulled_at = response.get("pulled_at", datetime.now(timezone.utc).isoformat())
    db.set_config("last_pull_at", pulled_at)

    return results


# ─── Retry Queue ─────────────────────────────────────────────


def process_sync_queue(
    db: LocalDB | None = None,
    max_retries: int = 3,
) -> dict[str, int]:
    """Process pending items in the sync queue.

    Called periodically or on `carl push --retry`.
    Items that fail more than max_retries times are marked as failed.

    Returns: {synced: N, failed: N, remaining: N}
    """
    db = db or LocalDB()
    pending = db.get_pending_sync()
    synced = 0
    failed = 0

    for item in pending:
        if item["retry_count"] >= max_retries:
            db.update_sync_status(item["id"], "failed", "Max retries exceeded")
            failed += 1
            continue

        try:
            payload = (
                json.loads(item["payload"]) if isinstance(item["payload"], str) else item["payload"]
            )

            _supabase_request(
                db,
                f"sync-{item['operation']}",
                body={
                    "entity_type": item["entity_type"],
                    "entities": [payload],
                },
            )

            db.update_sync_status(item["id"], "synced")
            synced += 1

        except SyncError as e:
            db.update_sync_status(item["id"], "pending", str(e))

    remaining = len(pending) - synced - failed
    return {"synced": synced, "failed": failed, "remaining": remaining}
