"""Raw carrier artifacts shared by isolated encoding and frozen-head fitting."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "carl.encoder-carriers/v1"
RUNGS = (128, 256, 512, 768)


def digest(value: Any) -> str:
    return hashlib.sha256(payload(value)).hexdigest()


def payload(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".encoder-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def sample_key(sample: dict[str, Any], processed_tokens: int) -> str:
    value = {k: v for k, v in sample.items() if k != "event_id"}
    value["max_tokens"] = min(value["max_tokens"], processed_tokens)
    value["parts"] = [{k: v for k, v in p.items() if k != "path"} for p in value["parts"]]
    return digest(value)


def inputs(request: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for rows in request["groups"].values():
        for row in rows:
            for sample in (row["query"], row["positive"], *row["negatives"]):
                result[sample_key(sample, request["settings"]["processed_tokens"])] = sample
    return result


def admit(values: list[float]) -> None:
    if len(values) != 768 or not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite or invalid encoder carrier")
    if any(sum(v * v for v in values[:d]) <= 1e-24 for d in RUNGS):
        raise ValueError("Collapsed encoder prefix")


def load_carriers(
    manifest: Path, binding: dict[str, Any], required: set[str]
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    document = json.loads(manifest.read_bytes())
    if document.get("schema") != SCHEMA or document.get("binding") != binding:
        raise ValueError("Embedding cache binding changed")
    entries = document["entries"]
    if not required <= entries.keys():
        raise ValueError("Embedding cache population incomplete")
    vectors: dict[str, list[float]] = {}
    for key in sorted(required):
        entry = entries[key]
        raw = Path(entry["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Embedding cache artifact bytes changed")
        carrier = json.loads(raw)
        if carrier["key"] != key or carrier["binding"] != digest(binding):
            raise ValueError("Embedding cache artifact identity changed")
        values = carrier["values"]
        admit(values)
        vectors[key] = values
    return vectors, document


def write_carrier(
    root: Path, key: str, values: list[float], binding: dict[str, Any]
) -> dict[str, str]:
    admit(values)
    carrier = {"key": key, "binding": digest(binding), "values": values, "dimensions": RUNGS}
    identity = digest(carrier)
    path = root / (identity + ".json")
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != identity:
            raise ValueError("Embedding cache artifact bytes changed")
    else:
        atomic_json(path, carrier)
    return {"path": str(path.resolve()), "sha256": identity}
