"""DataToolkit — agent-callable surface over ``carl_core.data_handles``.

The vault lives in carl-core (dependency-free). This module wires:

* Audit: every op emits a :class:`~carl_core.interaction.Step` into the
  supplied :class:`~carl_core.interaction.InteractionChain`, action types
  ``DATA_OPEN`` / ``DATA_READ`` / ``DATA_TRANSFORM`` / ``DATA_PUBLISH``.
* Convenience decoders: ``read_text`` / ``read_json`` so the agent doesn't
  re-roll base64 + utf-8 dances for the common case.
* Transforms: head / tail / gzip / digest — each yields a new
  :class:`~carl_core.data_handles.DataRef` so the chain shows the derivation.
* Publish sinks: ``file:`` destinations today; HTTP + carl.camp later.

Design rule: the toolkit NEVER returns the raw ``bytes`` directly to the
agent. Reads come back base64-encoded in a structured dict. Decoders
return text/json; the ``read`` tool is for bytes-in-transit between other
tools. This keeps large payloads from flowing through agent context —
the agent sees fingerprints + sizes + a base64 blob bounded by
``preview_bytes``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from carl_core.data_handles import (
    DataError,
    DataRef,
    DataVault,
)
from carl_core.errors import CARLError, ValidationError
from carl_core.interaction import ActionType, InteractionChain

__all__ = [
    "DataToolkit",
    "DataToolkitError",
    "TransformOp",
]


TransformOp = Literal["head", "tail", "gzip", "gunzip", "digest"]
_TRANSFORM_OPS: frozenset[str] = frozenset(
    {"head", "tail", "gzip", "gunzip", "digest"}
)


class DataToolkitError(CARLError):
    """Base for ``carl.data_toolkit.*`` errors.

    Distinct from :class:`carl_core.data_handles.DataError` so callers can
    tell toolkit-layer surface errors (publish destination invalid, unknown
    transform) from vault-layer errors (revoked, expired, not found).
    """

    code = "carl.data_toolkit"


@dataclass
class DataToolkit:
    """Agent-callable wrapper around :class:`DataVault` + audit chain.

    The toolkit is the "Python object" the CARLAgent tool-dispatcher binds
    to. Methods here are the schema the agent sees; they all take + return
    JSON-native dicts so they serialize cleanly across tool boundaries.
    """

    vault: DataVault
    chain: InteractionChain
    preview_bytes: int = 65536  # default cap on untruncated reads (64 KB)
    max_read_bytes: int = 16 * 1024 * 1024  # hard upper bound per read (16 MB)

    @classmethod
    def build(
        cls,
        chain: InteractionChain,
        *,
        preview_bytes: int = 65536,
        max_read_bytes: int = 16 * 1024 * 1024,
    ) -> DataToolkit:
        return cls(
            vault=DataVault(),
            chain=chain,
            preview_bytes=preview_bytes,
            max_read_bytes=max_read_bytes,
        )

    # -- open / ingest ---------------------------------------------------

    def open_file(
        self,
        path: str | Path,
        *,
        content_type: str | None = None,
        ttl_s: int | None = None,
    ) -> dict[str, Any]:
        """Register a filesystem path. Returns the public ref descriptor."""
        ref = self.vault.open_file(path, content_type=content_type, ttl_s=ttl_s)
        descriptor = ref.describe()
        self.chain.record(
            ActionType.DATA_OPEN,
            "data.open_file",
            input={"path": str(path), "content_type": content_type, "ttl_s": ttl_s},
            output=descriptor,
            success=True,
        )
        return descriptor

    def open_bytes(
        self,
        data: bytes,
        *,
        uri: str | None = None,
        content_type: str | None = None,
        ttl_s: int | None = None,
    ) -> dict[str, Any]:
        """Register inline bytes. Returns the public ref descriptor."""
        ref = self.vault.put_bytes(
            data, uri=uri, content_type=content_type, ttl_s=ttl_s
        )
        descriptor = ref.describe()
        self.chain.record(
            ActionType.DATA_OPEN,
            "data.open_bytes",
            input={
                "size_bytes": len(data),
                "uri": uri,
                "content_type": content_type,
                "ttl_s": ttl_s,
            },
            output=descriptor,
            success=True,
        )
        return descriptor

    def open_url(
        self,
        url: str,
        *,
        content_type: str | None = None,
        ttl_s: int | None = None,
    ) -> dict[str, Any]:
        """Register a remote URL. Vault does NOT fetch — resolver is external."""
        ref = self.vault.put_external(
            url, kind="url", content_type=content_type, ttl_s=ttl_s
        )
        descriptor = ref.describe()
        self.chain.record(
            ActionType.DATA_OPEN,
            "data.open_url",
            input={"url": url, "content_type": content_type, "ttl_s": ttl_s},
            output=descriptor,
            success=True,
        )
        return descriptor

    # -- read ------------------------------------------------------------

    def read(
        self,
        ref_id: str,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> dict[str, Any]:
        """Read bytes from a handle. Returns a structured descriptor — NOT raw bytes.

        ``length=None`` reads up to ``preview_bytes`` (default 64 KB). Larger
        reads must be explicit; ``length > max_read_bytes`` raises. This
        keeps the agent from accidentally slurping a 1 GB file into its
        own context.
        """
        ref = self._ref_from_id(ref_id)
        effective_length = length if length is not None else self.preview_bytes
        if effective_length > self.max_read_bytes:
            raise DataToolkitError(
                f"read length {effective_length} exceeds max_read_bytes "
                f"{self.max_read_bytes}; stream via repeated reads or use transform",
                code="carl.data_toolkit.read_too_large",
                context={"length": effective_length, "max": self.max_read_bytes},
            )

        data = self.vault.read(ref, offset=offset, length=effective_length)
        size_total = self.vault.size_of(ref)
        eof = (
            size_total is not None
            and offset + len(data) >= size_total
        ) or (len(data) < effective_length)
        digest = hashlib.sha256(data).hexdigest()[:12]

        payload = {
            "ref_id": ref_id,
            "offset": offset,
            "length_returned": len(data),
            "length_requested": effective_length,
            "eof": eof,
            "sha256_12": digest,
            "bytes_b64": base64.b64encode(data).decode("ascii"),
        }
        self.chain.record(
            ActionType.DATA_READ,
            "data.read",
            input={
                "ref_id": ref_id,
                "offset": offset,
                "length": length,
            },
            output={k: v for k, v in payload.items() if k != "bytes_b64"},
            success=True,
        )
        return payload

    def read_text(
        self,
        ref_id: str,
        *,
        offset: int = 0,
        max_bytes: int | None = None,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        """UTF-8 (by default) decode convenience. Returns ``{text, eof, ...}``."""
        raw = self.read(ref_id, offset=offset, length=max_bytes)
        try:
            text = base64.b64decode(raw["bytes_b64"]).decode(encoding)
        except UnicodeDecodeError as exc:
            raise DataToolkitError(
                f"ref {ref_id} did not decode as {encoding}",
                code="carl.data_toolkit.decode_error",
                context={"ref_id": ref_id, "encoding": encoding},
                cause=exc,
            ) from exc
        return {
            "ref_id": ref_id,
            "text": text,
            "offset": raw["offset"],
            "length_returned": raw["length_returned"],
            "eof": raw["eof"],
            "sha256_12": raw["sha256_12"],
        }

    def read_json(self, ref_id: str) -> dict[str, Any]:
        """Parse the whole ref as JSON. Returns ``{ref_id, value, sha256_12}``.

        Reads up to ``max_read_bytes``; JSON payloads larger than that should
        be pre-sliced via :meth:`transform` or handled stream-style (future).
        """
        raw = self.read(ref_id, offset=0, length=self.max_read_bytes)
        try:
            value = json.loads(base64.b64decode(raw["bytes_b64"]))
        except (json.JSONDecodeError, ValueError) as exc:
            raise DataToolkitError(
                f"ref {ref_id} did not parse as JSON",
                code="carl.data_toolkit.decode_error",
                context={"ref_id": ref_id},
                cause=exc,
            ) from exc
        return {
            "ref_id": ref_id,
            "value": value,
            "sha256_12": raw["sha256_12"],
            "eof": raw["eof"],
        }

    # -- transform -------------------------------------------------------

    def transform(
        self,
        ref_id: str,
        op: TransformOp,
        *,
        n: int | None = None,
    ) -> dict[str, Any]:
        """Produce a derived :class:`DataRef` by applying ``op``.

        Supported ops:

        * ``head`` — first ``n`` bytes (default 4096).
        * ``tail`` — last ``n`` bytes (default 4096).
        * ``gzip`` — gzip compression.
        * ``gunzip`` — gzip decompression.
        * ``digest`` — write the full sha256 hex as a new bytes ref.

        Returns the descriptor of the derived ref.
        """
        if op not in _TRANSFORM_OPS:
            raise ValidationError(
                f"unknown transform op: {op!r}; valid: {sorted(_TRANSFORM_OPS)}",
                code="carl.data_toolkit.unknown_op",
                context={"op": op, "valid": sorted(_TRANSFORM_OPS)},
            )
        ref = self._ref_from_id(ref_id)
        full = self.vault.read_all(ref)

        out_bytes: bytes
        derived_uri: str
        content_type: str | None = None

        if op == "head":
            take = n if n is not None else 4096
            out_bytes = full[:take]
            derived_uri = f"carl-data://derived/head/{ref_id}/{take}"
        elif op == "tail":
            take = n if n is not None else 4096
            out_bytes = full[-take:] if take > 0 else b""
            derived_uri = f"carl-data://derived/tail/{ref_id}/{take}"
        elif op == "gzip":
            import gzip as _gzip

            out_bytes = _gzip.compress(full)
            derived_uri = f"carl-data://derived/gzip/{ref_id}"
            content_type = "application/gzip"
        elif op == "gunzip":
            import gzip as _gzip

            try:
                out_bytes = _gzip.decompress(full)
            except OSError as exc:
                raise DataToolkitError(
                    f"ref {ref_id} is not a valid gzip stream",
                    code="carl.data_toolkit.transform_error",
                    context={"ref_id": ref_id, "op": op},
                    cause=exc,
                ) from exc
            derived_uri = f"carl-data://derived/gunzip/{ref_id}"
        elif op == "digest":
            out_bytes = hashlib.sha256(full).hexdigest().encode("ascii")
            derived_uri = f"carl-data://derived/digest/{ref_id}"
            content_type = "text/plain"
        else:  # pragma: no cover - guarded by _TRANSFORM_OPS check above
            raise ValidationError(
                f"unknown transform op: {op!r}",
                code="carl.data_toolkit.unknown_op",
                context={"op": op},
            )

        derived = self.vault.put_bytes(
            out_bytes, uri=derived_uri, content_type=content_type
        )
        # Mark as derived kind via put_external-style alias: keep kind=bytes
        # since the vault treats it as bytes; the derivation provenance is
        # recorded in the DATA_TRANSFORM step below.
        descriptor = derived.describe()
        self.chain.record(
            ActionType.DATA_TRANSFORM,
            f"data.transform.{op}",
            input={
                "source_ref_id": ref_id,
                "op": op,
                "n": n,
                "source_sha256_12": hashlib.sha256(full).hexdigest()[:12],
            },
            output=descriptor,
            success=True,
        )
        return descriptor

    # -- publish ---------------------------------------------------------

    def publish_to_file(
        self,
        ref_id: str,
        destination: str | Path,
        *,
        mode: Literal["w", "x"] = "w",
    ) -> dict[str, Any]:
        """Write the ref's contents to ``destination`` on the local filesystem.

        ``mode="x"`` fails if the destination already exists (safer default
        for agent workflows that shouldn't overwrite). ``mode="w"`` truncates
        existing files.
        """
        ref = self._ref_from_id(ref_id)
        full = self.vault.read_all(ref)
        dest = Path(destination).expanduser().resolve()

        if mode == "x" and dest.exists():
            raise DataToolkitError(
                f"destination {dest} already exists; use mode='w' to overwrite",
                code="carl.data_toolkit.destination_exists",
                context={"destination": str(dest)},
            )

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(full)
        sha12 = hashlib.sha256(full).hexdigest()[:12]

        result = {
            "ref_id": ref_id,
            "destination": str(dest),
            "bytes_written": len(full),
            "sha256_12": sha12,
        }
        self.chain.record(
            ActionType.DATA_PUBLISH,
            "data.publish_to_file",
            input={"ref_id": ref_id, "destination": str(dest), "mode": mode},
            output=result,
            success=True,
        )
        return result

    # -- lifecycle / query -----------------------------------------------

    def fingerprint(self, ref_id: str) -> str:
        """12-hex sha256 preview. Safe to show the agent / log."""
        return self.vault.fingerprint_of(self._ref_from_id(ref_id))

    def sha256(self, ref_id: str) -> str:
        """Full 64-hex sha256."""
        return self.vault.sha256_of(self._ref_from_id(ref_id))

    def describe(self, ref_id: str) -> dict[str, Any]:
        """Public descriptor for a ref (kind, size, uri, sha256, ttl)."""
        return self._ref_from_id(ref_id).describe()

    def list_handles(self) -> list[dict[str, Any]]:
        return [r.describe() for r in self.vault.list_refs()]

    def revoke(self, ref_id: str) -> bool:
        ref = self._ref_from_id(ref_id)
        return self.vault.revoke(ref)

    # -- agent schema helper (for tool registries that introspect) -------

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return Anthropic-tool-style schemas for each agent-callable method.

        Matches the ``tool_schema`` convention used by
        ``src/carl_studio/chat_agent.py``: flat dict with ``name`` /
        ``description`` / ``input_schema``. Not wired into the dispatcher
        here — the chat_agent owner chooses which tools to expose.
        """
        return [
            {
                "name": "data_open_file",
                "description": "Open a filesystem path as a DataRef handle.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content_type": {"type": ["string", "null"]},
                        "ttl_s": {"type": ["integer", "null"]},
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "data_read_text",
                "description": "Read a DataRef's content as decoded text.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ref_id": {"type": "string"},
                        "offset": {"type": "integer", "default": 0},
                        "max_bytes": {"type": ["integer", "null"]},
                        "encoding": {"type": "string", "default": "utf-8"},
                    },
                    "required": ["ref_id"],
                },
            },
            {
                "name": "data_read_json",
                "description": "Parse a DataRef as JSON.",
                "input_schema": {
                    "type": "object",
                    "properties": {"ref_id": {"type": "string"}},
                    "required": ["ref_id"],
                },
            },
            {
                "name": "data_transform",
                "description": "Produce a derived DataRef via head/tail/gzip/gunzip/digest.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ref_id": {"type": "string"},
                        "op": {
                            "type": "string",
                            "enum": sorted(_TRANSFORM_OPS),
                        },
                        "n": {"type": ["integer", "null"]},
                    },
                    "required": ["ref_id", "op"],
                },
            },
            {
                "name": "data_publish_to_file",
                "description": "Write a DataRef's content to a filesystem destination.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ref_id": {"type": "string"},
                        "destination": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": ["w", "x"],
                            "default": "w",
                        },
                    },
                    "required": ["ref_id", "destination"],
                },
            },
            {
                "name": "data_list_handles",
                "description": "List current non-revoked DataRefs.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    # -- internals -------------------------------------------------------

    def _ref_from_id(self, ref_id: str) -> DataRef:
        try:
            parsed = uuid.UUID(ref_id)
        except (TypeError, ValueError) as exc:
            raise DataToolkitError(
                f"ref_id is not a valid UUID: {ref_id!r}",
                code="carl.data_toolkit.invalid_ref_id",
                context={"ref_id": ref_id},
                cause=exc,
            ) from exc
        for ref in self.vault.list_refs():
            if ref.ref_id == parsed:
                return ref
        # Surface as DataError so callers already catching data.* codes see it.
        raise DataError(
            f"unknown or revoked data handle: {ref_id}",
            code="carl.data.not_found",
            context={"ref_id": ref_id},
        )
