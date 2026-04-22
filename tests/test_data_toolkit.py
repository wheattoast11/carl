"""Tests for carl_studio.handles.data.DataToolkit — agent-callable surface.

Covers the integration layer on top of ``carl_core.data_handles``:
audit-step emission, read preview cap, transforms, publish, JSON/text
decoders, tool_schemas introspection.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path
from typing import Any, cast

import pytest

from carl_core.data_handles import DataError
from carl_core.interaction import ActionType, InteractionChain
from carl_studio.handles.data import DataToolkit, DataToolkitError


# ---------------------------------------------------------------------------
# open_* methods emit DATA_OPEN + return descriptors
# ---------------------------------------------------------------------------


def test_open_bytes_emits_step_and_returns_descriptor() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    desc = toolkit.open_bytes(b"hello", content_type="text/plain")
    assert desc["kind"] == "bytes"
    assert desc["size_bytes"] == 5
    assert desc["content_type"] == "text/plain"
    assert desc["sha256"] is not None
    # One DATA_OPEN step recorded
    opens = chain.by_action(ActionType.DATA_OPEN)
    assert len(opens) == 1
    assert opens[0].name == "data.open_bytes"


def test_open_file_emits_step(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_bytes(b"hi")
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    desc = toolkit.open_file(p)
    assert desc["kind"] == "file"
    opens = chain.by_action(ActionType.DATA_OPEN)
    assert len(opens) == 1
    assert opens[0].name == "data.open_file"


def test_open_url_emits_step() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    desc = toolkit.open_url("https://example.com/a.json")
    assert desc["kind"] == "url"
    assert desc["uri"] == "https://example.com/a.json"
    opens = chain.by_action(ActionType.DATA_OPEN)
    assert len(opens) == 1
    assert opens[0].name == "data.open_url"


# ---------------------------------------------------------------------------
# read — preview cap, base64, eof
# ---------------------------------------------------------------------------


def test_read_returns_base64_and_eof() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    desc = toolkit.open_bytes(b"abc")
    result = toolkit.read(desc["ref_id"])
    assert base64.b64decode(result["bytes_b64"]) == b"abc"
    assert result["eof"] is True
    assert result["length_returned"] == 3
    # DATA_READ recorded without raw bytes in the output
    reads = chain.by_action(ActionType.DATA_READ)
    assert len(reads) == 1
    raw_out = cast(dict[str, Any], reads[0].output)
    assert "bytes_b64" not in raw_out
    assert raw_out["length_returned"] == 3


def test_read_preview_cap() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain, preview_bytes=10)
    desc = toolkit.open_bytes(b"x" * 1000)
    result = toolkit.read(desc["ref_id"])
    assert result["length_returned"] == 10
    assert result["eof"] is False


def test_read_explicit_length_bypass_preview() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain, preview_bytes=10)
    desc = toolkit.open_bytes(b"x" * 50)
    result = toolkit.read(desc["ref_id"], length=50)
    assert result["length_returned"] == 50
    assert result["eof"] is True


def test_read_rejects_oversized_length() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain, max_read_bytes=100)
    desc = toolkit.open_bytes(b"x" * 50)
    with pytest.raises(DataToolkitError) as exc:
        toolkit.read(desc["ref_id"], length=999)
    assert exc.value.code == "carl.data_toolkit.read_too_large"


def test_read_unknown_ref_raises() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    # Non-UUID string → toolkit rejects before hitting vault
    with pytest.raises(DataToolkitError) as exc:
        toolkit.read("not-a-uuid-just-gibberish")
    assert exc.value.code == "carl.data_toolkit.invalid_ref_id"
    # Valid UUID that's not registered → vault says not_found
    import uuid as _uuid

    with pytest.raises(DataError) as exc2:
        toolkit.read(str(_uuid.uuid4()))
    assert exc2.value.code == "carl.data.not_found"


# ---------------------------------------------------------------------------
# read_text / read_json decoders
# ---------------------------------------------------------------------------


def test_read_text_utf8() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    desc = toolkit.open_bytes("héllo".encode("utf-8"))
    result = toolkit.read_text(desc["ref_id"])
    assert result["text"] == "héllo"


def test_read_text_decode_error() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    # UTF-16 bytes — not valid UTF-8
    desc = toolkit.open_bytes("x".encode("utf-16"))
    with pytest.raises(DataToolkitError) as exc:
        toolkit.read_text(desc["ref_id"])
    assert exc.value.code == "carl.data_toolkit.decode_error"


def test_read_json() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    desc = toolkit.open_bytes(json.dumps({"a": 1, "b": [2, 3]}).encode())
    result = toolkit.read_json(desc["ref_id"])
    assert result["value"] == {"a": 1, "b": [2, 3]}


def test_read_json_decode_error() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    desc = toolkit.open_bytes(b"not json")
    with pytest.raises(DataToolkitError) as exc:
        toolkit.read_json(desc["ref_id"])
    assert exc.value.code == "carl.data_toolkit.decode_error"


# ---------------------------------------------------------------------------
# transform — head, tail, gzip, gunzip, digest
# ---------------------------------------------------------------------------


def test_transform_head_tail() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    desc = toolkit.open_bytes(b"abcdefghij")
    head = toolkit.transform(desc["ref_id"], "head", n=3)
    tail = toolkit.transform(desc["ref_id"], "tail", n=3)
    head_raw = toolkit.read(head["ref_id"], length=10)
    tail_raw = toolkit.read(tail["ref_id"], length=10)
    assert base64.b64decode(head_raw["bytes_b64"]) == b"abc"
    assert base64.b64decode(tail_raw["bytes_b64"]) == b"hij"
    # Transforms emit DATA_TRANSFORM
    assert len(chain.by_action(ActionType.DATA_TRANSFORM)) == 2


def test_transform_gzip_roundtrip() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    desc = toolkit.open_bytes(b"the quick brown fox jumps over the lazy dog" * 20)
    compressed = toolkit.transform(desc["ref_id"], "gzip")
    comp_raw = toolkit.read(compressed["ref_id"], length=4096)
    comp_bytes = base64.b64decode(comp_raw["bytes_b64"])
    # Round-trip via gunzip transform
    decompressed = toolkit.transform(compressed["ref_id"], "gunzip")
    decomp_raw = toolkit.read(decompressed["ref_id"], length=4096)
    assert base64.b64decode(decomp_raw["bytes_b64"]) == (
        b"the quick brown fox jumps over the lazy dog" * 20
    )
    # Manual gzip verification
    assert gzip.decompress(comp_bytes) == (
        b"the quick brown fox jumps over the lazy dog" * 20
    )


def test_transform_digest() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    desc = toolkit.open_bytes(b"hello")
    digest_ref = toolkit.transform(desc["ref_id"], "digest")
    raw = toolkit.read(digest_ref["ref_id"], length=256)
    hex_digest = base64.b64decode(raw["bytes_b64"]).decode("ascii")
    assert len(hex_digest) == 64
    import hashlib

    assert hex_digest == hashlib.sha256(b"hello").hexdigest()


def test_transform_rejects_unknown_op() -> None:
    from carl_core.errors import ValidationError

    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    desc = toolkit.open_bytes(b"x")
    with pytest.raises(ValidationError) as exc:
        toolkit.transform(desc["ref_id"], cast(Any, "nonexistent"))
    assert exc.value.code == "carl.data_toolkit.unknown_op"


def test_transform_gunzip_invalid_data() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    desc = toolkit.open_bytes(b"not gzip data")
    with pytest.raises(DataToolkitError) as exc:
        toolkit.transform(desc["ref_id"], "gunzip")
    assert exc.value.code == "carl.data_toolkit.transform_error"


# ---------------------------------------------------------------------------
# publish_to_file
# ---------------------------------------------------------------------------


def test_publish_writes_file_and_emits_step(tmp_path: Path) -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    desc = toolkit.open_bytes(b"write me")
    dest = tmp_path / "out.bin"
    result = toolkit.publish_to_file(desc["ref_id"], dest)
    assert dest.read_bytes() == b"write me"
    assert result["bytes_written"] == 8
    pubs = chain.by_action(ActionType.DATA_PUBLISH)
    assert len(pubs) == 1


def test_publish_mode_x_rejects_existing(tmp_path: Path) -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    desc = toolkit.open_bytes(b"x")
    dest = tmp_path / "exists.bin"
    dest.write_bytes(b"old")
    with pytest.raises(DataToolkitError) as exc:
        toolkit.publish_to_file(desc["ref_id"], dest, mode="x")
    assert exc.value.code == "carl.data_toolkit.destination_exists"


def test_publish_creates_parent_dirs(tmp_path: Path) -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    desc = toolkit.open_bytes(b"nested")
    dest = tmp_path / "a" / "b" / "c.bin"
    toolkit.publish_to_file(desc["ref_id"], dest)
    assert dest.read_bytes() == b"nested"


# ---------------------------------------------------------------------------
# Introspection: list_handles / describe / fingerprint / sha256 / tool_schemas
# ---------------------------------------------------------------------------


def test_list_handles_and_describe() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    d1 = toolkit.open_bytes(b"a")
    d2 = toolkit.open_bytes(b"bb")
    toolkit.revoke(d2["ref_id"])
    listed = toolkit.list_handles()
    assert len(listed) == 1
    assert listed[0]["ref_id"] == d1["ref_id"]
    assert toolkit.describe(d1["ref_id"])["size_bytes"] == 1


def test_fingerprint_and_sha256() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    desc = toolkit.open_bytes(b"abc")
    fp = toolkit.fingerprint(desc["ref_id"])
    sha = toolkit.sha256(desc["ref_id"])
    assert len(fp) == 12
    assert len(sha) == 64
    assert sha.startswith(fp)


def test_tool_schemas_cover_agent_surface() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    schemas = toolkit.tool_schemas()
    names = {s["name"] for s in schemas}
    # Agent needs at least: open_file, read_text, read_json, transform,
    # publish_to_file, list_handles
    for expected in (
        "data_open_file",
        "data_read_text",
        "data_read_json",
        "data_transform",
        "data_publish_to_file",
        "data_list_handles",
    ):
        assert expected in names
    for schema in schemas:
        assert "description" in schema
        assert "input_schema" in schema


# ---------------------------------------------------------------------------
# Audit invariants — no raw bytes in serialized chain
# ---------------------------------------------------------------------------


def test_chain_serialization_never_leaks_bytes() -> None:
    chain = InteractionChain()
    toolkit = DataToolkit.build(chain)
    secret_like = b"p4ssw0rd-equivalent-payload-bytes-xyz"
    desc = toolkit.open_bytes(secret_like)
    toolkit.read(desc["ref_id"])
    toolkit.transform(desc["ref_id"], "digest")
    # Serialize — the raw payload must not appear verbatim
    serialized = chain.to_jsonl()
    assert b"p4ssw0rd-equivalent-payload-bytes-xyz" not in serialized.encode()
    # Base64'd form shouldn't appear either (DATA_READ output elides bytes_b64)
    b64 = base64.b64encode(secret_like).decode("ascii")
    assert b64 not in serialized
