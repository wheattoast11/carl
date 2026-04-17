"""Property tests for carl_studio.x402._parse_x_payment_header.

Invariants:
- Well-formed JSON headers round-trip through the PaymentRequirement model.
- Completely arbitrary strings never crash the parser (best-effort fallback).
"""
from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from carl_studio.x402 import PaymentRequirement, _parse_x_payment_header


_amount = st.text(
    alphabet=st.characters(whitelist_categories=("Nd",), whitelist_characters="."),
    min_size=1,
    max_size=10,
).filter(lambda s: s.count(".") <= 1 and not s.endswith("."))

_identifier = st.text(
    alphabet=st.characters(
        min_codepoint=0x30,
        max_codepoint=0x7A,
        whitelist_categories=("Ll", "Lu", "Nd"),
    ),
    min_size=1,
    max_size=12,
)


@given(
    amount=_amount,
    token=_identifier,
    chain=_identifier,
    recipient=_identifier,
    facilitator=_identifier,
)
@settings(max_examples=100, deadline=None)
def test_wellformed_json_header_roundtrips(
    amount: str,
    token: str,
    chain: str,
    recipient: str,
    facilitator: str,
) -> None:
    """A JSON-encoded PaymentRequirement header parses back to the same fields."""
    payload = {
        "amount": amount,
        "token": token,
        "chain": chain,
        "recipient": recipient,
        "facilitator": facilitator,
    }
    header = json.dumps(payload)
    parsed = _parse_x_payment_header(header)
    assert isinstance(parsed, PaymentRequirement)
    assert parsed.amount == amount
    assert parsed.token == token
    assert parsed.chain == chain
    assert parsed.recipient == recipient
    assert parsed.facilitator == facilitator


@given(garbage=st.text(max_size=64))
@settings(max_examples=100, deadline=None)
def test_garbage_header_falls_back_to_requirement(garbage: str) -> None:
    """Parser should never raise on arbitrary string input.

    x402 headers in the wild are permissive: the function MUST fall back to a
    semicolon-separated key=value parse when JSON decoding fails, and always
    return a PaymentRequirement (possibly with defaults).
    """
    result = _parse_x_payment_header(garbage)
    assert isinstance(result, PaymentRequirement)
    # Defaults are strings; no field is ever None.
    assert isinstance(result.amount, str)
    assert isinstance(result.token, str)
    assert isinstance(result.chain, str)
    assert isinstance(result.recipient, str)
    assert isinstance(result.facilitator, str)


@given(
    amount=_amount,
    token=_identifier,
    chain=_identifier,
)
@settings(max_examples=50, deadline=None)
def test_semicolon_kv_fallback(amount: str, token: str, chain: str) -> None:
    """Parser accepts ``amount=X; token=Y; chain=Z`` when JSON fails."""
    header = f"amount={amount}; token={token}; chain={chain}"
    parsed = _parse_x_payment_header(header)
    assert parsed.amount == amount
    assert parsed.token == token
    assert parsed.chain == chain
