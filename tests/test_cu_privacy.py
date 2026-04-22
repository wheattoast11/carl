"""Tests for carl_studio.cu.privacy — content-level redaction."""

from __future__ import annotations

from carl_studio.cu.privacy import redact_preview_spans, redact_text


def test_redact_email() -> None:
    out = redact_text("contact tej@terminals.tech please")
    assert "tej@terminals.tech" not in out
    assert "<REDACTED:email>" in out


def test_redact_phone() -> None:
    out = redact_text("call (555) 123-4567 tomorrow")
    assert "555" not in out
    assert "<REDACTED:phone>" in out


def test_redact_ssn() -> None:
    out = redact_text("SSN: 123-45-6789")
    assert "123-45-6789" not in out
    assert "<REDACTED:ssn>" in out


def test_redact_credit_card() -> None:
    out = redact_text("card 4111 1111 1111 1111")
    assert "4111 1111 1111 1111" not in out
    assert "<REDACTED:credit_card>" in out


def test_redact_ipv4() -> None:
    out = redact_text("server at 192.168.1.1")
    assert "192.168.1.1" not in out
    assert "<REDACTED:ipv4>" in out


def test_redact_dob_dash() -> None:
    out = redact_text("dob 1994-05-20")
    assert "1994-05-20" not in out
    assert "<REDACTED:date_of_birth>" in out


def test_redact_dob_slash() -> None:
    out = redact_text("dob 05/20/1994")
    assert "05/20/1994" not in out
    assert "<REDACTED:date_of_birth>" in out


def test_no_false_positive_on_version_number() -> None:
    out = redact_text("Python 3.11.5 released")
    assert out == "Python 3.11.5 released"


def test_preview_spans_structured() -> None:
    spans = redact_preview_spans("mail tej@x.tech and ssn 123-45-6789")
    cats = {s.category for s in spans}
    assert "email" in cats
    assert "ssn" in cats


def test_multiple_redactions_preserve_non_sensitive_text() -> None:
    out = redact_text("Email tej@x.tech or call (555) 867-5309")
    assert out.startswith("Email ")
    assert "<REDACTED:email>" in out
    assert "<REDACTED:phone>" in out
    assert " or call " in out


def test_empty_input() -> None:
    assert redact_text("") == ""
    assert redact_preview_spans("") == []
