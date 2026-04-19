"""Secret-redaction tests for ``carl_core.interaction._json_safe``.

The chain persists durably, so any secret that reaches ``Step.input`` /
``Step.output`` / ``InteractionChain.context`` must be scrubbed out on
serialization. These tests exercise every shape the redactor recognizes
and confirm that non-secret payloads pass through unchanged.
"""
from __future__ import annotations

from carl_core.interaction import (
    ActionType,
    InteractionChain,
    Step,
    _json_safe,
    _scrub_secrets,
)


# ---------------------------------------------------------------------------
# String-shape redaction
# ---------------------------------------------------------------------------


class TestScrubSecrets:
    def test_jwt_scrubbed(self) -> None:
        jwt = (
            "eyJhbGciOiJIUzI1NiJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        out = _scrub_secrets(f"bearer {jwt} in log")
        assert "SflKxw" not in out, "JWT signature must not leak"
        assert "eyJhbG<redacted>" in out

    def test_openai_key_scrubbed(self) -> None:
        text = "use sk-proj1234567890abcdefghijklmn for openai"
        out = _scrub_secrets(text)
        assert "sk-pro<redacted>" in out
        assert "1234567890abcdef" not in out

    def test_anthropic_key_scrubbed_keeps_sk_ant(self) -> None:
        text = "OPENAI=sk-abc ANTHROPIC=sk-ant-api03-XyZaBcDeFgHiJkLmNoPqRsTuVwXyZ"
        out = _scrub_secrets(text)
        # The sk-ant- prefix should survive in the redacted preview, not
        # be truncated to plain "sk-".
        assert "sk-ant<redacted>" in out

    def test_hf_token_scrubbed(self) -> None:
        text = "token=hf_AbCdEfGhIjKlMnOpQrStUv in env"
        out = _scrub_secrets(text)
        assert "hf_AbC<redacted>" in out
        assert "QrStUv" not in out

    def test_evm_address_scrubbed(self) -> None:
        text = "pay 0x742d35Cc6634C0532925a3b844Bc454e4438f44e now"
        out = _scrub_secrets(text)
        assert "0x742d<redacted>" in out
        assert "Bc454e4438f44e" not in out

    def test_non_secret_text_unchanged(self) -> None:
        text = "user input with no credentials mentioned"
        assert _scrub_secrets(text) == text

    def test_empty_and_short_strings_safe(self) -> None:
        assert _scrub_secrets("") == ""
        assert _scrub_secrets("short") == "short"


# ---------------------------------------------------------------------------
# Step serialization — the public surface
# ---------------------------------------------------------------------------


def _step(**kwargs: object) -> Step:
    defaults: dict[str, object] = dict(
        action=ActionType.TOOL_CALL,
        name="test",
    )
    defaults.update(kwargs)
    return Step(**defaults)  # type: ignore[arg-type]


class TestStepRedaction:
    def test_step_redacts_jwt_in_input_string(self) -> None:
        jwt = (
            "eyJhbGciOiJIUzI1NiJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "abcdef1234567890ABCDEF"
        )
        step = _step(input=f"dispatch_cli --auth {jwt}")
        out = step.to_dict()
        assert "eyJhbG<redacted>" in out["input"]
        assert "abcdef1234567890ABCDEF" not in out["input"]

    def test_step_redacts_sensitive_dict_key_value(self) -> None:
        step = _step(
            input={
                "api_key": "super-secret-plaintext-123",
                "hint": "openai production account",
            }
        )
        out = step.to_dict()
        assert out["input"]["api_key"] == "<redacted>"
        assert out["input"]["hint"] == "openai production account"

    def test_step_redacts_nested_dict(self) -> None:
        step = _step(
            input={
                "req": {
                    "headers": {
                        "Authorization": "Bearer sk-ant-api03-XyZaBcDeFgHiJkLmNo",
                    },
                    "body": {"question": "hello"},
                },
            }
        )
        out = step.to_dict()
        assert out["input"]["req"]["headers"]["Authorization"] == "<redacted>"
        assert out["input"]["req"]["body"]["question"] == "hello"

    def test_step_preserves_non_secret_text(self) -> None:
        payload = {"name": "alice", "score": 0.92, "ok": True}
        step = _step(input=payload, output="all good, no secrets here")
        out = step.to_dict()
        assert out["input"] == payload
        assert out["output"] == "all good, no secrets here"

    def test_step_redacts_openai_key_sk_prefix(self) -> None:
        step = _step(
            output="error: invalid key sk-proj1234567890abcdefghijklmn"
        )
        out = step.to_dict()
        assert "sk-pro<redacted>" in out["output"]

    def test_step_redacts_hf_token(self) -> None:
        step = _step(
            input={"note": "HF_TOKEN=hf_AbCdEfGhIjKlMnOpQrStUv exported"}
        )
        out = step.to_dict()
        assert "hf_AbC<redacted>" in out["input"]["note"]

    def test_step_redacts_evm_wallet_address(self) -> None:
        step = _step(
            input={"wallet": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"}
        )
        out = step.to_dict()
        # 'wallet' isn't a sensitive *name*, so redaction must happen on the
        # string shape rather than on the key.
        assert "0x742d<redacted>" in out["input"]["wallet"]
        assert "Bc454e4438f44e" not in out["input"]["wallet"]

    def test_chain_context_is_also_redacted(self) -> None:
        chain = InteractionChain(
            context={"Authorization": "Bearer eyJabcdefghijklmnopqrstuvwxyz.abc.def"}
        )
        out = chain.to_dict()
        assert out["context"]["Authorization"] == "<redacted>"

    def test_json_safe_redacts_lists_of_dicts(self) -> None:
        payload = [
            {"user": "alice", "token": "hf_aaaaaaaaaaaaaaaaaaaaaa"},
            {"user": "bob", "notes": "no secrets"},
        ]
        out = _json_safe(payload)
        assert out[0]["token"] == "<redacted>"
        assert out[0]["user"] == "alice"
        assert out[1]["notes"] == "no secrets"


# ---------------------------------------------------------------------------
# Roundtrip robustness — the redacted form must still JSON-encode.
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_step_to_dict_is_still_json_encodable(self) -> None:
        import json

        step = _step(
            input={
                "key": "s3cr3t",
                "free_text": "call us at 0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            },
            output=[{"secret": "abc"}, {"visible": "ok"}],
        )
        encoded = json.dumps(step.to_dict())
        decoded = json.loads(encoded)
        assert decoded["input"]["key"] == "<redacted>"
        assert decoded["output"][0]["secret"] == "<redacted>"
        assert decoded["output"][1]["visible"] == "ok"
