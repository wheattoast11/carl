"""Tests for the v0.17 admin-gate seam.

Covers:
- `admin.load_private` prefers local resonance package (fast path) over HF
  dataset (fallback path).
- `AdminToken` issue + verify + expiry + hardware-fingerprint mismatch.
- `Vault.register_runtime_resolver` accepts + verifies an admin_token via
  duck-typed `.verify()`.

`admin.issue_token()` requires admin unlock; tests unlock via monkeypatching
the `is_admin` function rather than writing real machine keys.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest

from carl_core.errors import ValidationError
from carl_core.secrets import SecretRef, SecretVault
from carl_core.vault import VaultError

from carl_studio import admin


# ---------------------------------------------------------------------------
# load_private — resolution order
# ---------------------------------------------------------------------------


def test_load_private_requires_admin_unlock() -> None:
    with patch.object(admin, "is_admin", return_value=False):
        with pytest.raises(ImportError) as exc:
            admin.load_private("signals.constitutional")
    assert "admin unlock" in str(exc.value).lower() or "resonance" in str(exc.value).lower()


def test_load_private_prefers_local_resonance(monkeypatch: pytest.MonkeyPatch) -> None:
    """When admin is unlocked, load_private tries `resonance.<name>` first."""
    monkeypatch.setattr(admin, "is_admin", lambda: True)

    # Inject a fake `resonance.test_module` into sys.modules
    pkg = types.ModuleType("resonance")
    pkg.__path__ = []  # type: ignore[attr-defined]
    sub = types.ModuleType("resonance.test_module_xyz")
    sub.PAYLOAD = "from-local-resonance"  # type: ignore[attr-defined]
    sys.modules["resonance"] = pkg
    sys.modules["resonance.test_module_xyz"] = sub
    try:
        mod = admin.load_private("test_module_xyz")
        assert getattr(mod, "PAYLOAD", None) == "from-local-resonance"
    finally:
        sys.modules.pop("resonance.test_module_xyz", None)
        sys.modules.pop("resonance", None)


def test_load_private_rejects_dotted_path_without_local_resonance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dotted module names require the local resonance package — HF dataset
    has flat files, so we refuse the lookup early with a clear message."""
    monkeypatch.setattr(admin, "is_admin", lambda: True)
    # Ensure resonance isn't in sys.modules
    sys.modules.pop("resonance", None)
    sys.modules.pop("resonance.signals", None)
    sys.modules.pop("resonance.signals.does_not_exist", None)

    import importlib.util

    if importlib.util.find_spec("resonance") is not None:
        pytest.skip(
            "real resonance package installed; local-path will succeed via "
            "importlib regardless of our test intent"
        )

    with pytest.raises(ImportError) as exc:
        admin.load_private("signals.does_not_exist")
    assert "flat module names" in str(exc.value) or "resonance package" in str(exc.value)


# ---------------------------------------------------------------------------
# AdminToken
# ---------------------------------------------------------------------------


def test_issue_token_requires_admin_unlock() -> None:
    with patch.object(admin, "is_admin", return_value=False):
        with pytest.raises(ImportError) as exc:
            admin.issue_token()
    assert "admin unlock" in str(exc.value).lower()


def test_issue_token_returns_fresh_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admin, "is_admin", lambda: True)
    token = admin.issue_token()
    assert token.hw_fingerprint_prefix
    assert len(token.hw_fingerprint_prefix) == 12
    assert not token.is_expired()


def test_admin_token_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admin, "is_admin", lambda: True)
    token = admin.issue_token()
    # Simulate 2 hours later
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    assert token.is_expired(now=future)
    # verify() raises
    class _Frozen:
        @staticmethod
        def now(tz: Any = None) -> datetime:
            return future

    with patch("carl_studio.admin.datetime") as m_dt:
        m_dt.now = _Frozen.now
        with pytest.raises(ImportError) as exc:
            token.verify()
        assert "expired" in str(exc.value).lower()


def test_admin_token_hw_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admin, "is_admin", lambda: True)
    token = admin.issue_token()
    # Stub _hw_fingerprint to return a different value
    monkeypatch.setattr(admin, "_hw_fingerprint", lambda: b"\xaa" * 32)
    with pytest.raises(ImportError) as exc:
        token.verify()
    assert "hardware fingerprint mismatch" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Vault.register_runtime_resolver
# ---------------------------------------------------------------------------


def test_register_runtime_resolver_without_token_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from carl_studio.handles.resolvers import EnvResolver

    monkeypatch.setenv("CARL_TEST_RT_VAR", "from-rt")
    vault = SecretVault()
    vault.register_runtime_resolver("env", EnvResolver())
    ref = SecretRef(kind="env", uri="env://CARL_TEST_RT_VAR")
    vault.put_ref_only(ref)
    got = vault.resolve(ref, privileged=True)
    assert got == b"from-rt"


def test_register_runtime_resolver_verifies_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from carl_studio.handles.resolvers import EnvResolver

    monkeypatch.setattr(admin, "is_admin", lambda: True)
    token = admin.issue_token()

    monkeypatch.setenv("CARL_TEST_RT_VAR2", "value")
    vault = SecretVault()
    # Happy path — token verifies
    vault.register_runtime_resolver("env", EnvResolver(), admin_token=token)
    ref = SecretRef(kind="env", uri="env://CARL_TEST_RT_VAR2")
    vault.put_ref_only(ref)
    assert vault.resolve(ref, privileged=True) == b"value"


def test_register_runtime_resolver_rejects_stale_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(admin, "is_admin", lambda: True)
    token = admin.issue_token()

    # Stale: simulate hardware-fingerprint mismatch
    monkeypatch.setattr(admin, "_hw_fingerprint", lambda: b"\xbb" * 32)
    vault = SecretVault()
    with pytest.raises(ImportError) as exc:
        vault.register_runtime_resolver(
            "env",
            lambda ref: b"x",
            admin_token=token,
        )
    assert "hardware fingerprint mismatch" in str(exc.value).lower()


def test_register_runtime_resolver_rejects_bad_token_shape() -> None:
    vault = SecretVault()
    # Object without .verify() method
    class _FakeToken:
        pass

    with pytest.raises(ValidationError) as exc:
        vault.register_runtime_resolver(
            "env",
            lambda ref: b"x",
            admin_token=_FakeToken(),
        )
    assert exc.value.code == "carl.secrets.invalid_admin_token"


def test_register_runtime_resolver_rejects_non_callable_resolver() -> None:
    vault = SecretVault()
    with pytest.raises(ValidationError) as exc:
        vault.register_runtime_resolver("env", "not-a-callable")  # type: ignore[arg-type]
    # Falls through to register_resolver's check
    assert exc.value.code == "carl.secrets.invalid_resolver"


# ---------------------------------------------------------------------------
# End-to-end: runtime-registered resolver works across all three vaults
# ---------------------------------------------------------------------------


def test_runtime_resolver_on_resource_vault(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admin, "is_admin", lambda: True)
    token = admin.issue_token()

    from carl_core.resource_handles import ResourceRef, ResourceVault

    vault = ResourceVault()

    def mock_rollout_resolver(ref: Any) -> Any:
        return {"simulated": "engine", "uri": ref.uri}

    vault.register_runtime_resolver(
        "rollout_engine",
        mock_rollout_resolver,
        admin_token=token,
    )
    ref = ResourceRef(
        kind="rollout_engine",
        provider="sglang",
        uri="sglang://endpoint/1",
    )
    vault.put_ref_only(ref)
    got = vault.resolve(ref, privileged=True)
    assert got == {"simulated": "engine", "uri": "sglang://endpoint/1"}


# Silence unused import (kept for completeness of the test module)
_ = VaultError
