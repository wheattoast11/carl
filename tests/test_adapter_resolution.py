"""Regression tests for the v0.18 adapter/compute-backend split.

The v0.17.x schema conflated training-adapter and compute-orchestration on
a single ``backend`` field, so ``carl project init`` would write
``backend: hf_jobs`` (compute) and ``carl train`` would try to resolve it
in the adapter registry (fails). These tests pin the fixed resolution.
"""

from __future__ import annotations

from typing import Any

from carl_studio.cli.training import (
    _resolve_adapter_name,  # pyright: ignore[reportPrivateUsage]
)


def test_adapter_cli_flag_wins() -> None:
    """Explicit --backend flag beats everything."""
    raw: dict[str, Any] = {"adapter": "unsloth", "backend": "trl"}
    assert _resolve_adapter_name(cli_override="slime", raw=raw) == "slime"


def test_new_adapter_field_preferred_over_legacy_backend() -> None:
    """v0.17.1+ `adapter:` in carl.yaml wins over legacy `backend:`."""
    raw: dict[str, Any] = {"adapter": "unsloth", "backend": "trl"}
    assert _resolve_adapter_name(cli_override="", raw=raw) == "unsloth"


def test_legacy_backend_used_if_adapter_absent_and_backend_is_valid() -> None:
    """Old carl.yaml that correctly used `backend: trl` still works."""
    raw: dict[str, Any] = {"backend": "trl"}
    assert _resolve_adapter_name(cli_override="", raw=raw) == "trl"


def test_hf_jobs_in_legacy_backend_field_falls_through_to_default() -> None:
    """The bug this test exists for: ``backend: hf_jobs`` is compute, not an
    adapter. Previously this crashed carl train with 'unknown training backend'.
    Now it silently falls through to "trl" so the wizard's carl.yaml just works.
    """
    raw: dict[str, Any] = {"backend": "hf_jobs"}
    assert _resolve_adapter_name(cli_override="", raw=raw) == "trl"


def test_runpod_in_legacy_backend_field_falls_through() -> None:
    raw: dict[str, Any] = {"backend": "runpod"}
    assert _resolve_adapter_name(cli_override="", raw=raw) == "trl"


def test_local_in_legacy_backend_field_falls_through() -> None:
    raw: dict[str, Any] = {"backend": "local"}
    assert _resolve_adapter_name(cli_override="", raw=raw) == "trl"


def test_empty_config_defaults_to_trl() -> None:
    assert _resolve_adapter_name(cli_override="", raw={}) == "trl"


def test_adapter_field_lowercased() -> None:
    """Normalization: 'TRL' and 'Trl' both resolve to 'trl'."""
    assert _resolve_adapter_name(cli_override="", raw={"adapter": "TRL"}) == "trl"
    assert _resolve_adapter_name(cli_override="", raw={"adapter": "Slime"}) == "slime"


def test_cli_flag_with_whitespace() -> None:
    assert _resolve_adapter_name(cli_override=" Slime ", raw={}) == "slime"


def test_project_schema_defaults() -> None:
    """CARLProject instantiated bare uses the new split."""
    from carl_studio.project import CARLProject

    p = CARLProject()
    assert p.adapter == "trl"
    assert p.compute_backend == "local"
    assert p.compute_target == "local"


def test_project_to_training_config_carries_adapter() -> None:
    """to_training_config() threads the new split through."""
    from carl_studio.project import CARLProject

    p = CARLProject(adapter="slime", compute_backend="hf_jobs", compute_target="l40sx1")
    cfg = p.to_training_config()
    assert cfg["adapter"] == "slime"
    assert cfg["compute_backend"] == "hf_jobs"
    assert cfg["compute_target"] == "l40sx1"
