"""Regression tests for the dep-probe auto-heal branch in carl init.

Covers the three fan-out states of ``_offer_extras``:

- All healthy → fast path, records ``already_present``.
- One sibling-dep corruption (the HF scenario that motivated this) →
  auto-heal offered, targets the sibling (not the package that raised).
- Auto-heal declined → returns False cleanly, records the decline.

Plus a smoke test confirming the HF ValueError scenario is classified
end-to-end exactly as the live traceback showed.
"""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import patch

from carl_core.dependency_probe import DepProbeResult
from carl_core.interaction import InteractionChain

from carl_studio.cli import init as init_mod


def _ok(name: str, version: str = "1.0.0") -> DepProbeResult:
    return DepProbeResult(
        name=name,
        normalized_name=name.replace("_", "-").lower(),
        status="ok",
        import_ok=True,
        import_version=version,
        metadata_version=version,
        import_error=None,
        metadata_error=None,
        repair_command="",
    )


def _import_value_error(name: str, sibling_msg: str) -> DepProbeResult:
    return DepProbeResult(
        name=name,
        normalized_name=name.replace("_", "-").lower(),
        status="import_value_error",
        import_ok=False,
        import_version=None,
        metadata_version=None,
        import_error=f"ValueError: {sibling_msg}",
        metadata_error=None,
        repair_command=f"pip install --force-reinstall --no-deps {name}",
    )


def _metadata_corrupt(name: str) -> DepProbeResult:
    return DepProbeResult(
        name=name,
        normalized_name=name.replace("_", "-").lower(),
        status="metadata_corrupt",
        import_ok=True,
        import_version="1.9.2",
        metadata_version=None,
        import_error=None,
        metadata_error=None,
        repair_command=f"pip install --force-reinstall --no-deps {name}",
    )


# ---------------------------------------------------------------------------
# Fast path
# ---------------------------------------------------------------------------


def test_offer_extras_all_healthy_takes_fast_path() -> None:
    """Every probe returns ``ok`` → returns True, records already_present."""
    chain = InteractionChain()
    probes = [_ok("torch", "2.9.0"), _ok("transformers", "5.3.0"), _ok("huggingface-hub", "1.11.0")]
    with patch.object(init_mod, "_probe_training_extras", return_value=probes):
        result = init_mod._offer_extras(chain)  # pyright: ignore[reportPrivateUsage]
    assert result is True
    steps = [s for s in chain.steps if s.name == "install_extras"]
    assert len(steps) == 1
    step_input: Any = steps[0].input
    assert step_input == {"already_present": True}


# ---------------------------------------------------------------------------
# Auto-heal branch — the HF sibling scenario end-to-end
# ---------------------------------------------------------------------------


def test_offer_extras_auto_heal_targets_corrupt_sibling() -> None:
    """The HF scenario: import transformers raises ValueError about huggingface-hub.

    Auto-heal targets huggingface-hub (the sibling named in the error),
    NOT transformers. After repair the post-probes are healthy.
    """
    chain = InteractionChain()
    # First probe: transformers can't import due to corrupt hf
    first = [
        _ok("torch", "2.9.0"),
        _import_value_error(
            "transformers",
            "Unable to compare versions for huggingface-hub>=1.3.0,<2.0: "
            "need=1.3.0 found=None",
        ),
        _metadata_corrupt("huggingface_hub"),
    ]
    # Second probe (after repair): all healthy
    post = [_ok("torch", "2.9.0"), _ok("transformers", "5.3.0"), _ok("huggingface-hub", "1.11.0")]

    # Probe sibling resolution: extract_corrupt_sibling → "huggingface-hub",
    # then probe() is called to classify it; we return a corrupt result.
    sibling_probe = _metadata_corrupt("huggingface-hub")

    probe_side_effects = [sibling_probe]

    def _probe_by_name(name: str, *, import_name: str | None = None) -> DepProbeResult:
        del import_name
        return probe_side_effects.pop(0)

    with (
        patch.object(init_mod, "_probe_training_extras", side_effect=[first, post]),
        patch("carl_core.dependency_probe.probe", side_effect=_probe_by_name),
        patch("typer.confirm", return_value=True),
        patch.object(subprocess, "run", return_value=subprocess.CompletedProcess(args=[], returncode=0)) as run,
    ):
        result = init_mod._offer_extras(chain)  # pyright: ignore[reportPrivateUsage]

    assert result is True
    # subprocess.run was called with the sibling's repair command, not
    # transformers'. The command should start with pip install
    # --force-reinstall --no-deps huggingface-hub.
    invoked_args: list[list[str]] = [list(call.args[0]) for call in run.call_args_list]
    assert any("huggingface-hub" in args for args in invoked_args), invoked_args
    assert not any("transformers" in args for args in invoked_args), invoked_args

    # Auto-heal step recorded.
    repair_steps = [s for s in chain.steps if s.name == "pip_auto_heal"]
    assert len(repair_steps) == 1
    step_input: Any = repair_steps[0].input
    assert step_input == {"targets": ["huggingface-hub"]}


# ---------------------------------------------------------------------------
# Auto-heal declined
# ---------------------------------------------------------------------------


def test_offer_extras_auto_heal_declined_returns_false() -> None:
    """User says no to the repair → returns False, records the decline."""
    chain = InteractionChain()
    probes = [
        _ok("torch", "2.9.0"),
        _import_value_error(
            "transformers",
            "Unable to compare versions for huggingface-hub>=1.3.0,<2.0",
        ),
        _metadata_corrupt("huggingface_hub"),
    ]
    sibling_probe = _metadata_corrupt("huggingface-hub")

    with (
        patch.object(init_mod, "_probe_training_extras", return_value=probes),
        patch("carl_core.dependency_probe.probe", return_value=sibling_probe),
        patch("typer.confirm", return_value=False),
    ):
        result = init_mod._offer_extras(chain)  # pyright: ignore[reportPrivateUsage]

    assert result is False
    steps = [s for s in chain.steps if s.name == "install_extras"]
    assert len(steps) == 1
    step_input: Any = steps[0].input
    assert step_input == {
        "action": "auto_heal_declined",
        "targets": ["huggingface-hub"],
    }


# ---------------------------------------------------------------------------
# Fresh-install branch (plain "missing" case)
# ---------------------------------------------------------------------------


def test_offer_extras_missing_packages_takes_fresh_install_branch() -> None:
    """When no corruption is present but packages are missing, the current
    'install the training extra' flow still runs (user declines)."""
    chain = InteractionChain()
    missing = DepProbeResult(
        name="torch",
        normalized_name="torch",
        status="missing",
        import_ok=False,
        import_version=None,
        metadata_version=None,
        import_error="ImportError: No module named 'torch'",
        metadata_error=None,
        repair_command="pip install torch",
    )
    probes = [missing, missing, missing]
    with (
        patch.object(init_mod, "_probe_training_extras", return_value=probes),
        patch("typer.confirm", return_value=False),
    ):
        result = init_mod._offer_extras(chain)  # pyright: ignore[reportPrivateUsage]
    assert result is False
    steps = [s for s in chain.steps if s.name == "install_extras"]
    assert any(s.input == {"answer": "no"} for s in steps)
