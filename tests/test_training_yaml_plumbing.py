"""YAML plumbing for reproducibility: reward_class · deterministic · cascade.gate_mode.

Covers tickets B1 (reward-class selector), B2 (full determinism), and B3
(cascade gate_mode YAML plumbing). The goal is to prove that every
researcher-visible toggle survives the round trip:

    carl.yaml -> TrainingConfig -> CARLTrainer -> CascadeRewardManager
                                              -> make_carl_reward
                                              -> _apply_determinism

These are pure-Python tests — no torch or transformers required on the
test machine. Heavy deps are mocked at the import site.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from carl_studio.training.cascade import CascadeRewardManager
from carl_studio.training.rewards.composite import (
    CARLReward,
    PhaseAdaptiveCARLReward,
    make_carl_reward,
)
from carl_studio.training.trainer import _apply_determinism
from carl_studio.types.config import (
    CascadeConfig,
    ComputeTarget,
    TrainingConfig,
    TrainingMethod,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _base_config(**overrides: Any) -> TrainingConfig:
    """Minimal valid TrainingConfig with overrides applied."""
    defaults: dict[str, Any] = dict(
        run_name="yaml-plumbing-test",
        base_model="test/base",
        output_repo="test/out",
        method=TrainingMethod.GRPO,
        dataset_repo="test/dataset",
        compute_target=ComputeTarget.LOCAL,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


class _DummyModel:
    """No-op stand-in so make_carl_reward can close over something."""

    training = False
    device = "cpu"

    def eval(self) -> None: ...
    def train(self) -> None: ...


class _DummyTokenizer:
    vocab_size = 128


# ---------------------------------------------------------------------------
# B1 — reward_class selector dispatches CARLReward vs PhaseAdaptiveCARLReward
# ---------------------------------------------------------------------------


def _patch_torch_module() -> ModuleType:
    """Install a stub ``torch`` module so make_carl_reward can import it.

    make_carl_reward uses ``import torch`` at function-entry time (to attach
    @torch.no_grad to the returned closure). We don't exercise the closure
    in these tests — we only need the factory to construct — so a stub
    with a no-op no_grad decorator is enough.

    If torch is already in sys.modules, preserve the existing module — we
    only want to patch, not replace, so the rest of the suite keeps its
    torch bindings intact.
    """
    existing = sys.modules.get("torch")
    if existing is not None and hasattr(existing, "no_grad"):
        return existing
    mod = ModuleType("torch")

    def no_grad(fn: Any) -> Any:  # pragma: no cover - trivial passthrough
        return fn

    mod.no_grad = no_grad  # type: ignore[attr-defined]
    sys.modules["torch"] = mod
    return mod


def test_reward_class_static_dispatches_carl_reward() -> None:
    """reward_class='static' constructs a plain CARLReward internally."""
    _patch_torch_module()
    with patch.object(
        sys.modules["carl_studio.training.rewards.composite"],
        "CARLReward",
        wraps=CARLReward,
    ) as mock_static, patch.object(
        sys.modules["carl_studio.training.rewards.composite"],
        "PhaseAdaptiveCARLReward",
        wraps=PhaseAdaptiveCARLReward,
    ) as mock_par:
        fn = make_carl_reward(
            model=_DummyModel(),
            tokenizer=_DummyTokenizer(),
            reward_class="static",
        )
        assert callable(fn)
        assert mock_static.call_count == 1
        assert mock_par.call_count == 0


def test_reward_class_phase_adaptive_dispatches_par() -> None:
    """reward_class='phase_adaptive' constructs PhaseAdaptiveCARLReward."""
    _patch_torch_module()
    with patch.object(
        sys.modules["carl_studio.training.rewards.composite"],
        "CARLReward",
        wraps=CARLReward,
    ) as mock_static, patch.object(
        sys.modules["carl_studio.training.rewards.composite"],
        "PhaseAdaptiveCARLReward",
        wraps=PhaseAdaptiveCARLReward,
    ) as mock_par:
        fn = make_carl_reward(
            model=_DummyModel(),
            tokenizer=_DummyTokenizer(),
            reward_class="phase_adaptive",
        )
        assert callable(fn)
        assert mock_par.call_count == 1
        assert mock_static.call_count == 0


def test_reward_class_unknown_raises() -> None:
    """Typo in YAML should fail loudly, not silently fall back."""
    _patch_torch_module()
    with pytest.raises(ValueError, match="reward_class"):
        make_carl_reward(
            model=_DummyModel(),
            tokenizer=_DummyTokenizer(),
            reward_class="nonsense",  # type: ignore[arg-type]
        )


def test_reward_class_threads_through_training_config_yaml() -> None:
    """YAML -> TrainingConfig round-trip preserves reward_class."""
    cfg = _base_config(reward_class="phase_adaptive")
    assert cfg.reward_class == "phase_adaptive"


# ---------------------------------------------------------------------------
# B2 — determinism wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip CUBLAS / PYTHONHASHSEED so we can assert on insert vs leave-alone."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)


def _install_transformers_stub(
    monkeypatch: pytest.MonkeyPatch, set_seed_mock: MagicMock
) -> ModuleType:
    """Install or extend a ``transformers`` stub with a mocked set_seed.

    Uses monkeypatch so the original ``transformers`` binding (if any) is
    restored at test teardown. This prevents downstream tests — notably
    ``tests/test_gate.py`` which does ``from transformers import
    TrainerCallback`` — from picking up our stub.
    """
    existing = sys.modules.get("transformers")
    if existing is not None:
        # Patch set_seed on the existing module so we share its namespace
        # with whatever the rest of the suite expects to be present.
        monkeypatch.setattr(existing, "set_seed", set_seed_mock, raising=False)
        return existing
    # No transformers module cached — install a disposable stub.
    mod = ModuleType("transformers")
    mod.set_seed = set_seed_mock  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return mod


def _install_torch_stub(
    monkeypatch: pytest.MonkeyPatch,
    manual_seed: MagicMock,
    use_det: MagicMock,
) -> ModuleType:
    """Install or extend a ``torch`` stub with mocked determinism functions.

    Uses monkeypatch.setattr (with raising=False so we can add missing
    attrs on a real torch module) for clean teardown.
    """
    existing = sys.modules.get("torch")
    if existing is not None:
        monkeypatch.setattr(existing, "manual_seed", manual_seed, raising=False)
        monkeypatch.setattr(
            existing, "use_deterministic_algorithms", use_det, raising=False
        )
        return existing
    mod = ModuleType("torch")
    mod.manual_seed = manual_seed  # type: ignore[attr-defined]
    mod.use_deterministic_algorithms = use_det  # type: ignore[attr-defined]

    def no_grad(fn: Any) -> Any:  # pragma: no cover - trivial passthrough
        return fn

    mod.no_grad = no_grad  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod


def test_deterministic_sets_cublas_and_pythonhashseed(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    """deterministic=True pins CUBLAS_WORKSPACE_CONFIG, PYTHONHASHSEED, and seeds."""
    set_seed_mock = MagicMock()
    manual_seed_mock = MagicMock()
    use_det_mock = MagicMock()
    _install_transformers_stub(monkeypatch, set_seed_mock)
    _install_torch_stub(monkeypatch, manual_seed_mock, use_det_mock)

    cfg = _base_config(seed=123, deterministic=True)
    _apply_determinism(cfg)

    # transformers.set_seed ALWAYS fires when transformers is importable —
    # even when deterministic=False — because it covers TRL's seed arg
    # alone (the status quo) is insufficient.
    set_seed_mock.assert_called_once_with(123)
    # Env-level determinism: these are what cuBLAS + interpreter demand.
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert os.environ.get("PYTHONHASHSEED") == "123"
    # torch + numpy pinning with the same seed
    manual_seed_mock.assert_called_once_with(123)
    use_det_mock.assert_called_once_with(True, warn_only=True)


def test_deterministic_false_leaves_env_untouched(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    """deterministic=False only primes transformers.set_seed — no env mutation."""
    set_seed_mock = MagicMock()
    manual_seed_mock = MagicMock()
    use_det_mock = MagicMock()
    _install_transformers_stub(monkeypatch, set_seed_mock)
    _install_torch_stub(monkeypatch, manual_seed_mock, use_det_mock)

    cfg = _base_config(seed=7, deterministic=False)
    _apply_determinism(cfg)

    set_seed_mock.assert_called_once_with(7)
    # No env pins, no torch pins — the non-deterministic path is a
    # pure seed call to cover TRL's randomness reach.
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None
    assert os.environ.get("PYTHONHASHSEED") is None
    manual_seed_mock.assert_not_called()
    use_det_mock.assert_not_called()


def test_deterministic_without_transformers_is_noop(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    """When transformers is absent, _apply_determinism silently returns."""
    # setitem with None marks the module as explicitly unimportable so
    # ``from transformers import set_seed`` raises ImportError — monkeypatch
    # restores the prior binding at teardown.
    monkeypatch.setitem(sys.modules, "transformers", None)
    cfg = _base_config(seed=42, deterministic=True)
    # No exception, no env mutation.
    _apply_determinism(cfg)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None
    assert os.environ.get("PYTHONHASHSEED") is None


def test_deterministic_threads_through_training_config_yaml() -> None:
    """TrainingConfig round-trip preserves the deterministic flag."""
    cfg = _base_config(deterministic=True)
    assert cfg.deterministic is True
    cfg2 = _base_config()
    assert cfg2.deterministic is False  # default


# ---------------------------------------------------------------------------
# B3 — cascade.gate_mode YAML plumbing
# ---------------------------------------------------------------------------


def test_cascade_config_defaults_match_fixed_mode_behavior() -> None:
    """Default CascadeConfig preserves pre-B3 fixed-mode behavior."""
    cfg = CascadeConfig()
    assert cfg.carl_start == 50
    assert cfg.warmup_steps == 10
    assert cfg.gate_mode == "metric"
    assert cfg.n_crystallizations_required == 3
    assert cfg.crystallization_window == 10


def test_cascade_gate_mode_propagates_from_config() -> None:
    """Every nested field reaches CascadeRewardManager constructor args."""
    cfg = _base_config(
        cascade={
            "carl_start": 25,
            "warmup_steps": 5,
            "gate_mode": "crystallization",
            "n_crystallizations_required": 7,
            "crystallization_window": 20,
        }
    )
    assert cfg.cascade.gate_mode == "crystallization"
    assert cfg.cascade.n_crystallizations_required == 7

    # Construct a manager with these fields — this is what the trainer does.
    manager = CascadeRewardManager(
        carl_start=cfg.cascade.carl_start,
        warmup_steps=cfg.cascade.warmup_steps,
        gate_mode=cfg.cascade.gate_mode,
        n_crystallizations_required=cfg.cascade.n_crystallizations_required,
        crystallization_window=cfg.cascade.crystallization_window,
    )
    # Crystallization mode flips the cascade to adaptive regardless of
    # carl_start — this is the CascadeRewardManager semantic we are
    # relying on, so assert it here to catch regressions.
    assert manager._mode == "adaptive"
    assert manager._gate_mode == "crystallization"
    assert manager._n_crystallizations_required == 7
    assert manager._crystallization_window == 20
    assert manager.warmup_steps == 5


def test_cascade_gate_mode_metric_is_still_default() -> None:
    """When cascade is omitted entirely, gate stays in metric (fixed) mode."""
    cfg = _base_config()
    assert cfg.cascade.gate_mode == "metric"
    # A static reward_class + metric gate = the pre-B3 status quo.
    assert cfg.reward_class == "static"


def test_phase_adaptive_auto_selects_crystallization_gate_when_unset() -> None:
    """reward_class=phase_adaptive + no cascade.gate_mode => crystallization."""
    cfg = _base_config(reward_class="phase_adaptive")
    assert cfg.cascade.gate_mode == "crystallization"


def test_phase_adaptive_auto_selects_when_cascade_partially_specified() -> None:
    """Partial cascade dict (no gate_mode) still gets the crystallization default."""
    cfg = _base_config(
        reward_class="phase_adaptive",
        cascade={"carl_start": 30, "warmup_steps": 5},
    )
    assert cfg.cascade.carl_start == 30
    assert cfg.cascade.warmup_steps == 5
    assert cfg.cascade.gate_mode == "crystallization"


def test_explicit_metric_gate_overrides_auto_default() -> None:
    """User explicitly pinning metric gate wins over the phase_adaptive heuristic."""
    cfg = _base_config(
        reward_class="phase_adaptive",
        cascade={"gate_mode": "metric"},
    )
    assert cfg.reward_class == "phase_adaptive"
    assert cfg.cascade.gate_mode == "metric"


def test_explicit_crystallization_gate_with_static_reward_is_allowed() -> None:
    """User may mix static reward + crystallization gate (unusual but legal)."""
    cfg = _base_config(cascade={"gate_mode": "crystallization"})
    assert cfg.reward_class == "static"
    assert cfg.cascade.gate_mode == "crystallization"


def test_prebuilt_cascade_instance_is_respected_verbatim() -> None:
    """A pre-built CascadeConfig instance takes precedence over the heuristic.

    Passing ``cascade=CascadeConfig(gate_mode="metric")`` from Python
    code should preserve the explicit metric gate even if the user
    also sets ``reward_class="phase_adaptive"``. The auto-selector
    only fires on raw dicts / missing cascade values — it must not
    mutate caller-constructed instances.
    """
    explicit = CascadeConfig(gate_mode="metric", carl_start=42)
    cfg = _base_config(reward_class="phase_adaptive", cascade=explicit)
    assert cfg.cascade.gate_mode == "metric"
    assert cfg.cascade.carl_start == 42


# ---------------------------------------------------------------------------
# Integration — end-to-end plumbing through _build_rewards
# ---------------------------------------------------------------------------


def test_trainer_build_rewards_threads_full_cascade_config() -> None:
    """CARLTrainer._build_rewards must honor every new YAML field.

    Smoke test for the integration contract: every knob added to
    CascadeConfig in B3 ends up as a constructor kwarg on the manager
    that the GRPO loop will actually use. If a future refactor drops
    one of these, this test catches it before training breaks.
    """
    from carl_studio.training.trainer import CARLTrainer

    cfg = _base_config(
        reward_class="phase_adaptive",
        cascade={
            "carl_start": 80,
            "warmup_steps": 15,
            "n_crystallizations_required": 4,
            "crystallization_window": 12,
            # gate_mode intentionally omitted -> auto-crystallization
        },
    )
    trainer = CARLTrainer(cfg, skip_credits=True)

    _patch_torch_module()
    with patch(
        "carl_studio.training.rewards.composite.make_carl_reward",
        wraps=make_carl_reward,
    ) as spy_make_carl:
        rewards = trainer._build_rewards(_DummyModel(), _DummyTokenizer())

    # _build_rewards must hand back a non-empty reward list and bind the
    # cascade manager on the trainer.
    assert len(rewards) > 0
    assert trainer._cascade_manager is not None
    mgr = trainer._cascade_manager
    assert mgr._gate_mode == "crystallization"
    assert mgr._n_crystallizations_required == 4
    assert mgr._crystallization_window == 12
    assert mgr.warmup_steps == 15

    # make_carl_reward was called with the explicit reward_class.
    kwargs = spy_make_carl.call_args.kwargs
    assert kwargs.get("reward_class") == "phase_adaptive"


# ---------------------------------------------------------------------------
# TRL training-args builder: full_determinism fallback path
# ---------------------------------------------------------------------------


def test_build_sft_args_passes_full_determinism_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Builder passes full_determinism=True when the TRL signature accepts it."""
    from carl_studio.training.trainer import CARLTrainer

    _install_transformers_stub(monkeypatch, MagicMock())

    calls: dict[str, Any] = {}

    def fake_sft_config(**kwargs: Any) -> SimpleNamespace:
        calls.update(kwargs)
        return SimpleNamespace(**kwargs)

    cfg = _base_config(method=TrainingMethod.SFT, deterministic=True, seed=99)
    trainer = CARLTrainer(cfg, skip_credits=True)
    result = trainer._build_sft_training_args(
        SFTConfig=fake_sft_config,
        output_dir="out",
        hf_token=None,
    )
    assert calls.get("full_determinism") is True
    assert calls.get("seed") == 99
    assert result.output_dir == "out"


def test_build_grpo_args_falls_back_when_full_determinism_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older TRLs without full_determinism raise TypeError — we swallow + retry."""
    from carl_studio.training.trainer import CARLTrainer

    _install_transformers_stub(monkeypatch, MagicMock())

    attempts: list[dict[str, Any]] = []

    def fake_grpo_config(**kwargs: Any) -> SimpleNamespace:
        attempts.append(kwargs)
        if "full_determinism" in kwargs:
            raise TypeError("unexpected keyword 'full_determinism'")
        return SimpleNamespace(**kwargs)

    cfg = _base_config(method=TrainingMethod.GRPO, deterministic=True)
    trainer = CARLTrainer(cfg, skip_credits=True)
    result = trainer._build_grpo_training_args(
        GRPOConfig=fake_grpo_config,
        output_dir="out",
        hf_token=None,
    )
    assert len(attempts) == 2
    assert "full_determinism" in attempts[0]
    assert "full_determinism" not in attempts[1]
    assert result.output_dir == "out"


def test_build_sft_args_skips_full_determinism_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """deterministic=False never attempts the full_determinism kwarg."""
    from carl_studio.training.trainer import CARLTrainer

    _install_transformers_stub(monkeypatch, MagicMock())

    calls: list[dict[str, Any]] = []

    def fake_sft_config(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    cfg = _base_config(method=TrainingMethod.SFT, deterministic=False)
    trainer = CARLTrainer(cfg, skip_credits=True)
    trainer._build_sft_training_args(
        SFTConfig=fake_sft_config,
        output_dir="out",
        hf_token=None,
    )
    assert len(calls) == 1
    assert "full_determinism" not in calls[0]
