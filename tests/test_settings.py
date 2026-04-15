"""Tests for carl_studio.settings and carl_studio.tier."""

import pytest
import yaml

import carl_studio.tier as tier_mod
import carl_studio.settings as settings_mod
from carl_studio.tier import (
    Tier,
    FEATURE_TIERS,
    TierGateError,
    detect_effective_tier,
    feature_tier,
    tier_allows,
    tier_gate,
    tier_message,
)
from carl_studio.settings import (
    CARLSettings,
    Preset,
    _mask,
    _serialize_enums,
    set_field,
)
from carl_studio.types.config import ComputeTarget


# ---------------------------------------------------------------------------
# Tier ordering
# ---------------------------------------------------------------------------


class TestTierOrdering:
    def test_free_lt_pro(self):
        assert Tier.FREE < Tier.PRO

    def test_pro_lt_enterprise(self):
        assert Tier.PRO == Tier.ENTERPRISE == Tier.PAID

    def test_enterprise_ge_pro(self):
        assert Tier.ENTERPRISE >= Tier.PRO

    def test_free_le_free(self):
        assert Tier.FREE <= Tier.FREE

    def test_pro_not_gt_enterprise(self):
        assert not Tier.PRO > Tier.ENTERPRISE


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------


class TestFeatureTiers:
    def test_observe_is_free(self):
        assert feature_tier("observe") == Tier.FREE

    def test_train_is_free(self):
        assert feature_tier("train") == Tier.FREE

    def test_mcp_is_enterprise(self):
        assert feature_tier("mcp") == Tier.ENTERPRISE

    def test_unknown_feature_defaults_free(self):
        assert feature_tier("nonexistent_feature_xyz") == Tier.FREE

    def test_tier_allows_free_observe(self):
        assert tier_allows(Tier.FREE, "observe") is True

    def test_tier_allows_free_train(self):
        assert tier_allows(Tier.FREE, "train") is True

    def test_tier_allows_pro_train(self):
        assert tier_allows(Tier.PRO, "train") is True

    def test_tier_allows_enterprise_everything(self):
        for feature in FEATURE_TIERS:
            assert tier_allows(Tier.ENTERPRISE, feature) is True


# ---------------------------------------------------------------------------
# Auto-elevation
# ---------------------------------------------------------------------------


class TestAutoElevation:
    def test_no_credentials_stays_free(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(tier_mod, "_detect_hf_token", lambda: None)
        assert detect_effective_tier(Tier.FREE) == Tier.FREE

    def test_provider_keys_do_not_upgrade_to_paid(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
        monkeypatch.setattr(tier_mod, "_detect_hf_token", lambda: "hf_test_token")
        assert detect_effective_tier(Tier.FREE) == Tier.FREE

    def test_cached_paid_session_elevates_free(self, monkeypatch):
        import carl_studio.db as db_mod

        class FakeDB:
            def get_auth(self, key: str):
                return "paid" if key == "tier" else None

        monkeypatch.setattr(db_mod, "LocalDB", FakeDB)
        assert detect_effective_tier(Tier.FREE) == Tier.PAID

    def test_never_downgrades(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(tier_mod, "_detect_hf_token", lambda: None)
        assert detect_effective_tier(Tier.PRO) == Tier.PRO


# ---------------------------------------------------------------------------
# Tier message
# ---------------------------------------------------------------------------


class TestTierMessage:
    def test_allowed_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            tier_mod,
            "check_tier",
            lambda f: (True, Tier.PRO, Tier.FREE),
        )
        assert tier_message("observe") is None

    def test_denied_returns_message(self, monkeypatch):
        monkeypatch.setattr(
            tier_mod,
            "check_tier",
            lambda f: (False, Tier.FREE, Tier.PRO),
        )
        msg = tier_message("train")
        assert msg is not None
        assert "CARL Paid" in msg
        assert "carl.camp/pricing" in msg


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


class TestSettings:
    def test_defaults(self):
        s = CARLSettings()
        assert s.tier == Tier.FREE
        assert s.preset == Preset.CUSTOM
        assert s.default_compute == ComputeTarget.L40SX1
        assert s.default_model == "Tesslate/OmniCoder-9B"
        assert s.log_level == "info"

    def test_log_level_validation(self):
        with pytest.raises(Exception):
            CARLSettings(log_level="INVALID")

    def test_log_level_case_insensitive(self):
        s = CARLSettings(log_level="DEBUG")
        assert s.log_level == "debug"

    def test_observe_defaults(self):
        s = CARLSettings()
        assert s.observe_defaults.show_entropy is True
        assert s.observe_defaults.show_phi is True
        assert s.observe_defaults.default_poll_interval == 2.0

    def test_display_dict_masks_secrets(self):
        s = CARLSettings(hf_token="hf_test_1234567890", anthropic_api_key=None)
        d = s.display_dict(mask_secrets=True)
        assert "****" in d["hf_token"]
        assert d["anthropic_api_key"] == "(not set)"

    def test_display_dict_unmasks(self):
        s = CARLSettings(hf_token="hf_test_token")
        d = s.display_dict(mask_secrets=False)
        assert d["hf_token"] == "hf_test_token"


# ---------------------------------------------------------------------------
# Preset application
# ---------------------------------------------------------------------------


class TestPresets:
    def test_research_preset_sets_debug(self):
        s = CARLSettings(preset=Preset.RESEARCH)
        assert s.log_level == "debug"
        assert s.default_compute == ComputeTarget.A100_LARGE

    def test_quick_preset_sets_l4(self):
        s = CARLSettings(preset=Preset.QUICK)
        assert s.default_compute == ComputeTarget.L4X1

    def test_production_preset_sets_warning(self):
        s = CARLSettings(preset=Preset.PRODUCTION)
        assert s.log_level == "warning"

    def test_explicit_overrides_preset(self):
        # If user explicitly sets log_level, preset shouldn't override it
        s = CARLSettings(preset=Preset.RESEARCH, log_level="error")
        # The model_validator only overrides defaults, but since we pass
        # log_level explicitly, it should keep "error"
        assert s.log_level == "error"


# ---------------------------------------------------------------------------
# YAML persistence
# ---------------------------------------------------------------------------


class TestYAMLPersistence:
    def test_save_and_load(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        s = CARLSettings(
            tier=Tier.PRO,
            default_model="test/model",
            log_level="debug",
            naming_prefix="test-prefix",
        )
        s.save(config_path)

        assert config_path.is_file()
        with open(config_path) as f:
            data = yaml.safe_load(f)

        assert data["tier"] == "paid"
        assert data["default_model"] == "test/model"
        assert data["log_level"] == "debug"

    def test_save_excludes_secrets(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        s = CARLSettings(hf_token="secret", anthropic_api_key="also_secret")
        s.save(config_path)

        with open(config_path) as f:
            data = yaml.safe_load(f)

        assert "hf_token" not in data
        assert "anthropic_api_key" not in data

    def test_from_yaml(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        data = {
            "tier": "pro",
            "default_model": "my/model",
            "log_level": "warning",
        }
        with open(config_path, "w") as f:
            yaml.dump(data, f)

        s = CARLSettings.from_yaml(config_path)
        assert s.tier == Tier.PAID
        assert s.default_model == "my/model"
        assert s.log_level == "warning"

    def test_from_yaml_invalid_returns_defaults(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("not: valid: yaml: [")
        # Should handle gracefully
        s = CARLSettings.from_yaml(config_path)
        assert s.tier == Tier.FREE


# ---------------------------------------------------------------------------
# set_field
# ---------------------------------------------------------------------------


class TestSetField:
    def test_set_tier(self):
        s = CARLSettings()
        s = set_field(s, "tier", "pro")
        assert s.tier == Tier.PAID

    def test_set_log_level(self):
        s = CARLSettings()
        s = set_field(s, "log_level", "debug")
        assert s.log_level == "debug"

    def test_set_default_compute(self):
        s = CARLSettings()
        s = set_field(s, "default_compute", "a100-large")
        assert s.default_compute == ComputeTarget.A100_LARGE

    def test_set_nested_observe(self):
        s = CARLSettings()
        s = set_field(s, "observe_defaults.show_entropy", "false")
        assert s.observe_defaults.show_entropy is False

    def test_set_nested_poll_interval(self):
        s = CARLSettings()
        s = set_field(s, "observe_defaults.default_poll_interval", "5.0")
        assert s.observe_defaults.default_poll_interval == 5.0

    def test_set_invalid_key_raises(self):
        s = CARLSettings()
        with pytest.raises(ValueError, match="Unknown setting"):
            set_field(s, "nonexistent_key", "value")

    def test_set_invalid_tier_raises(self):
        s = CARLSettings()
        with pytest.raises(ValueError, match="Invalid tier"):
            set_field(s, "tier", "platinum")

    def test_set_preset(self):
        s = CARLSettings()
        s = set_field(s, "preset", "research")
        assert s.preset == Preset.RESEARCH


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


class TestMasking:
    def test_mask_none(self):
        assert _mask(None) == "(not set)"

    def test_mask_empty(self):
        assert _mask("") == "(not set)"

    def test_mask_short(self):
        assert _mask("abcd") == "****"

    def test_mask_long(self):
        result = _mask("hf_1234567890abcdef")
        assert result.startswith("hf_1")
        assert result.endswith("cdef")
        assert "****" in result


# ---------------------------------------------------------------------------
# Enum serialization
# ---------------------------------------------------------------------------


class TestEnumSerialization:
    def test_serialize_flat(self):
        data = {"tier": Tier.PRO, "name": "test"}
        _serialize_enums(data)
        assert data["tier"] == "paid"
        assert data["name"] == "test"

    def test_serialize_nested(self):
        data = {"outer": {"compute": ComputeTarget.A100_LARGE}}
        _serialize_enums(data)
        assert data["outer"]["compute"] == "a100-large"


# ---------------------------------------------------------------------------
# Tier gate decorator
# ---------------------------------------------------------------------------


class TestTierGateDecorator:
    def test_gate_blocks_insufficient_tier(self, monkeypatch):
        fake_settings_cls = type(
            "FakeSettings",
            (),
            {
                "load": staticmethod(lambda: type("S", (), {"tier": Tier.FREE})()),
            },
        )
        monkeypatch.setattr(settings_mod, "CARLSettings", fake_settings_cls)
        monkeypatch.setattr(tier_mod, "detect_effective_tier", lambda t: t)

        @tier_gate(Tier.PRO, feature="train")
        def my_func():
            return "ok"

        with pytest.raises(TierGateError):
            my_func()

    def test_gate_allows_sufficient_tier(self, monkeypatch):
        fake_settings_cls = type(
            "FakeSettings",
            (),
            {
                "load": staticmethod(lambda: type("S", (), {"tier": Tier.PRO})()),
            },
        )
        monkeypatch.setattr(settings_mod, "CARLSettings", fake_settings_cls)
        monkeypatch.setattr(tier_mod, "detect_effective_tier", lambda t: t)

        @tier_gate(Tier.PRO, feature="train")
        def my_func():
            return "ok"

        assert my_func() == "ok"

    def test_gate_preserves_metadata(self):
        @tier_gate(Tier.ENTERPRISE, feature="mcp.serve")
        def serve():
            """Docstring."""
            pass

        assert serve.__tier_required__ == Tier.ENTERPRISE
        assert serve.__tier_feature__ == "mcp.serve"
        assert serve.__doc__ == "Docstring."


# ---------------------------------------------------------------------------
# TierGateError
# ---------------------------------------------------------------------------


class TestTierGateError:
    def test_message_contains_tier_names(self):
        err = TierGateError("train", Tier.PRO, Tier.FREE)
        assert "Paid" in str(err)
        assert "Free" in str(err)
        assert "train" in str(err)

    def test_message_contains_upgrade_url(self):
        err = TierGateError("mcp", Tier.ENTERPRISE, Tier.PRO)
        assert "carl.camp/pricing" in str(err)


# ---------------------------------------------------------------------------
# Settings tier_allows method
# ---------------------------------------------------------------------------


class TestSettingsTierAllows:
    def test_free_allows_observe(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(tier_mod, "_detect_hf_token", lambda: None)
        s = CARLSettings(tier=Tier.FREE)
        assert s.tier_allows("observe") is True

    def test_free_allows_train(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(tier_mod, "_detect_hf_token", lambda: None)
        s = CARLSettings(tier=Tier.FREE)
        assert s.tier_allows("train") is True

    def test_pro_allows_train(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(tier_mod, "_detect_hf_token", lambda: None)
        s = CARLSettings(tier=Tier.PRO)
        assert s.tier_allows("train") is True
