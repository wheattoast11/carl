"""CARL Studio settings system.

Layered configuration with auto-detection:
  env vars -> ~/.carl/config.yaml -> carl.yaml (project-local) -> defaults

Uses Pydantic v2 BaseSettings with YAML source support.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, Self, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from carl_studio.tier import Tier
from carl_studio.types.config import ComputeTarget, normalize_compute_target


# ---------------------------------------------------------------------------
# Tier alias resolution — single source of truth for legacy tier names.
# ---------------------------------------------------------------------------

_TIER_ALIASES: dict[str, Tier] = {
    "free": Tier.FREE,
    "paid": Tier.PAID,
    "pro": Tier.PAID,
    "enterprise": Tier.PAID,
}


def _resolve_tier_alias(value: str) -> Tier:
    """Accept legacy tier names while storing the active FREE/PAID model."""
    normalized = value.strip().lower()
    if normalized not in _TIER_ALIASES:
        raise ValueError(f"Invalid tier '{value}'. Must be: free or paid")
    return _TIER_ALIASES[normalized]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ObserveDefaults(BaseModel):
    """Default flags for carl observe."""

    show_entropy: bool = Field(default=True, description="Display entropy metric")
    show_phi: bool = Field(default=True, description="Display coherence (Phi)")
    show_sparkline: bool = Field(default=True, description="Inline sparklines")
    show_discontinuity: bool = Field(default=True, description="Discontinuity alerts")
    default_poll_interval: float = Field(
        default=2.0, ge=0.5, le=60.0, description="Live poll interval (seconds)"
    )
    default_source: str = Field(default="file", description="Default data source: file or trackio")


class Preset(str, Enum):
    """Built-in configuration presets."""

    RESEARCH = "research"
    PRODUCTION = "production"
    QUICK = "quick"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict[str, Any]] = {
    "research": {
        "log_level": "debug",
        "default_compute": ComputeTarget.A100_LARGE,
        "observe_defaults": ObserveDefaults(
            show_entropy=True,
            show_phi=True,
            show_sparkline=True,
            show_discontinuity=True,
            default_poll_interval=1.0,
        ),
    },
    "production": {
        "log_level": "warning",
        "default_compute": ComputeTarget.A100_LARGE,
        "observe_defaults": ObserveDefaults(
            show_entropy=False,
            show_phi=True,
            show_sparkline=False,
            show_discontinuity=True,
            default_poll_interval=10.0,
        ),
    },
    "quick": {
        "log_level": "info",
        "default_compute": ComputeTarget.L4X1,
        "observe_defaults": ObserveDefaults(
            show_entropy=True,
            show_phi=True,
            show_sparkline=True,
            show_discontinuity=False,
            default_poll_interval=5.0,
        ),
    },
}


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

def carl_home() -> Path:
    """Return the canonical CARL home directory.

    Resolution order:
      1. ``CARL_HOME`` environment variable (expanded + resolved) if set.
      2. ``~/.carl`` otherwise.

    This is the single source of truth for the CARL home directory. Modules
    that previously hardcoded ``Path.home() / ".carl"`` should call this
    helper (or import the :data:`CARL_HOME` constant below) so the env var
    is honored uniformly across the codebase.
    """
    env = os.environ.get("CARL_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".carl").resolve()


#: Snapshot of :func:`carl_home` at import time. Kept for back-compat with
#: callers that imported the name directly; prefer :func:`carl_home` in new
#: code because it reflects the current env state on every call.
CARL_HOME = carl_home()
GLOBAL_CONFIG = CARL_HOME / "config.yaml"
LOCAL_CONFIG_NAME = "carl.yaml"


def _find_local_config() -> Path | None:
    """Walk up from cwd to find a carl.yaml project config."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        candidate = parent / LOCAL_CONFIG_NAME
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Main settings model
# ---------------------------------------------------------------------------


class CARLSettings(BaseSettings):
    """CARL Studio user settings.

    Resolution order (highest priority first):
      1. Environment variables (CARL_TIER, CARL_LOG_LEVEL, etc.)
      2. ~/.carl/config.yaml (global user settings)
      3. carl.yaml in project directory (project-local overrides)
      4. Defaults defined here
    """

    model_config = SettingsConfigDict(
        env_prefix="CARL_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Tier --
    tier: Tier = Field(default=Tier.FREE, description="Subscription tier")
    preset: Preset = Field(default=Preset.CUSTOM, description="Configuration preset")

    # -- Credentials (auto-detect where possible) --
    hf_token: str | None = Field(default=None, description="HuggingFace token (auto-detected)")
    anthropic_api_key: str | None = Field(
        default=None, description="Anthropic API key (for --diagnose)"
    )
    openrouter_api_key: str | None = Field(default=None, description="OpenRouter API key")
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    llm_model: str = Field(default="", description="Default LLM model for synthesis")
    llm_base_url: str = Field(default="", description="Custom OpenAI-compatible base URL")
    default_chat_model: str = Field(
        default="claude-sonnet-4-6",
        description="Default model for carl chat (Anthropic model ID)",
    )

    # Platform (carl.camp)
    supabase_url: str = Field(default="", description="Supabase project URL")
    supabase_anon_key: str = Field(default="", description="Supabase anon key (public)")

    # -- Defaults --
    default_compute: ComputeTarget = Field(
        default=ComputeTarget.L40SX1, description="Default compute target"
    )
    default_model: str = Field(default="", description="Default base model")
    hub_namespace: str = Field(
        default="", description="HF Hub namespace (auto-detected from whoami)"
    )
    trackio_url: str | None = Field(default=None, description="Trackio dashboard URL")
    naming_prefix: str = Field(default="", description="Naming convention prefix")
    log_level: str = Field(default="info", description="Logging level")
    show_per_turn_cost: bool = Field(
        default=True,
        description="Show per-turn cost delta and session running total in carl chat",
    )

    # -- Observe defaults --
    observe_defaults: ObserveDefaults = Field(default_factory=ObserveDefaults)

    @field_validator("tier", mode="before")
    @classmethod
    def normalize_tier_alias(cls, v: Any) -> Any:
        if isinstance(v, str):
            return _resolve_tier_alias(v)
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"debug", "info", "warning", "error", "critical"}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return v_lower

    @model_validator(mode="after")
    def apply_preset(self) -> Self:
        """Apply preset values as defaults (explicit values override)."""
        if self.preset != Preset.CUSTOM and self.preset.value in _PRESETS:
            preset_vals = _PRESETS[self.preset.value]
            # Only apply preset values if the field is still at its default
            for field_name, preset_val in preset_vals.items():
                model_field = type(self).model_fields.get(field_name)
                if model_field is not None:
                    current = getattr(self, field_name)
                    if current == model_field.default or (
                        callable(model_field.default_factory)
                        and current == model_field.default_factory()
                    ):
                        object.__setattr__(self, field_name, preset_val)
        return self

    # -- Auto-detection --

    def _auto_detect_credentials(self) -> None:
        """Fill in credentials from HF hub credentials, then env fallback."""
        if self.hf_token is None:
            try:
                from huggingface_hub import get_token

                self.hf_token = get_token()
            except Exception:
                pass
            if self.hf_token is None:
                self.hf_token = os.environ.get("HF_TOKEN")

        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not self.hub_namespace and self.hf_token:
            try:
                from huggingface_hub import HfApi

                info = HfApi(token=self.hf_token).whoami()
                self.hub_namespace = info.get("name", "")
            except Exception:
                pass

    # -- Persistence --

    @classmethod
    def load(cls) -> CARLSettings:
        """Load settings with full resolution chain.

        Priority: env vars -> ~/.carl/config.yaml -> carl.yaml -> defaults.

        The project-local ``carl.yaml`` is filtered against
        :data:`~pydantic.BaseModel.model_fields` so only keys declared on
        this model are pulled in — any unrelated project-config keys are
        ignored. New settings fields are picked up automatically without
        an allow-list edit.
        """
        merged: dict[str, Any] = {}

        # Layer 1: project-local carl.yaml (lowest file priority).
        # Filter by the declared model fields so the file can hold arbitrary
        # project keys without polluting settings construction.
        local_config = _find_local_config()
        if local_config is not None:
            local_data = _load_yaml(local_config)
            allowed = local_data.keys() & cls.model_fields.keys()
            for key in allowed:
                merged[key] = local_data[key]

        # Layer 2: global ~/.carl/config.yaml (overrides project)
        if GLOBAL_CONFIG.is_file():
            global_data = _load_yaml(GLOBAL_CONFIG)
            merged.update(global_data)

        # Layer 3: env vars are handled by BaseSettings automatically
        settings = cls(**merged)
        settings._auto_detect_credentials()
        return settings

    def save(self, path: Path | None = None) -> Path:
        """Save settings to YAML. Defaults to ~/.carl/config.yaml."""
        target = path or GLOBAL_CONFIG
        target.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(
            exclude={
                "hf_token",
                "anthropic_api_key",
                "openrouter_api_key",
                "openai_api_key",
            },  # Never persist secrets
            exclude_defaults=False,
            mode="json",
        )

        # Convert enums to their string values for YAML readability
        _serialize_enums(data)

        with open(target, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        return target

    @classmethod
    def from_yaml(cls, path: Path | str) -> CARLSettings:
        """Load settings from a specific YAML file."""
        data = _load_yaml(Path(path))
        settings = cls(**data)
        settings._auto_detect_credentials()
        return settings

    def tier_allows(self, feature: str) -> bool:
        """Check if the current effective tier allows a feature."""
        from carl_studio.tier import detect_effective_tier, tier_allows

        effective = detect_effective_tier(self.tier)
        return tier_allows(effective, feature)

    def get_effective_tier(self) -> Tier:
        """Get the effective tier after auto-elevation."""
        from carl_studio.tier import detect_effective_tier

        return detect_effective_tier(self.tier)

    def display_dict(self, mask_secrets: bool = True) -> dict[str, str]:
        """Get a display-friendly dict with optional secret masking."""
        effective = self.get_effective_tier()
        result: dict[str, str] = {
            "tier": self.tier.value,
            "effective_tier": effective.value,
            "preset": self.preset.value,
            "default_model": self.default_model,
            "default_compute": self.default_compute.value,
            "hub_namespace": self.hub_namespace or "(auto-detect)",
            "naming_prefix": self.naming_prefix,
            "log_level": self.log_level,
            "trackio_url": self.trackio_url or "(not set)",
        }

        if mask_secrets:
            result["hf_token"] = _mask(self.hf_token)
            result["anthropic_api_key"] = _mask(self.anthropic_api_key)
        else:
            result["hf_token"] = self.hf_token or "(not set)"
            result["anthropic_api_key"] = self.anthropic_api_key or "(not set)"

        # Observe defaults
        obs = self.observe_defaults
        result["observe.entropy"] = str(obs.show_entropy)
        result["observe.phi"] = str(obs.show_phi)
        result["observe.sparkline"] = str(obs.show_sparkline)
        result["observe.poll_interval"] = f"{obs.default_poll_interval}s"
        result["observe.source"] = obs.default_source

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning empty dict on any error."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _mask(value: str | None) -> str:
    """Mask a sensitive value for display."""
    if not value:
        return "(not set)"
    if len(value) <= 8:
        return "****"
    return value[:4] + "****" + value[-4:]


def _serialize_enums(data: dict[str, Any]) -> None:
    """Recursively convert enum values to strings for YAML serialization."""
    for key, val in data.items():
        if isinstance(val, Enum):
            data[key] = val.value
        elif isinstance(val, dict):
            _serialize_enums(val)


# ---------------------------------------------------------------------------
# Settable fields (for `carl config set`) — derived from model_fields
# ---------------------------------------------------------------------------

# Fields excluded from the CLI settable surface. Secrets are never exposed
# as settable; structured sub-models (``observe_defaults``) are expanded into
# their nested leaves automatically.
_SETTABLE_EXCLUDE: frozenset[str] = frozenset({
    "hf_token",
    "anthropic_api_key",
    "openrouter_api_key",
    "openai_api_key",
})


def _python_type_from_annotation(annotation: Any) -> type:
    """Collapse a field annotation to the coerced scalar type.

    ``bool`` / ``float`` / ``int`` are preserved verbatim; ``X | None`` and
    ``Optional[X]`` unwrap to the non-``None`` arm; enums and strings alike
    coerce through :class:`str` at the CLI boundary.
    """
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _python_type_from_annotation(args[0])
    if isinstance(annotation, type):
        if annotation is bool:
            return bool
        if annotation is float:
            return float
        if annotation is int:
            return int
    return str


def _build_settable_fields() -> dict[str, tuple[type, ...]]:
    """Build the settable-field registry from ``CARLSettings.model_fields``.

    Each top-level scalar field on the model contributes one entry. Nested
    Pydantic models (e.g. ``ObserveDefaults``) contribute one entry per leaf
    in the form ``"<parent>.<child>"``. New fields are picked up automatically
    the next time this module is imported.
    """
    registry: dict[str, tuple[type, ...]] = {}
    for name, field in CARLSettings.model_fields.items():
        if name in _SETTABLE_EXCLUDE:
            continue
        annotation = field.annotation
        origin_annotation = annotation
        # Unwrap Optional so isinstance check below reaches the real class.
        if get_origin(annotation) in (Union, UnionType):
            origin_annotation = next(
                (a for a in get_args(annotation) if a is not type(None)),
                annotation,
            )
        if isinstance(origin_annotation, type) and issubclass(
            origin_annotation, BaseModel
        ):
            for sub_name, sub_field in origin_annotation.model_fields.items():
                sub_type = _python_type_from_annotation(sub_field.annotation)
                registry[f"{name}.{sub_name}"] = (sub_type,)
            continue
        registry[name] = (_python_type_from_annotation(annotation),)
    return registry


SETTABLE_FIELDS: dict[str, tuple[type, ...]] = _build_settable_fields()


def set_field(settings: CARLSettings, key: str, value: str) -> CARLSettings:
    """Set a field on CARLSettings by dotted key path. Returns updated settings.

    Raises ValueError for invalid keys or values.
    """
    if key not in SETTABLE_FIELDS:
        valid = ", ".join(sorted(SETTABLE_FIELDS.keys()))
        raise ValueError(f"Unknown setting '{key}'. Valid keys: {valid}")

    # Parse the value to the right type
    parsed: Any
    if key == "tier":
        try:
            parsed = _resolve_tier_alias(value)
        except ValueError:
            raise ValueError(f"Invalid tier '{value}'. Must be: free or paid")
    elif key == "preset":
        try:
            parsed = Preset(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid preset '{value}'. Must be: research, production, quick, custom"
            )
    elif key == "default_compute":
        try:
            parsed = ComputeTarget(normalize_compute_target(value))
        except ValueError:
            valid = ", ".join(ct.value for ct in ComputeTarget)
            raise ValueError(f"Invalid compute target '{value}'. Must be one of: {valid}")
    elif bool in SETTABLE_FIELDS[key]:
        parsed = value.lower() in ("true", "1", "yes", "on")
    elif float in SETTABLE_FIELDS[key]:
        try:
            parsed = float(value)
        except ValueError:
            raise ValueError(f"'{key}' must be a number, got '{value}'")
    else:
        parsed = value

    # Apply to the settings object
    parts = key.split(".")
    if len(parts) == 1:
        setattr(settings, parts[0], parsed)
    elif len(parts) == 2:
        sub = getattr(settings, parts[0])
        setattr(sub, parts[1], parsed)
    else:
        raise ValueError(f"Unsupported key depth: {key}")

    return settings


def reset_settings() -> CARLSettings:
    """Reset to defaults (removes ~/.carl/config.yaml)."""
    if GLOBAL_CONFIG.is_file():
        GLOBAL_CONFIG.unlink()
    return CARLSettings()
