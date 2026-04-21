"""NL-interpretable config requirements primitive.

Adapters, compute backends, and any component with external configuration
surface emit ``list[ConfigRequirement]`` via a ``config_requirements()``
method. Carl's chat agent reads these to answer natural-language setup
questions ("what do I need to run slime?") **without ever seeing the
values**.

Design constraints
------------------

1. ``status`` is boolean-valued (``set | unset | invalid``) — never carries
   the actual configured value. A token might be set to "hf_abcdef…" but
   the requirement only records ``status="set"`` and optionally a 12-hex
   fingerprint of the value for dedup / change detection.

2. ``location`` tells the user **where** to set the value (env var name,
   config file path, CLI flag) — not how it flows through the system.

3. ``description`` is one-to-three sentences of natural language explaining
   what this requirement is *for*, so Carl can paraphrase when asked.

4. Error codes are stable under ``carl.config.*``.

This primitive lives in carl-core so downstream packages (carl-studio,
carl-agent, carl-training, carl-marketplace) can produce and consume
config reports without circular dependencies.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field


__all__ = [
    "ConfigLocation",
    "ConfigRequirement",
    "ConfigReport",
    "ConfigStatus",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConfigStatus(str, Enum):
    """Boolean-valued state of a single requirement.

    Never carries the actual value. ``invalid`` indicates the value was
    observed to be present but failed validation (e.g., a wrong-format
    URL, a zero-length token).
    """

    SET = "set"
    UNSET = "unset"
    INVALID = "invalid"


class ConfigLocation(str, Enum):
    """Where the user sets this configuration value.

    CARL's NL surface maps these to human phrasings:
      * env → "Export in your shell: ``export NAME=...``"
      * yaml → "Set the ``<name>`` field in ``carl.yaml``."
      * user_config → "Add to ``~/.carl/config.yaml``."
      * credentials_file → "Create ``~/.carl/credentials/<name>``."
      * cli_flag → "Pass ``--<name>`` on the command line."
      * external → "Configured outside CARL (e.g., HF Jobs web UI)."
    """

    ENV = "env"
    YAML = "yaml"
    USER_CONFIG = "user_config"
    CREDENTIALS_FILE = "credentials_file"
    CLI_FLAG = "cli_flag"
    EXTERNAL = "external"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ConfigRequirement(BaseModel):
    """A single configuration surface item.

    Producers emit these without ever reading the actual value. ``fingerprint``
    is optional: when set, it's the 12-hex sha256 prefix of the value (safe
    to log) used for change detection across runs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        ...,
        description="The identifier the user recognizes (env var name, "
        "YAML key path, CLI flag). Example: CARL_CAMP_HF_TOKEN, base_model, "
        "slime.tensor_parallel.",
    )
    location: ConfigLocation = Field(
        ...,
        description="Where the value gets set. Drives the NL phrasing Carl uses.",
    )
    required: bool = Field(
        ...,
        description="True if the component refuses to run without it; False "
        "when there's a reasonable default or it's purely ergonomic.",
    )
    status: ConfigStatus = Field(
        ...,
        description="Current state. Never carries the actual value.",
    )
    description: str = Field(
        ...,
        description="One-to-three-sentence NL explanation of what this "
        "requirement is for. Carl paraphrases this when asked.",
    )
    fingerprint: str | None = Field(
        default=None,
        description="Optional 12-hex sha256 prefix of the value for change "
        "detection. Safe to log. ``None`` when the value is unset or the "
        "producer chose not to fingerprint.",
    )
    default_value: str | None = Field(
        default=None,
        description="Optional human-facing hint about the default CARL uses "
        "when unset. Never a secret-shaped value.",
    )
    category: str = Field(
        default="general",
        description="Free-form grouping (e.g., 'auth', 'dataset', 'compute', "
        "'parallelism'). Used by Carl to batch related prompts together.",
    )


class ConfigReport(BaseModel):
    """A component's full configuration surface as a typed bundle.

    Every adapter / compute backend / CLI entry that wants to participate
    in Carl's NL interpretability story returns a :class:`ConfigReport`
    from its ``config_requirements()`` method.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component: str = Field(
        ...,
        description="Canonical component name (e.g., 'slime', 'trl', "
        "'x402.payment_rail'). Used to group reports in Carl's UI.",
    )
    requirements: list[ConfigRequirement] = Field(
        default_factory=lambda: cast(list[ConfigRequirement], [])
    )

    def missing(self) -> list[ConfigRequirement]:
        """Return the subset of requirements that are required AND unset/invalid."""
        return [
            r
            for r in self.requirements
            if r.required and r.status != ConfigStatus.SET
        ]

    def summary(self) -> dict[str, Any]:
        """Compact status snapshot: {component, ok, missing_count, total}."""
        missing = self.missing()
        total = len(self.requirements)
        return {
            "component": self.component,
            "ok": not missing,
            "missing_count": len(missing),
            "total": total,
        }

    def describe_nl(self) -> str:
        """One-paragraph NL summary suitable for Carl to read aloud.

        Pure data → string mapping; no LLM involvement. The chat agent
        can use this directly OR paraphrase further with model help.
        """
        missing = self.missing()
        if not missing:
            return (
                f"{self.component} is fully configured. "
                f"{len(self.requirements)} requirements, all set."
            )
        lines = [
            f"{self.component} needs attention. "
            f"{len(missing)} of {len(self.requirements)} required items are missing:",
        ]
        for req in missing:
            verb = _LOCATION_VERB[req.location]
            lines.append(f"  - {req.name} ({verb}): {req.description}")
        return "\n".join(lines)


# Status-level mapping of ConfigLocation → short NL verb phrase used in
# describe_nl(). Kept dict-simple so Carl's chat layer can override per-
# locale if we ever go i18n.
_LOCATION_VERB: dict[ConfigLocation, str] = {
    ConfigLocation.ENV: "export in your shell",
    ConfigLocation.YAML: "set in carl.yaml",
    ConfigLocation.USER_CONFIG: "add to ~/.carl/config.yaml",
    ConfigLocation.CREDENTIALS_FILE: "create a credentials file",
    ConfigLocation.CLI_FLAG: "pass on the CLI",
    ConfigLocation.EXTERNAL: "configure outside CARL",
}


# ---------------------------------------------------------------------------
# Helpers for producers
# ---------------------------------------------------------------------------


def fingerprint_value(value: str | bytes) -> str:
    """Return a 12-hex sha256 prefix of ``value`` — safe to log.

    Use at the producer boundary to record ``ConfigRequirement.fingerprint``
    without persisting the raw value. Matches the 12-hex convention used
    throughout ``carl_core.interaction.Step.probe_call``.
    """
    import hashlib

    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()[:12]


def require_env(
    name: str,
    *,
    description: str,
    category: str = "general",
    default_value: str | None = None,
    value_accessor: Any = None,
) -> ConfigRequirement:
    """Build a :class:`ConfigRequirement` for an env var.

    ``value_accessor`` (typically ``os.environ.get``) is invoked to probe
    presence WITHOUT leaking the value into the producer's context; only
    ``bool(value)`` and, when present, ``fingerprint_value(value)`` are
    retained.
    """
    import os

    accessor: Any = value_accessor if value_accessor is not None else os.environ.get
    raw = accessor(name)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        status = ConfigStatus.UNSET
        fp = None
    else:
        status = ConfigStatus.SET
        try:
            fp = fingerprint_value(raw if isinstance(raw, str) else str(raw))
        except Exception:
            fp = None
    return ConfigRequirement(
        name=name,
        location=ConfigLocation.ENV,
        required=True,
        status=status,
        description=description,
        fingerprint=fp,
        default_value=default_value,
        category=category,
    )


def require_yaml_key(
    name: str,
    carl_config: dict[str, Any],
    *,
    description: str,
    required: bool = True,
    category: str = "general",
    default_value: str | None = None,
) -> ConfigRequirement:
    """Build a :class:`ConfigRequirement` for a ``carl.yaml`` key path.

    ``name`` supports dotted paths: ``"base_model"`` or ``"slime.tensor_parallel"``.
    Presence (bool) + fingerprint are recorded; value never returned.
    """
    parts = name.split(".")
    cursor: Any = carl_config
    present = True
    raw_final: Any = None
    for part in parts:
        if isinstance(cursor, dict):
            cursor_dict = cast(dict[str, Any], cursor)
            if part in cursor_dict:
                cursor = cursor_dict[part]
                raw_final = cursor
            else:
                present = False
                break
        else:
            present = False
            break

    if not present or raw_final in (None, "", [], {}):
        status = ConfigStatus.UNSET
        fp = None
    else:
        status = ConfigStatus.SET
        try:
            fp = fingerprint_value(str(raw_final))
        except Exception:
            fp = None

    return ConfigRequirement(
        name=name,
        location=ConfigLocation.YAML,
        required=required,
        status=status,
        description=description,
        fingerprint=fp,
        default_value=default_value,
        category=category,
    )


# Re-export helpers so downstream producers can do a single-line import.
ConfigRequirement.model_rebuild()  # finalize forward-ref resolution


# Extended __all__ with helpers.
__all__ += ["fingerprint_value", "require_env", "require_yaml_key"]
