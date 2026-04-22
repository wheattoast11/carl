"""Knowledge and dependency freshness check.

Runs silently on startup. Reports only when action is needed.

Surface:

    report = run_freshness_check()              # may be TTL-short-circuited elsewhere
    report = run_freshness_check(force=True)    # always runs
    if report.has_errors: ...
    for issue in report.issues: ...

Each ``FreshnessIssue`` is a structured, stable record keyed by ``code`` so
callers can react programmatically (e.g. auto-remediate a credential gap,
surface a tier-appropriate nudge, or ignore informational notes). All issue
codes live under the ``carl.freshness.*`` namespace; this matches the
``carl_core.errors`` convention and keeps the telemetry surface predictable.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from carl_core.errors import ValidationError

logger = logging.getLogger(__name__)

CARL_HOME = Path.home() / ".carl"
FRESHNESS_FILE = CARL_HOME / "freshness.yaml"
CHECK_INTERVAL_DAYS = 1  # TTL default: 24h (per spec).

# ---------------------------------------------------------------------------
# Issue codes
# ---------------------------------------------------------------------------

#: Installed package is older than the recommended floor.
CODE_STALE_PKG = "carl.freshness.stale_pkg"
#: Installed package's metadata is corrupt / unreadable — typically a stale
#: ``*.dist-info`` left by an interrupted pip upgrade. The wizard can auto-
#: heal via ``pip install --force-reinstall --no-deps <pkg>``.
CODE_DEP_CORRUPT = "carl.freshness.dep_corrupt"
#: ``carl.yaml`` references an internal-default value that should be overridden.
CODE_CONFIG_INTERNAL_DEFAULT = "carl.freshness.config_internal_default"
#: ``carl.yaml`` on disk failed to parse as YAML.
CODE_CONFIG_INVALID = "carl.freshness.config_invalid"
#: No HuggingFace token available for Hub-gated workflows.
CODE_MISSING_HF_TOKEN = "carl.freshness.missing_hf_token"
#: Camp session JWT expired — login is required to refresh.
CODE_CAMP_SESSION_EXPIRED = "carl.freshness.camp_session_expired"
#: No Anthropic / OpenAI / OpenRouter provider key found for agent flows.
CODE_MISSING_LLM_KEY = "carl.freshness.missing_llm_key"

# ---------------------------------------------------------------------------
# Severity / category string constants (kept explicit to avoid enum churn in
# JSON payloads and telemetry — matching ``carl_core.errors`` conventions).
# ---------------------------------------------------------------------------

SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_ERROR = "error"

CATEGORY_PACKAGE = "package"
CATEGORY_CONFIG = "config"
CATEGORY_CREDENTIAL = "credential"

_VALID_SEVERITIES = {SEVERITY_INFO, SEVERITY_WARN, SEVERITY_ERROR}
_VALID_CATEGORIES = {CATEGORY_PACKAGE, CATEGORY_CONFIG, CATEGORY_CREDENTIAL}


class FreshnessIssue(BaseModel):
    """A single finding produced by a freshness check.

    Stable fields, keyed by ``code`` under the ``carl.freshness.*`` namespace.
    ``severity`` and ``category`` are string constants rather than enums so
    JSON payloads stay portable across SDK versions.
    """

    code: str
    severity: str
    category: str
    subject: str
    detail: str
    remediation: str

    def __init__(self, **data: Any) -> None:
        severity = data.get("severity")
        if severity not in _VALID_SEVERITIES:
            raise ValidationError(
                f"invalid severity {severity!r}",
                code="carl.freshness.invalid_severity",
                context={"severity": severity, "valid": sorted(_VALID_SEVERITIES)},
            )
        category = data.get("category")
        if category not in _VALID_CATEGORIES:
            raise ValidationError(
                f"invalid category {category!r}",
                code="carl.freshness.invalid_category",
                context={"category": category, "valid": sorted(_VALID_CATEGORIES)},
            )
        super().__init__(**data)


class FreshnessReport(BaseModel):
    """Results of a freshness check.

    Issues are first-class (see :class:`FreshnessIssue`). Legacy string-list
    accessors are preserved for backward-compatibility with older callers and
    tests; new code should iterate ``issues`` directly.
    """

    issues: list[FreshnessIssue] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=datetime.now)

    # Back-compat shims: older tests and call sites instantiated the report
    # with ``stale_packages=[...]`` / ``config_warnings=[...]`` / etc. We honor
    # those keyword arguments by translating them into structured issues at
    # construction time.
    def __init__(self, **data: Any) -> None:
        stale_packages = data.pop("stale_packages", None)
        config_warnings = data.pop("config_warnings", None)
        credential_warnings = data.pop("credential_warnings", None)

        if "checked_at" in data and isinstance(data["checked_at"], str):
            raw = data["checked_at"]
            if raw:
                try:
                    data["checked_at"] = datetime.fromisoformat(raw)
                except ValueError:
                    data["checked_at"] = datetime.now()
            else:
                data["checked_at"] = datetime.now()

        issues: list[FreshnessIssue] = list(data.pop("issues", []) or [])
        if stale_packages:
            for entry in stale_packages:
                issues.append(
                    FreshnessIssue(
                        code=CODE_STALE_PKG,
                        severity=SEVERITY_WARN,
                        category=CATEGORY_PACKAGE,
                        subject=str(entry).split(":", 1)[0].strip() or "package",
                        detail=str(entry),
                        remediation="pip install --upgrade <package>",
                    )
                )
        if config_warnings:
            for entry in config_warnings:
                issues.append(
                    FreshnessIssue(
                        code=CODE_CONFIG_INTERNAL_DEFAULT,
                        severity=SEVERITY_WARN,
                        category=CATEGORY_CONFIG,
                        subject="carl.yaml",
                        detail=str(entry),
                        remediation="edit carl.yaml or run 'carl config set'",
                    )
                )
        if credential_warnings:
            for entry in credential_warnings:
                issues.append(
                    FreshnessIssue(
                        code=CODE_CAMP_SESSION_EXPIRED,
                        severity=SEVERITY_WARN,
                        category=CATEGORY_CREDENTIAL,
                        subject="credentials",
                        detail=str(entry),
                        remediation="carl camp login",
                    )
                )
        data["issues"] = issues
        super().__init__(**data)

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------
    @property
    def has_errors(self) -> bool:
        return any(i.severity == SEVERITY_ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == SEVERITY_WARN for i in self.issues)

    @property
    def has_issues(self) -> bool:
        """Any issue at warn severity or above.

        Preserved for existing callers and tests.
        """
        return self.has_errors or self.has_warnings

    @property
    def summary(self) -> str:
        if not self.issues:
            return "all clear"
        pkg = sum(1 for i in self.issues if i.category == CATEGORY_PACKAGE)
        cfg = sum(1 for i in self.issues if i.category == CATEGORY_CONFIG)
        cred = sum(1 for i in self.issues if i.category == CATEGORY_CREDENTIAL)
        parts: list[str] = []
        if pkg:
            parts.append(f"{pkg} stale package(s)")
        if cfg:
            parts.append(f"{cfg} config warning(s)")
        if cred:
            parts.append(f"{cred} credential issue(s)")
        return ", ".join(parts) if parts else "all clear"

    # ------------------------------------------------------------------
    # Back-compat string-list views over ``issues``
    # ------------------------------------------------------------------
    @property
    def stale_packages(self) -> list[str]:
        return [i.detail for i in self.issues if i.category == CATEGORY_PACKAGE]

    @property
    def config_warnings(self) -> list[str]:
        return [i.detail for i in self.issues if i.category == CATEGORY_CONFIG]

    @property
    def credential_warnings(self) -> list[str]:
        return [i.detail for i in self.issues if i.category == CATEGORY_CREDENTIAL]

    # ------------------------------------------------------------------
    # Mutation helpers (used by the private ``_check_*`` functions)
    # ------------------------------------------------------------------
    def add(self, issue: FreshnessIssue) -> None:
        self.issues.append(issue)


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------


def needs_check() -> bool:
    """Return True if a freshness check is due (TTL exceeded).

    TTL is ``CHECK_INTERVAL_DAYS`` (default 24h). Missing, empty, or corrupt
    state files all force a check.
    """
    if not FRESHNESS_FILE.exists():
        return True
    try:
        data = yaml.safe_load(FRESHNESS_FILE.read_text()) or {}
        if not isinstance(data, dict):
            return True
        last_check = data.get("last_check", "")
        if not last_check:
            return True
        last_dt = datetime.fromisoformat(str(last_check))
        return datetime.now() - last_dt > timedelta(days=CHECK_INTERVAL_DAYS)
    except Exception:
        logger.debug("freshness: could not read %s, check needed", FRESHNESS_FILE)
        return True


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_freshness_check(*, force: bool = False) -> FreshnessReport:
    """Run all freshness checks. Returns a structured report.

    - ``force=False`` (default): runs unconditionally today (no TTL
      short-circuit here — callers gate on :func:`needs_check` when they want
      the TTL behavior, e.g. ``carl start``).
    - ``force=True``: identical semantics; accepted for symmetry with
      ``carl doctor --check-freshness`` and ``carl flow /freshness``, which
      want to be explicit about bypassing any caller-side TTL.

    Checks:
      1. Installed package versions vs floor requirements.
      2. ``carl.yaml`` for internal-default / deprecated options.
      3. HuggingFace token and camp session freshness (local data only).
    """
    del force  # accepted for API symmetry; see docstring.
    report = FreshnessReport(checked_at=datetime.now())

    _check_packages(report)
    _check_config(report)
    _check_credentials(report)

    _save_check_time(report.checked_at.isoformat())
    return report


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_packages(report: FreshnessReport) -> None:
    """Check installed packages: stale versions AND corrupt metadata.

    Uses :func:`carl_core.dependency_probe.probe` so we catch the
    import-ok/metadata-corrupt class of failure that previously escaped
    silent-swallow ``except (ValueError, AttributeError, TypeError): continue``.
    The HF huggingface-hub stale-dist-info scenario surfaces here as a
    ``dep_corrupt`` error with a concrete repair command; the wizard's
    ``_offer_extras`` auto-heal path picks it up from there.
    """
    import re

    from carl_core.dependency_probe import probe

    # Key packages with recommended minimums (kept aligned with pyproject floors).
    checks: dict[str, str] = {
        "trl": "1.0",
        "transformers": "5.0",
        "anthropic": "0.95",
        "huggingface-hub": "1.0",
        "peft": "0.15",
    }

    for pkg, min_ver in checks.items():
        result = probe(pkg)

        # Not installed / not registered: silently skip. Install-missing is
        # a separate surface handled by `_offer_extras` in `carl init`;
        # metadata_missing often means intentional manual install.
        if result.is_missing or result.status == "metadata_missing":
            continue

        # Corruption we cannot recover a version from: emit dep_corrupt.
        # ``version_mismatch`` falls through to the floor check below —
        # it still has a readable metadata_version to compare.
        if result.status in ("metadata_corrupt", "import_error", "import_value_error"):
            detail_source = result.import_error or result.metadata_error
            detail = (
                detail_source.splitlines()[0][:160]
                if detail_source
                else "metadata unreadable"
            )
            report.add(
                FreshnessIssue(
                    code=CODE_DEP_CORRUPT,
                    severity=SEVERITY_ERROR,
                    category=CATEGORY_PACKAGE,
                    subject=pkg,
                    detail=f"{pkg}: {result.status} — {detail}",
                    remediation=result.repair_command,
                )
            )
            continue

        # Healthy / version_mismatch: compare against recommended floor.
        installed = result.metadata_version
        if not installed:
            continue
        try:
            installed_parts = [
                int(m.group())
                for x in installed.split(".")[:2]
                if (m := re.match(r"\d+", x))
            ]
            min_parts = [int(x) for x in min_ver.split(".")[:2]]
        except (ValueError, IndexError):
            continue
        if installed_parts and installed_parts < min_parts:
            report.add(
                FreshnessIssue(
                    code=CODE_STALE_PKG,
                    severity=SEVERITY_WARN,
                    category=CATEGORY_PACKAGE,
                    subject=pkg,
                    detail=f"{pkg}: installed {installed}, recommended >={min_ver}",
                    remediation=f"pip install --upgrade '{pkg}>={min_ver}'",
                )
            )


def _check_config(report: FreshnessReport) -> None:
    """Check carl.yaml for deprecated or problematic options."""
    config_path = Path.cwd() / "carl.yaml"
    if not config_path.exists():
        return

    try:
        data = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        report.add(
            FreshnessIssue(
                code=CODE_CONFIG_INVALID,
                severity=SEVERITY_ERROR,
                category=CATEGORY_CONFIG,
                subject="carl.yaml",
                detail="carl.yaml is not valid YAML",
                remediation="fix YAML syntax in carl.yaml",
            )
        )
        return

    if not isinstance(data, dict):
        return

    if data.get("naming_prefix") == "il-terminals-carl":
        report.add(
            FreshnessIssue(
                code=CODE_CONFIG_INTERNAL_DEFAULT,
                severity=SEVERITY_WARN,
                category=CATEGORY_CONFIG,
                subject="naming_prefix",
                detail=(
                    "naming_prefix is set to 'il-terminals-carl' (internal default) -- "
                    "consider setting your own prefix"
                ),
                remediation="carl config set naming_prefix YOUR_PREFIX",
            )
        )
    if data.get("hub_namespace") == "wheattoast11":
        report.add(
            FreshnessIssue(
                code=CODE_CONFIG_INTERNAL_DEFAULT,
                severity=SEVERITY_WARN,
                category=CATEGORY_CONFIG,
                subject="hub_namespace",
                detail=(
                    "hub_namespace is set to 'wheattoast11' (internal default) -- "
                    "set your own namespace with: carl config set hub_namespace YOUR_NAME"
                ),
                remediation="carl config set hub_namespace YOUR_NAME",
            )
        )


def _check_credentials(report: FreshnessReport) -> None:
    """Check credential freshness using local data only."""
    # HuggingFace token (local cached check, no network).
    try:
        from huggingface_hub import get_token

        token = get_token()
        if not token:
            report.add(
                FreshnessIssue(
                    code=CODE_MISSING_HF_TOKEN,
                    severity=SEVERITY_WARN,
                    category=CATEGORY_CREDENTIAL,
                    subject="HuggingFace token",
                    detail=(
                        "No HuggingFace token found -- run 'hf auth login' for Hub access"
                    ),
                    remediation="hf auth login",
                )
            )
    except ImportError:
        pass  # huggingface_hub not installed

    # Camp session (``get_auth`` returns None when expired).
    try:
        from carl_studio.db import LocalDB

        db = LocalDB()
        try:
            jwt = db.get_auth("jwt")
            if not jwt:
                had_session = bool(
                    db.get_auth("tier") or db.get_config("camp_profile")
                )
                if had_session:
                    report.add(
                        FreshnessIssue(
                            code=CODE_CAMP_SESSION_EXPIRED,
                            severity=SEVERITY_WARN,
                            category=CATEGORY_CREDENTIAL,
                            subject="carl.camp session",
                            detail=(
                                "carl.camp session expired -- run "
                                "'carl camp login' to refresh"
                            ),
                            remediation="carl camp login",
                        )
                    )
        finally:
            db.close()
    except Exception:
        pass  # DB not available


def _save_check_time(timestamp: str) -> None:
    """Persist the last check timestamp."""
    try:
        CARL_HOME.mkdir(parents=True, exist_ok=True)
        FRESHNESS_FILE.write_text(yaml.dump({"last_check": timestamp}))
    except OSError:
        logger.debug("freshness: could not write %s", FRESHNESS_FILE)


__all__ = [
    "CARL_HOME",
    "FRESHNESS_FILE",
    "CHECK_INTERVAL_DAYS",
    "CODE_STALE_PKG",
    "CODE_DEP_CORRUPT",
    "CODE_CONFIG_INTERNAL_DEFAULT",
    "CODE_CONFIG_INVALID",
    "CODE_MISSING_HF_TOKEN",
    "CODE_MISSING_LLM_KEY",
    "CODE_CAMP_SESSION_EXPIRED",
    "SEVERITY_INFO",
    "SEVERITY_WARN",
    "SEVERITY_ERROR",
    "CATEGORY_PACKAGE",
    "CATEGORY_CONFIG",
    "CATEGORY_CREDENTIAL",
    "FreshnessIssue",
    "FreshnessReport",
    "needs_check",
    "run_freshness_check",
]
