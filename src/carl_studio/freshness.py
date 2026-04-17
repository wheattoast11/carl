"""Knowledge and dependency freshness check.

Runs silently on startup. Reports only when action is needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

CARL_HOME = Path.home() / ".carl"
FRESHNESS_FILE = CARL_HOME / "freshness.yaml"
CHECK_INTERVAL_DAYS = 7


@dataclass
class FreshnessReport:
    """Results of a freshness check."""

    checked_at: str = ""
    stale_packages: list[str] = field(default_factory=list)
    config_warnings: list[str] = field(default_factory=list)
    credential_warnings: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(
            self.stale_packages or self.config_warnings or self.credential_warnings
        )

    @property
    def summary(self) -> str:
        parts: list[str] = []
        if self.stale_packages:
            parts.append(f"{len(self.stale_packages)} stale package(s)")
        if self.config_warnings:
            parts.append(f"{len(self.config_warnings)} config warning(s)")
        if self.credential_warnings:
            parts.append(f"{len(self.credential_warnings)} credential issue(s)")
        return ", ".join(parts) if parts else "all clear"


def needs_check() -> bool:
    """Return True if a freshness check is due (>7 days since last check)."""
    if not FRESHNESS_FILE.exists():
        return True
    try:
        data = yaml.safe_load(FRESHNESS_FILE.read_text()) or {}
        last_check = data.get("last_check", "")
        if not last_check:
            return True
        last_dt = datetime.fromisoformat(last_check)
        return datetime.now() - last_dt > timedelta(days=CHECK_INTERVAL_DAYS)
    except Exception:
        logger.debug("freshness: could not read %s, check needed", FRESHNESS_FILE)
        return True


def run_freshness_check(force: bool = False) -> FreshnessReport:
    """Run all freshness checks. Returns report.

    Checks:
    1. Installed package versions vs floor requirements
    2. carl.yaml config for deprecated options
    3. HuggingFace token validity and camp session expiry
    """
    report = FreshnessReport(checked_at=datetime.now().isoformat())

    _check_packages(report)
    _check_config(report)
    _check_credentials(report)

    _save_check_time(report.checked_at)

    return report


def _check_packages(report: FreshnessReport) -> None:
    """Check installed packages against minimum recommended versions."""
    import importlib.metadata as meta

    # Key packages with recommended minimums
    checks: dict[str, str] = {
        "trl": "1.0",
        "transformers": "5.0",
        "anthropic": "0.95",
        "huggingface-hub": "1.0",
        "peft": "0.15",
    }

    for pkg, min_ver in checks.items():
        try:
            installed = meta.version(pkg)
            if not installed:
                continue
            import re
            installed_parts = [
                int(m.group()) for x in installed.split(".")[:2]
                if (m := re.match(r"\d+", x))
            ]
            min_parts = [int(x) for x in min_ver.split(".")[:2]]
            if installed_parts < min_parts:
                report.stale_packages.append(
                    f"{pkg}: installed {installed}, recommended >={min_ver}"
                )
        except meta.PackageNotFoundError:
            pass  # Not installed -- not an issue
        except (ValueError, IndexError, AttributeError, TypeError):
            pass  # Unparseable version -- skip


def _check_config(report: FreshnessReport) -> None:
    """Check carl.yaml for deprecated or problematic options."""
    config_path = Path.cwd() / "carl.yaml"
    if not config_path.exists():
        return

    try:
        data = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        report.config_warnings.append("carl.yaml is not valid YAML")
        return

    if data.get("naming_prefix") == "il-terminals-carl":
        report.config_warnings.append(
            "naming_prefix is set to 'il-terminals-carl' (internal default) -- "
            "consider setting your own prefix"
        )
    if data.get("hub_namespace") == "wheattoast11":
        report.config_warnings.append(
            "hub_namespace is set to 'wheattoast11' (internal default) -- "
            "set your own namespace with: carl config set hub_namespace YOUR_NAME"
        )


def _check_credentials(report: FreshnessReport) -> None:
    """Check credential freshness using local data only."""
    # HuggingFace token (local cached check, no network)
    try:
        from huggingface_hub import get_token

        token = get_token()
        if not token:
            report.credential_warnings.append(
                "No HuggingFace token found -- run 'hf auth login' for Hub access"
            )
    except ImportError:
        pass  # huggingface_hub not installed

    # Camp session (get_auth returns None when expired)
    try:
        from carl_studio.db import LocalDB

        db = LocalDB()
        jwt = db.get_auth("jwt")
        if not jwt:
            # Only warn if there was ever a session (tier or profile cached)
            had_session = bool(
                db.get_auth("tier") or db.get_config("camp_profile")
            )
            if had_session:
                report.credential_warnings.append(
                    "carl.camp session expired -- run 'carl camp login' to refresh"
                )
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
