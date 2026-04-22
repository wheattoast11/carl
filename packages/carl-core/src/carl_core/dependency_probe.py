"""Dependency health probe — classify optional-dep state for UX.

The CARL CLI probes optional dependencies at multiple surfaces:
``carl init`` offers to install missing ones, ``carl doctor`` surfaces
broken ones, ``carl train`` gates on them. All three want the same
classification: is this dep present, importable, and registered with
pip's metadata system?

Naive probes do ``try: import X; return True; except ImportError: return False``.
That misses the class of bugs where ``import X`` succeeds but sibling-dep
metadata is corrupt (stale ``*.dist-info`` dirs, half-finished pip upgrades,
conda+pip mixing). The downstream package then raises ``ValueError`` at
import time from its own ``dependency_versions_check``, which escapes the
naive probe and crashes the wizard.

This module classifies each probe into one of seven states and emits a
suggested repair command. Callers compose multiple probes and present
auto-heal UX when the state warrants it.

All logic lives in carl-core because the shape is generic — no knowledge
of carl-studio's extras, no pip execution, no network.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from dataclasses import dataclass
from typing import Literal


ProbeStatus = Literal[
    "ok",                 # import + metadata both clean
    "missing",            # not installed at all
    "import_error",       # `import X` raised ImportError (broken files, missing C ext)
    "import_value_error", # `import X` raised ValueError (sibling-dep corruption)
    "metadata_missing",   # import works; metadata.version raises PackageNotFoundError
    "metadata_corrupt",   # import works; metadata.version returns None or raises
    "version_mismatch",   # import works; __version__ differs from metadata version
]


@dataclass(frozen=True)
class DepProbeResult:
    """Classification of a single dependency's health."""

    name: str
    normalized_name: str
    status: ProbeStatus
    import_ok: bool
    import_version: str | None
    metadata_version: str | None
    import_error: str | None
    metadata_error: str | None
    repair_command: str

    @property
    def healthy(self) -> bool:
        return self.status == "ok"

    @property
    def needs_repair(self) -> bool:
        """True when the package state is actively broken.

        ``metadata_missing`` is deliberately excluded — that status means
        the package was copied in manually or installed via ``--target``,
        which is the user's intent. Repair would override their choice.
        Callers that want to offer force-reinstall anyway can check
        ``status == "metadata_missing"`` explicitly.
        """
        return self.status in (
            "import_error",
            "import_value_error",
            "metadata_corrupt",
            "version_mismatch",
        )

    @property
    def is_missing(self) -> bool:
        return self.status == "missing"


def probe(name: str, *, import_name: str | None = None) -> DepProbeResult:
    """Probe a dependency and return its health classification.

    Args:
        name: Distribution name, e.g. ``"huggingface_hub"`` or
            ``"huggingface-hub"``. PEP-503-normalized internally so either
            form yields the same result.
        import_name: Override when the import name differs from the
            distribution name (e.g. ``"PIL"`` for ``"Pillow"``). Defaults
            to the normalized name with hyphens swapped for underscores.

    Returns:
        A :class:`DepProbeResult` describing the state. The call never
        raises — exotic import errors are captured into the result.
    """
    normalized = _normalize_pep503(name)
    effective_import_name = import_name or normalized.replace("-", "_")

    import_ok = False
    import_version: str | None = None
    import_error: str | None = None

    try:
        mod = importlib.import_module(effective_import_name)
        import_ok = True
        raw_version = getattr(mod, "__version__", None)
        if isinstance(raw_version, str):
            import_version = raw_version
    except ImportError as exc:
        import_error = f"{type(exc).__name__}: {exc}"
    except ValueError as exc:
        # Typically a sibling-dep metadata issue — another package's
        # dependency_versions_check raised because its dist-info is corrupt.
        # The caller can pattern-match on import_error via
        # extract_corrupt_sibling() to identify the corrupt sibling.
        import_error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        import_error = f"{type(exc).__name__}: {exc}"

    metadata_version, metadata_error, metadata_not_found = _lookup_metadata(normalized)

    status = _classify(
        import_ok=import_ok,
        import_version=import_version,
        import_error=import_error,
        metadata_version=metadata_version,
        metadata_error=metadata_error,
        metadata_not_found=metadata_not_found,
    )

    return DepProbeResult(
        name=name,
        normalized_name=normalized,
        status=status,
        import_ok=import_ok,
        import_version=import_version,
        metadata_version=metadata_version,
        import_error=import_error,
        metadata_error=metadata_error,
        repair_command=_repair_command(status, normalized),
    )


def extract_corrupt_sibling(error_text: str) -> str | None:
    """Parse a ``dependency_versions_check``-style error to find a corrupt dep.

    Used when ``import transformers`` raises
    ``ValueError: Unable to compare versions for huggingface-hub>=1.3.0,<2.0``
    — the broken sibling is ``huggingface-hub``, not ``transformers``.

    Returns the package name if the pattern matches; ``None`` otherwise.
    Callers should re-probe the returned name to confirm corruption before
    running a repair.
    """
    m = re.search(
        r"[Uu]nable to compare versions for\s+([A-Za-z0-9._-]+)",
        error_text,
    )
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _normalize_pep503(name: str) -> str:
    """PEP-503 normalize: lowercase + runs of [-_.] to single -."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _lookup_metadata(normalized_name: str) -> tuple[str | None, str | None, bool]:
    """Look up metadata version.

    Returns ``(version, error_message, was_not_found)``.

    - ``(version, None, False)`` — healthy metadata.
    - ``(None, None, True)`` — ``PackageNotFoundError`` (not in pip registry).
    - ``(None, "empty_version_field", False)`` — ``importlib.metadata`` returned
      empty/None (dist-info exists but Metadata-Version is empty).
    - ``(None, str, False)`` — any other exception during lookup.

    We distinguish "not found" from "empty/null version" because they call for
    different UX: the first is often a manual-install quirk (skip silently),
    the second is real corruption (surface + auto-heal).
    """
    try:
        v = importlib.metadata.version(normalized_name)
    except importlib.metadata.PackageNotFoundError:
        return None, None, True
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", False

    if not v:
        # Dist-info directory was found but the Version field was empty/None.
        # This is the exact HF corruption fingerprint.
        return None, "empty_version_field", False
    return v, None, False


def _classify(
    *,
    import_ok: bool,
    import_version: str | None,
    import_error: str | None,
    metadata_version: str | None,
    metadata_error: str | None,
    metadata_not_found: bool,
) -> ProbeStatus:
    if not import_ok:
        if import_error and import_error.startswith("ValueError"):
            return "import_value_error"
        if metadata_not_found:
            return "missing"
        return "import_error"

    # Import succeeded.
    if metadata_error is not None:
        return "metadata_corrupt"
    if metadata_not_found:
        # Installed but not in pip's registry — manually copied, or pip
        # registry broken. Skip silently in most UX; auto-heal can still
        # offer force-reinstall if requested.
        return "metadata_missing"
    if metadata_version is None:
        return "metadata_corrupt"
    if import_version and import_version != metadata_version:
        return "version_mismatch"
    return "ok"


def _repair_command(status: ProbeStatus, normalized_name: str) -> str:
    if status == "ok":
        return ""
    if status == "missing":
        return f"pip install {normalized_name}"
    if status == "import_error":
        return f"pip install --force-reinstall {normalized_name}"
    # metadata_corrupt / metadata_missing / version_mismatch / import_value_error
    # — force-reinstall + --no-deps cleans stale dist-info without disturbing
    # the rest of the dependency tree.
    return f"pip install --force-reinstall --no-deps {normalized_name}"


__all__ = [
    "DepProbeResult",
    "ProbeStatus",
    "probe",
    "extract_corrupt_sibling",
]
