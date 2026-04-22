"""Tests for carl_core.dependency_probe.

Covers the 7 ProbeStatus values plus PEP-503 normalization, import-name
override, and the ``extract_corrupt_sibling`` parser. The status matrix
mirrors the real failure modes observed in the HF/transformers scenario
that motivated this module.

Patching note: we use ``patch.object`` against the already-imported
modules rather than string paths. ``patch("importlib.import_module")``
would break subsequent patch setup because ``unittest.mock``'s own
path resolver calls ``importlib.import_module`` — self-referential fail.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from types import SimpleNamespace
from unittest.mock import patch

from carl_core.dependency_probe import (
    extract_corrupt_sibling,
    probe,
)


def _fake_module(version: str | None = None) -> SimpleNamespace:
    m = SimpleNamespace()
    if version is not None:
        setattr(m, "__version__", version)
    return m


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_status_ok() -> None:
    with (
        patch.object(importlib, "import_module", return_value=_fake_module("1.0.0")),
        patch.object(importlib.metadata, "version", return_value="1.0.0"),
    ):
        r = probe("xyz")
    assert r.status == "ok"
    assert r.healthy
    assert not r.needs_repair
    assert r.repair_command == ""
    assert r.import_version == "1.0.0"
    assert r.metadata_version == "1.0.0"


# ---------------------------------------------------------------------------
# Synthesized failure modes
# ---------------------------------------------------------------------------


def test_status_missing() -> None:
    """Not installed: import ImportError + metadata PackageNotFoundError."""
    with (
        patch.object(
            importlib,
            "import_module",
            side_effect=ImportError("No module named 'xyz'"),
        ),
        patch.object(
            importlib.metadata,
            "version",
            side_effect=importlib.metadata.PackageNotFoundError("xyz"),
        ),
    ):
        r = probe("xyz")
    assert r.status == "missing"
    assert r.is_missing
    assert r.repair_command == "pip install xyz"


def test_status_import_error_with_metadata_present() -> None:
    """Install half-broken: ImportError but metadata says version X is there."""
    with (
        patch.object(importlib, "import_module", side_effect=ImportError("broken .so")),
        patch.object(importlib.metadata, "version", return_value="2.3.4"),
    ):
        r = probe("xyz")
    assert r.status == "import_error"
    assert r.needs_repair
    assert "--force-reinstall" in r.repair_command
    assert "--no-deps" not in r.repair_command


def test_status_import_value_error_hf_scenario() -> None:
    """The HF bug: `import transformers` raises ValueError about huggingface-hub."""
    with patch.object(
        importlib,
        "import_module",
        side_effect=ValueError(
            "Unable to compare versions for huggingface-hub>=1.3.0,<2.0: "
            "need=1.3.0 found=None. This is unusual. Consider reinstalling huggingface-hub."
        ),
    ):
        r = probe("transformers")
    assert r.status == "import_value_error"
    assert r.needs_repair
    assert r.import_error is not None
    assert "Unable to compare" in r.import_error


def test_status_metadata_corrupt_returns_none() -> None:
    """Import OK + metadata.version() returns None — the exact HF corruption mode.

    Reproduces the stale-dist-info poisoning observed on Tej's machine:
    ``huggingface_hub-1.5.0.dist-info`` had no METADATA file; importlib.metadata
    returned None instead of raising PackageNotFoundError.
    """
    with (
        patch.object(importlib, "import_module", return_value=_fake_module("1.9.2")),
        patch.object(importlib.metadata, "version", return_value=None),
    ):
        r = probe("huggingface_hub")
    assert r.status == "metadata_corrupt"
    assert r.needs_repair
    assert "--force-reinstall --no-deps" in r.repair_command
    assert r.import_version == "1.9.2"
    assert r.metadata_version is None


def test_status_metadata_corrupt_lookup_exception() -> None:
    """Import OK + metadata.version() raises non-PackageNotFoundError."""
    with (
        patch.object(importlib, "import_module", return_value=_fake_module("1.0.0")),
        patch.object(
            importlib.metadata,
            "version",
            side_effect=ValueError("invalid literal for int() with base 10: ''"),
        ),
    ):
        r = probe("xyz")
    assert r.status == "metadata_corrupt"
    assert r.metadata_error is not None
    assert "ValueError" in r.metadata_error


def test_status_metadata_missing_when_not_registered() -> None:
    """Manual copy into site-packages: import works, metadata has no entry.

    This is distinct from corruption: the user intentionally dropped files in,
    or pip's registry disagrees. ``needs_repair`` stays False so we don't
    clobber the user's choice; the status is purely informational.
    """
    with (
        patch.object(importlib, "import_module", return_value=_fake_module("1.0.0")),
        patch.object(
            importlib.metadata,
            "version",
            side_effect=importlib.metadata.PackageNotFoundError("xyz"),
        ),
    ):
        r = probe("xyz")
    assert r.status == "metadata_missing"
    assert not r.needs_repair  # deliberate — see DepProbeResult.needs_repair docstring
    assert r.import_ok


def test_status_version_mismatch() -> None:
    """__version__ != metadata version → mismatch."""
    with (
        patch.object(importlib, "import_module", return_value=_fake_module("1.0.0")),
        patch.object(importlib.metadata, "version", return_value="2.0.0"),
    ):
        r = probe("xyz")
    assert r.status == "version_mismatch"
    assert r.needs_repair


# ---------------------------------------------------------------------------
# Normalization + import-name override
# ---------------------------------------------------------------------------


def test_normalization_hyphen_and_underscore_collapse() -> None:
    """Hyphens and underscores normalize to the same form."""
    with (
        patch.object(importlib, "import_module", side_effect=ImportError),
        patch.object(
            importlib.metadata,
            "version",
            side_effect=importlib.metadata.PackageNotFoundError,
        ),
    ):
        r1 = probe("huggingface-hub")
        r2 = probe("huggingface_hub")
        r3 = probe("Huggingface.Hub")
    assert r1.normalized_name == r2.normalized_name == r3.normalized_name == "huggingface-hub"


def test_import_name_override() -> None:
    """Pillow's distribution name is 'pillow' but import is 'PIL'."""
    with (
        patch.object(
            importlib,
            "import_module",
            return_value=_fake_module("10.0.0"),
        ) as imp,
        patch.object(importlib.metadata, "version", return_value="10.0.0"),
    ):
        r = probe("Pillow", import_name="PIL")
    imp.assert_called_with("PIL")
    assert r.normalized_name == "pillow"
    assert r.status == "ok"


# ---------------------------------------------------------------------------
# extract_corrupt_sibling parser
# ---------------------------------------------------------------------------


def test_extract_corrupt_sibling_from_hf_error() -> None:
    msg = "Unable to compare versions for huggingface-hub>=1.3.0,<2.0: need=1.3.0 found=None"
    assert extract_corrupt_sibling(msg) == "huggingface-hub"


def test_extract_corrupt_sibling_case_insensitive() -> None:
    msg = "unable to compare versions for numpy>=1.24"
    assert extract_corrupt_sibling(msg) == "numpy"


def test_extract_corrupt_sibling_none_when_no_match() -> None:
    assert extract_corrupt_sibling("ImportError: No module named 'xyz'") is None
    assert extract_corrupt_sibling("") is None


# ---------------------------------------------------------------------------
# probe() never raises
# ---------------------------------------------------------------------------


def test_probe_captures_exotic_import_error() -> None:
    """Arbitrary exception during import → captured, not re-raised."""
    with (
        patch.object(importlib, "import_module", side_effect=RuntimeError("wild")),
        patch.object(
            importlib.metadata,
            "version",
            side_effect=importlib.metadata.PackageNotFoundError("xyz"),
        ),
    ):
        r = probe("xyz")
    assert not r.import_ok
    assert r.status in ("import_error", "missing")
    assert r.import_error is not None
    assert "RuntimeError" in r.import_error


# ---------------------------------------------------------------------------
# Live probe (no mocks) — sanity check against installed carl-core itself
# ---------------------------------------------------------------------------


def test_live_probe_self() -> None:
    """Probing carl-core (which must be installed for these tests to run)
    should never crash and should classify as ok or metadata_corrupt
    depending on the editable-install state."""
    r = probe("carl-core")
    assert r.normalized_name == "carl-core"
    assert r.import_ok  # carl-core must be importable for the test to run
