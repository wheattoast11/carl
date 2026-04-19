"""Supply-chain coverage tests for ``pyproject.toml`` optional extras.

Guards the invariant that the ``[all]`` meta-extra references every declared
extra except:

* meta-extras themselves (``all``, ``quickstart``) — they are aggregators.
* extras the project has explicitly declared in conflict with ``all`` via
  ``[tool.uv].conflicts`` (upstream dependency graphs make them mutually
  exclusive with a full install).

Also pins the ``quickstart`` ergonomics contract.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, cast

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _load() -> dict[str, Any]:
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))


def _extras(data: dict[str, Any]) -> dict[str, list[str]]:
    project = cast(dict[str, Any], data["project"])
    extras_raw = cast(dict[str, Any], project["optional-dependencies"])
    normalised: dict[str, list[str]] = {}
    for name, raw_specs in extras_raw.items():
        assert isinstance(name, str)
        assert isinstance(raw_specs, list)
        specs_list: list[str] = []
        for spec in cast(list[Any], raw_specs):
            assert isinstance(spec, str)
            specs_list.append(spec)
        normalised[name] = specs_list
    return normalised


def _parse_extra_refs(spec: str) -> set[str]:
    """Extract ``x,y,z`` from a self-reference like ``carl-studio[x,y,z]``."""
    match = re.search(r"\[([^\]]+)\]", spec)
    if not match:
        return set()
    return {part.strip() for part in match.group(1).split(",") if part.strip()}


def _conflict_groups(data: dict[str, Any]) -> list[list[str]]:
    """Return each ``[tool.uv].conflicts`` group as a list of extra names."""
    tool_raw: object = data.get("tool")
    if not isinstance(tool_raw, dict):
        return []
    tool = cast(dict[str, Any], tool_raw)
    uv_raw: object = tool.get("uv")
    if not isinstance(uv_raw, dict):
        return []
    uv_cfg = cast(dict[str, Any], uv_raw)
    raw_conflicts: object = uv_cfg.get("conflicts")
    if not isinstance(raw_conflicts, list):
        return []
    groups: list[list[str]] = []
    for raw_group in cast(list[Any], raw_conflicts):
        assert isinstance(raw_group, list)
        names: list[str] = []
        for raw_item in cast(list[Any], raw_group):
            assert isinstance(raw_item, dict)
            item = cast(dict[str, Any], raw_item)
            name_raw = item.get("extra")
            assert isinstance(name_raw, str)
            names.append(name_raw)
        groups.append(names)
    return groups


def _extras_conflicting_with(data: dict[str, Any], target: str) -> set[str]:
    out: set[str] = set()
    for group in _conflict_groups(data):
        if target in group:
            out.update(n for n in group if n != target)
    return out


def test_all_extra_covers_every_declared_extra() -> None:
    data = _load()
    extras = _extras(data)

    # Meta-extras are aggregators, not leaves we need to re-list in [all].
    meta_extras = {"all", "quickstart"}
    declared = set(extras.keys()) - meta_extras
    excluded_by_conflict = _extras_conflicting_with(data, "all")
    required = declared - excluded_by_conflict

    all_refs: set[str] = set()
    for spec in extras["all"]:
        all_refs.update(_parse_extra_refs(spec))

    missing = required - all_refs
    assert not missing, (
        f"[all] does not cover: {sorted(missing)}. "
        "Either add them to [all] or declare a conflict in [tool.uv].conflicts."
    )

    unexpected = all_refs & excluded_by_conflict
    assert not unexpected, (
        f"[all] references extras declared in conflict with it: {sorted(unexpected)}. "
        "Remove them from [all] or drop the conflict declaration."
    )


def test_quickstart_extra_exists_and_is_minimal() -> None:
    data = _load()
    extras = _extras(data)

    assert "quickstart" in extras, "quickstart extra missing from pyproject.toml"
    specs = extras["quickstart"]
    assert specs, "quickstart extra must be non-empty"

    referenced: set[str] = set()
    for spec in specs:
        referenced.update(_parse_extra_refs(spec))

    # Contract: quickstart bundles training + hf + observe. That is the
    # ergonomic default documented in the README and docs/INSTALL.md.
    expected = {"training", "hf", "observe"}
    assert expected.issubset(referenced), (
        f"quickstart must reference at least {sorted(expected)}; got {sorted(referenced)}"
    )


def test_declared_conflicts_reference_existing_extras() -> None:
    data = _load()
    extras = _extras(data)
    groups = _conflict_groups(data)

    for group in groups:
        assert len(group) >= 2, f"conflict group must have 2+ entries: {group}"
        for name in group:
            assert name in extras, (
                f"conflict references unknown extra {name!r}; "
                f"known extras: {sorted(extras.keys())}"
            )
