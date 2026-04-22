---
last_updated: 2026-04-22
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.17.1-planned
classification: internal — dep-probe doctrine for optional-dep surfaces
---

# Dependency probe doctrine (v0.17.1+)

Every surface that checks an optional dependency — ``carl init`` offering
to install extras, ``carl doctor`` surfacing environment issues,
``carl train`` gating on torch/transformers — goes through the same
classifier: ``carl_core.dependency_probe.probe()``. This doc pins why
and how.

## The problem it solves

Naive pattern:

```python
try:
    import torch
    import transformers
    return True
except ImportError:
    return False
```

This catches "not installed". It does NOT catch:

- **Sibling-dep metadata corruption.** ``import transformers`` raises
  ``ValueError`` (not ``ImportError``) from
  ``dependency_versions_check.py`` when ``huggingface_hub`` has a
  stale ``.dist-info`` directory. Escapes the naive probe; crashes
  the wizard. This was the live trigger for v0.17.1.
- **Empty metadata.** ``importlib.metadata.version(pkg)`` returns
  ``None`` instead of raising when the ``.dist-info`` exists but
  the METADATA file is corrupt or missing Metadata-Version. Escapes
  ``try/except PackageNotFoundError``.
- **Import/metadata version mismatch.** Code loaded is 2.0 but pip
  says 0.12 — common after partial upgrade.
- **Manual-copy installs.** ``import`` works, pip has no record.
  Not "broken" — intentional — but needs different UX.

## The classifier

```python
from carl_core.dependency_probe import probe, DepProbeResult, ProbeStatus

result = probe("huggingface-hub")
# result.status is one of:
#   "ok"                  — healthy
#   "missing"             — not installed at all
#   "import_error"        — ImportError + metadata present
#   "import_value_error"  — ValueError (sibling-dep issue)
#   "metadata_missing"    — installed but not in pip's registry
#   "metadata_corrupt"    — installed but metadata returned None / raised
#   "version_mismatch"    — __version__ != metadata.version
```

Plus three derived properties:

- ``healthy`` = ``status == "ok"``
- ``needs_repair`` = broken state where force-reinstall helps
  (excludes ``metadata_missing`` deliberately — it's intentional)
- ``is_missing`` = ``status == "missing"``

And a concrete remediation:

- ``repair_command: str`` — the exact shell command to fix this
  state (``""`` when healthy, ``pip install ...`` otherwise).

## Sibling parsing

When the offending dep is hidden inside another package's error
message — the classic "transformers raises ValueError about
huggingface-hub" — we parse it out:

```python
from carl_core.dependency_probe import extract_corrupt_sibling

msg = exc_info.import_error  # "ValueError: Unable to compare versions for huggingface-hub>=1.3.0,<2.0"
sibling = extract_corrupt_sibling(msg)  # "huggingface-hub"
sibling_probe = probe(sibling)
# Auto-heal targets the sibling, not the symptom package.
```

## Auto-heal UX contract

When ``_offer_extras`` (or any future auto-heal surface) finds a
``needs_repair`` probe, the UX MUST:

1. **Show per-package state** — a one-line status for each probed
   dep: ``✓``, ``·``, or ``✗``.
2. **Show the exact repair command** before asking permission.
3. **Ask permission** — never silent auto-heal. Use ``ui.confirm``.
4. **Run the repair** via ``subprocess.run`` — no surprise commands.
5. **Re-probe after** — confirm the repair worked. If not, fall
   through to the fresh-install flow or manual guidance.
6. **Record the action** on the ``InteractionChain`` with typed
   input/output — gives ``carl doctor`` visibility into what happened.

## Freshness integration

``carl_studio/freshness.py::_check_packages`` now uses ``probe()``.
It emits two issue codes:

- ``carl.freshness.stale_pkg`` (warn) — installed version below
  recommended floor.
- ``carl.freshness.dep_corrupt`` (error) — import-side or
  metadata-side corruption; remediation is the probe's
  ``repair_command``.

``metadata_missing`` is silently skipped — user-intentional manual
installs don't warrant an error.

## Where to use probe()

Use ``probe()`` at every optional-dep check site. Examples:

| Site | Use |
|---|---|
| ``carl init``'s extras step | Probe torch/transformers/huggingface_hub; offer auto-heal on corruption, fresh-install on missing. |
| ``carl doctor`` | Surface ``dep_corrupt`` issues prominently. |
| ``carl train``'s backend gating | Probe the backend's deps before attempting to instantiate. |
| Any SDK bridge that lazy-imports an optional | Probe before import; fall back clearly if absent. |

## When NOT to use probe()

- **Hard dependencies.** We don't probe ``pydantic`` or ``typer``
  — they're required, import failure is fatal. Probes are for
  optional extras.
- **Hot paths.** Probe is fast (≤1ms per package) but still does
  file I/O via ``importlib.metadata``. Cache results across a
  single CLI invocation; don't call it in a loop per token.

## Test coverage

``packages/carl-core/tests/test_dependency_probe.py`` covers every
``ProbeStatus`` value via ``unittest.mock.patch.object`` against
``importlib.import_module`` and ``importlib.metadata.version``.
``tests/test_init_auto_heal.py`` covers the wizard's auto-heal fan-out.
``tests/test_freshness.py::TestCheckPackages::test_hf_style_corruption_emits_dep_corrupt_error``
pins the end-to-end freshness path for the exact HF scenario.

## References

- Implementation: ``packages/carl-core/src/carl_core/dependency_probe.py``
- Probe tests: ``packages/carl-core/tests/test_dependency_probe.py``
- Auto-heal wizard: ``src/carl_studio/cli/init.py::_offer_extras``
- Auto-heal tests: ``tests/test_init_auto_heal.py``
- Freshness integration: ``src/carl_studio/freshness.py::_check_packages``
- Plan doc: ``docs/v17_cli_ux_and_dep_probe_plan.md``
- Companion: ``docs/v17_cli_ux_doctrine.md``
