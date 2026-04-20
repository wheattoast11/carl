"""Training run delta — aggregate + per-step comparison between two runs.

This module is intentionally import-light: no torch/transformers/trl.
It reads the persisted run dict (``LocalDB.get_run``) and its metrics
stream (``LocalDB.get_metrics``) and produces a structured
:class:`RunDiffReport` that the CLI renders via :class:`CampConsole`.

Aggregate fields compared:
    - ``phi_mean`` (coherence mean)
    - ``q_hat`` (contraction estimate)
    - ``crystallization_count``
    - ``contraction_holds``
    - ``sample_size``

The ``result`` column is persisted by :mod:`carl_studio.db` as JSON, but
older rows may carry a raw string or ``None``. :func:`summarize_run`
handles all three shapes.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field


class RunSummary(BaseModel):
    """Aggregate view of a single run."""

    run_id: str
    phi_mean: float | None = None
    q_hat: float | None = None
    sample_size: int | None = None
    contraction_holds: bool | None = None
    crystallization_count: int | None = None
    step_count: int = 0
    status: str | None = None


class RunDiffReport(BaseModel):
    """Delta between two runs.

    ``contraction_holds_change`` is ``None`` when both sides agree or
    either side is missing; otherwise it is a short literal like
    ``"True->False"``.
    """

    a: RunSummary
    b: RunSummary
    phi_mean_delta: float | None = None
    q_hat_delta: float | None = None
    crystallization_delta: int | None = None
    sample_size_delta: int | None = None
    contraction_holds_change: str | None = None
    divergence_step: int | None = None
    step_rows: list[dict[str, Any]] = Field(default_factory=list)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_result(result: Any) -> dict[str, Any]:
    """Return ``result`` as a dict regardless of how it was persisted.

    ``LocalDB`` normally JSON-decodes ``result`` on read, but defensive
    handling keeps older/raw inputs usable.
    """
    if result is None:
        return {}
    if isinstance(result, dict):
        # Stored run rows always use string keys; narrow without churning
        # the dict. Pyright treats the incoming ``dict`` as unknown-typed,
        # so we explicitly cast via ``dict()``.
        typed: dict[str, Any] = dict(result)  # type: ignore[arg-type]
        return typed
    if isinstance(result, str):
        try:
            parsed: Any = json.loads(result)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
        if isinstance(parsed, dict):
            typed_parsed: dict[str, Any] = dict(parsed)  # type: ignore[arg-type]
            return typed_parsed
        return {}
    return {}


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _crystallization_count(result: dict[str, Any]) -> int | None:
    """Prefer explicit ``crystallization_count``; fall back to list length."""
    explicit = _as_int(result.get("crystallization_count"))
    if explicit is not None:
        return explicit
    crystallizations: Any = result.get("crystallizations")
    if isinstance(crystallizations, list):
        return len(crystallizations)  # type: ignore[arg-type]
    return None


def _metric_phi(data: Any) -> float | None:
    """Extract a phi-like value from a metrics row's ``data`` payload."""
    if not isinstance(data, dict):
        return None
    typed: dict[str, Any] = dict(data)  # type: ignore[arg-type]
    for key in ("phi", "phi_mean"):
        val = _as_float(typed.get(key))
        if val is not None:
            return val
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarize_run(run: dict[str, Any], metrics: list[dict[str, Any]]) -> RunSummary:
    """Build a :class:`RunSummary` from the persisted run row + metrics stream."""
    result = _coerce_result(run.get("result"))
    run_id = str(run.get("id") or "")
    status = run.get("status")
    status_str = str(status) if status is not None else None

    return RunSummary(
        run_id=run_id,
        phi_mean=_as_float(result.get("phi_mean")),
        q_hat=_as_float(result.get("q_hat")),
        sample_size=_as_int(result.get("sample_size")),
        contraction_holds=_as_bool(result.get("contraction_holds")),
        crystallization_count=_crystallization_count(result),
        step_count=len(metrics) if metrics else 0,
        status=status_str,
    )


def _delta_float(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def _delta_int(a: int | None, b: int | None) -> int | None:
    if a is None or b is None:
        return None
    return b - a


def _contraction_change(a: bool | None, b: bool | None) -> str | None:
    if a is None or b is None:
        return None
    if a == b:
        return None
    return f"{a}->{b}"


def _first_divergence(
    metrics_a: list[dict[str, Any]],
    metrics_b: list[dict[str, Any]],
    threshold: float,
) -> int | None:
    n = min(len(metrics_a), len(metrics_b))
    for i in range(n):
        phi_a = _metric_phi(metrics_a[i].get("data"))
        phi_b = _metric_phi(metrics_b[i].get("data"))
        if phi_a is None or phi_b is None:
            continue
        if abs(phi_a - phi_b) > threshold:
            return i
    return None


def _build_step_rows(
    metrics_a: list[dict[str, Any]],
    metrics_b: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    n = min(len(metrics_a), len(metrics_b))
    rows: list[dict[str, Any]] = []
    for i in range(n):
        phi_a = _metric_phi(metrics_a[i].get("data"))
        phi_b = _metric_phi(metrics_b[i].get("data"))
        delta = (phi_b - phi_a) if (phi_a is not None and phi_b is not None) else None
        rows.append({"step": i, "phi_a": phi_a, "phi_b": phi_b, "delta": delta})
    return rows


def compute_diff(
    run_a: dict[str, Any],
    metrics_a: list[dict[str, Any]],
    run_b: dict[str, Any],
    metrics_b: list[dict[str, Any]],
    *,
    steps: bool = False,
    divergence_threshold: float = 0.1,
) -> RunDiffReport:
    """Compare two runs and return a :class:`RunDiffReport`.

    ``divergence_threshold`` gates ``divergence_step``: the first aligned
    index where ``|phi_a - phi_b|`` strictly exceeds the threshold. When
    ``steps=True`` the report also includes per-step rows up to the
    shorter of the two metric streams.
    """
    if divergence_threshold < 0:
        raise ValueError("divergence_threshold must be non-negative")

    a = summarize_run(run_a, metrics_a)
    b = summarize_run(run_b, metrics_b)

    report = RunDiffReport(
        a=a,
        b=b,
        phi_mean_delta=_delta_float(a.phi_mean, b.phi_mean),
        q_hat_delta=_delta_float(a.q_hat, b.q_hat),
        crystallization_delta=_delta_int(a.crystallization_count, b.crystallization_count),
        sample_size_delta=_delta_int(a.sample_size, b.sample_size),
        contraction_holds_change=_contraction_change(a.contraction_holds, b.contraction_holds),
        divergence_step=_first_divergence(metrics_a, metrics_b, divergence_threshold),
    )

    if steps:
        report.step_rows = _build_step_rows(metrics_a, metrics_b)

    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_float(value: float | None, fmt: str = ".4f") -> str:
    if value is None:
        return "-"
    return f"{value:{fmt}}"


def _fmt_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def _fmt_bool(value: bool | None) -> str:
    return "-" if value is None else str(value)


def _fmt_delta_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}"


def _fmt_delta_int(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:+d}"


def render_diff(report: RunDiffReport, console: Any | None = None) -> None:
    """Render a :class:`RunDiffReport` via :class:`CampConsole`.

    A fresh console is constructed when ``console`` is ``None`` so the
    function is usable from notebooks and tests without the CLI wiring.
    """
    if console is None:
        from carl_studio.console import CampConsole

        console = CampConsole()

    a = report.a
    b = report.b

    console.blank()
    console.header(f"Run Diff  --  {a.run_id}  vs  {b.run_id}")
    console.blank()

    table = console.make_table("Metric", "A", "B", "Delta", title="Aggregate")
    table.add_row(
        "phi_mean",
        _fmt_float(a.phi_mean),
        _fmt_float(b.phi_mean),
        _fmt_delta_float(report.phi_mean_delta),
    )
    table.add_row(
        "q_hat",
        _fmt_float(a.q_hat),
        _fmt_float(b.q_hat),
        _fmt_delta_float(report.q_hat_delta),
    )
    table.add_row(
        "sample_size",
        _fmt_int(a.sample_size),
        _fmt_int(b.sample_size),
        _fmt_delta_int(report.sample_size_delta),
    )
    table.add_row(
        "crystallization_count",
        _fmt_int(a.crystallization_count),
        _fmt_int(b.crystallization_count),
        _fmt_delta_int(report.crystallization_delta),
    )
    table.add_row(
        "contraction_holds",
        _fmt_bool(a.contraction_holds),
        _fmt_bool(b.contraction_holds),
        report.contraction_holds_change or "-",
    )
    table.add_row(
        "step_count",
        _fmt_int(a.step_count),
        _fmt_int(b.step_count),
        _fmt_delta_int(_delta_int(a.step_count, b.step_count)),
    )
    console.print(table)

    if report.divergence_step is not None:
        console.blank()
        console.info(f"First divergence at step {report.divergence_step}")

    if report.step_rows:
        console.blank()
        step_table = console.make_table("Step", "phi_a", "phi_b", "Delta", title="Per-Step")
        for row in report.step_rows:
            step_table.add_row(
                str(row["step"]),
                _fmt_float(row.get("phi_a")),
                _fmt_float(row.get("phi_b")),
                _fmt_delta_float(row.get("delta")),
            )
        console.print(step_table)

    console.blank()
