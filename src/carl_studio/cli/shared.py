"""Shared CLI utilities used across command modules."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from carl_studio import __version__
from carl_studio.console import CampConsole, get_console

from .apps import app

def _camp_header() -> None:
    """Print the Camp CARL banner with current theme."""
    c = get_console()
    c.banner(f"v{__version__}")


# ---------------------------------------------------------------------------
# Pipeline event renderer (used by --send-it)
# ---------------------------------------------------------------------------


def _render_pipeline_event(c: CampConsole, event: Any) -> None:
    """Render a SendItPipeline event to the console."""
    stage = event.stage.value
    if stage == "failed":
        c.error(event.message)
    elif stage == "done":
        c.ok(event.message)
    elif event.progress >= 1.0:
        c.ok(f"[{stage}] {event.message}")
    else:
        c.info(f"[{stage}] {event.message}")


def _slugify_identifier(value: str) -> str:
    """Convert a human/model identifier into a safe run slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "carl-run"


def _default_run_name(raw: dict[str, Any]) -> str:
    """Derive a run name for CLI-first training flows."""
    if raw.get("run_name"):
        return str(raw["run_name"])
    if raw.get("output_repo"):
        return _slugify_identifier(str(raw["output_repo"]).split("/")[-1])

    model_name = str(raw.get("base_model", "model")).split("/")[-1]
    method = str(raw.get("method", "train"))
    return _slugify_identifier(f"{model_name}-{method}")


def _normalize_training_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply CLI conveniences before constructing TrainingConfig."""
    from carl_studio.types.config import normalize_compute_target

    normalized = dict(raw)
    compute = normalized.get("compute_target")
    if isinstance(compute, str):
        normalized["compute_target"] = normalize_compute_target(compute)

    if not normalized.get("run_name") and (
        normalized.get("base_model") or normalized.get("output_repo")
    ):
        normalized["run_name"] = _default_run_name(normalized)

    return normalized


def _print_observe_usage(c: CampConsole) -> None:
    """Render concise observe usage examples."""
    c.blank()
    c.print("  [camp.muted]Usage:[/]")
    c.print("    carl observe --url https://wheattoast11-trackio.hf.space/ --run your-run")
    c.print(
        "    carl observe --url https://huggingface.co/spaces/owner/trackio --project your-project --run your-run"
    )
    c.print("    carl observe --file logs/train.jsonl")
    c.blank()


def _render_training_config_error(
    c: CampConsole, raw: dict[str, Any], exc: ValidationError
) -> None:
    """Convert Pydantic validation output into actionable CLI guidance."""
    from carl_studio.types.config import ComputeTarget

    missing_fields: list[str] = []
    invalid_fields: list[tuple[str, str]] = []
    for err in exc.errors():
        field = ".".join(str(part) for part in err.get("loc", ())) or "config"
        if err.get("type") == "missing":
            missing_fields.append(field)
        else:
            invalid_fields.append((field, err.get("msg", "Invalid value")))

    c.error("Training config is incomplete or invalid.")
    if missing_fields:
        c.info(f"Missing required fields: {', '.join(sorted(set(missing_fields)))}")
    for field, message in invalid_fields:
        c.info(f"{field}: {message}")

    valid_compute = ", ".join(ct.value for ct in ComputeTarget)
    c.blank()
    c.info(f"Valid compute targets: {valid_compute}")
    c.info("Aliases accepted on the CLI: a100 -> a100-large, a10g -> a10g-large")
    c.blank()

    example_model = str(raw.get("base_model", "Tesslate/OmniCoder-9B"))
    example_method = str(raw.get("method", "grpo"))
    c.print("  [camp.primary]Quick start[/]")
    c.print(
        "    carl train "
        f"--model {example_model} --method {example_method} "
        "--dataset your-org/your-dataset --output-repo your-org/your-model "
        "--compute a100-large"
    )
    c.info("Tip: run 'carl project init' to create carl.yaml interactively.")


def _render_extra_install_hint(
    c: CampConsole,
    extra: str,
    context: str,
    exc: Exception | None = None,
    quoted: bool = False,
) -> None:
    """Render a consistent optional-dependency hint."""
    from rich.markup import escape

    c.error(context)
    package = f"carl-studio[{extra}]"
    c.info(f"Install: pip install {escape(package)}")
    if quoted:
        shell_package = f"'carl-studio[{extra}]'"
        c.info(f"Install: pip install {escape(shell_package)}")
    if exc is not None:
        c.info(str(exc))


_COMMAND_CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    ("Core Workbench", ("start", "doctor", "project", "train", "run", "observe", "eval", "infer")),
    (
        "Data and Compute",
        ("config", "compute", "browse", "data", "checkpoint", "bundle"),
    ),
    ("Camp Platform", ("camp",)),
    ("Advanced and Lab", ("bench", "align", "learn", "chat", "mcp", "lab")),
]

_PROJECT_REQUIRED_FIELDS = ("base_model", "dataset_repo", "output_repo")
_START_DEPENDENCY_GROUPS: dict[str, tuple[str, ...]] = {
    "training": ("torch", "transformers"),
    "hf": ("huggingface_hub",),
    "tui": ("textual",),
    "diagnose": ("anthropic",),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_json_object(value: Any) -> dict[str, Any]:
    """Parse a JSON object if possible, otherwise return an empty dict."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _module_available(module_name: str) -> bool:
    """Return True if a module can be resolved without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None or module_name in sys.modules
    except ValueError:
        return module_name in sys.modules


def _read_yaml_mapping(path: Path) -> tuple[dict[str, Any], str | None]:
    """Read a YAML mapping, surfacing parse errors for UX commands."""
    import yaml

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        return {}, None
    except Exception as exc:
        return {}, str(exc)

    if data is None:
        return {}, None
    if not isinstance(data, dict):
        return {}, "top-level YAML must be a mapping"
    return data, None


def _summarize_project_config(path: Path | None) -> dict[str, Any]:
    """Inspect the nearest carl.yaml without assuming a specific schema."""
    if path is None:
        return {
            "path": None,
            "error": None,
            "name": "",
            "missing_fields": list(_PROJECT_REQUIRED_FIELDS),
            "has_training_fields": False,
        }

    raw, error = _read_yaml_mapping(path)
    missing_fields = [field for field in _PROJECT_REQUIRED_FIELDS if not raw.get(field)]
    return {
        "path": str(path),
        "path_name": path.name,
        "error": error,
        "name": str(raw.get("name", "")),
        "missing_fields": missing_fields,
        "has_training_fields": not error and not missing_fields,
        "raw": raw,
    }


def _registered_command_tree() -> dict[str, list[str]]:
    """Return the currently installed command tree."""
    from typer.main import get_command

    command = get_command(app)
    tree: dict[str, list[str]] = {}
    for name, cmd in command.commands.items():
        subcommands = sorted(cmd.commands) if hasattr(cmd, "commands") else []
        tree[name] = subcommands
    return tree


def _inventory_rows(command_tree: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Create grouped inventory rows from the registered command tree."""
    rows: list[tuple[str, str]] = []
    for title, commands in _COMMAND_CATEGORIES:
        labels: list[str] = []
        for name in commands:
            if name not in command_tree:
                continue
            children = command_tree[name]
            if children:
                preview = ", ".join(children[:4])
                if len(children) > 4:
                    preview += ", ..."
                labels.append(f"{name}[{preview}]")
            else:
                labels.append(name)
        if labels:
            rows.append((title, ", ".join(labels)))
    return rows


def _render_command_inventory(c: CampConsole, command_tree: dict[str, list[str]]) -> None:
    """Render the installed command inventory grouped by journey."""
    rows = _inventory_rows(command_tree)
    table = c.make_table("Journey", "Commands", title="Current Command Inventory")
    for title, commands in rows:
        table.add_row(title, commands)
    c.print(table)


def _camp_connection() -> dict[str, Any]:
    """Return the current carl.camp session state without raising."""
    try:
        from carl_studio.db import LocalDB

        db = LocalDB()
        return {
            "connected": bool(db.get_auth("jwt")),
            "supabase_configured": bool(db.get_config("supabase_url")),
        }
    except Exception:
        return {"connected": False, "supabase_configured": False}


def _build_readiness_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Derive actionable readiness flags from the raw startup summary."""
    project = summary["project"]
    global_config = summary["global_config"]
    dependencies = summary["dependencies"]
    auth = summary["auth"]

    blocking_issues: list[str] = []
    if global_config["error"]:
        blocking_issues.append(f"Fix invalid user config at {global_config['path']}")

    if project["error"] and project.get("path"):
        blocking_issues.append(f"Fix invalid project file at {project['path']}")
    elif not project["path"]:
        blocking_issues.append("Create a project file with 'carl project init'")
    elif project["missing_fields"]:
        missing = ", ".join(project["missing_fields"])
        blocking_issues.append(f"Fill missing project fields: {missing}")

    if not dependencies["training"]:
        blocking_issues.append("Install local training deps: pip install 'carl-studio[training]'")

    return {
        "guided_workbench": not blocking_issues,
        "remote_jobs": dependencies["hf"] and auth["hf_ready"],
        "live_observe": dependencies["tui"],
        "diagnose": dependencies["diagnose"] and auth["anthropic_ready"],
        "blocking_issues": blocking_issues,
    }


def _build_start_summary() -> dict[str, Any]:
    """Inspect readiness for the common first-run and daily workflows."""
    from carl_studio.settings import CARLSettings, GLOBAL_CONFIG, _find_local_config
    from carl_studio.theme import THEME_FILE
    from carl_studio.tier import _detect_hf_token

    settings = CARLSettings.load()
    effective_tier = settings.get_effective_tier()

    global_raw, global_error = _read_yaml_mapping(GLOBAL_CONFIG)
    local_config = _find_local_config()
    project = _summarize_project_config(local_config)

    dependency_groups = {
        name: all(_module_available(module) for module in modules)
        for name, modules in _START_DEPENDENCY_GROUPS.items()
    }

    command_tree = _registered_command_tree()

    summary: dict[str, Any] = {
        "tier": settings.tier.value,
        "effective_tier": effective_tier.value,
        "theme_configured": THEME_FILE.exists(),
        "global_config": {
            "path": str(GLOBAL_CONFIG),
            "exists": GLOBAL_CONFIG.is_file(),
            "error": global_error,
            "preset": str(global_raw.get("preset", settings.preset.value))
            if not global_error
            else "",
        },
        "project": project,
        "defaults": {
            "model": settings.default_model,
            "compute": settings.default_compute.value,
            "observe_source": settings.observe_defaults.default_source,
            "observe_poll": settings.observe_defaults.default_poll_interval,
            "trackio_url": settings.trackio_url or "",
        },
        "camp": _camp_connection(),
        "auth": {
            "hf_ready": bool(settings.hf_token or _detect_hf_token()),
            "anthropic_ready": bool(settings.anthropic_api_key),
        },
        "dependencies": dependency_groups,
        "command_inventory": command_tree,
    }
    summary["readiness"] = _build_readiness_summary(summary)
    summary["next_steps"] = _recommended_next_steps(summary)
    return summary


def _recommended_next_steps(summary: dict[str, Any]) -> list[str]:
    """Suggest the next highest-leverage commands based on current state."""
    steps: list[str] = []
    project = summary["project"]
    global_config = summary["global_config"]
    defaults = summary["defaults"]

    if global_config["error"]:
        steps.append(f"Fix invalid config at {global_config['path']}")
    elif not global_config["exists"]:
        steps.append("carl config init")

    if not summary["theme_configured"]:
        steps.append("carl setup")

    if project["error"] and project["path"]:
        steps.append(f"Fix invalid project file at {project['path']}")
    elif not project["path"]:
        steps.append("carl project init")
    elif project["missing_fields"]:
        missing = ", ".join(project["missing_fields"])
        steps.append(f"Fill project fields in {project['path_name']}: {missing}")
    else:
        steps.append(f"carl train --config {project['path_name']}")
        steps.append("carl run list")

    if not summary["auth"]["hf_ready"]:
        steps.append("hf auth login  # for remote jobs, gated models, and push")

    if defaults["trackio_url"]:
        steps.append("carl observe --run your-run")
    else:
        steps.append("carl observe --file logs/train.jsonl")

    deduped: list[str] = []
    for step in steps:
        if step not in deduped:
            deduped.append(step)
    return deduped[:4]


def _run_remote_id(run_record: dict[str, Any]) -> str:
    """Resolve the remote job identifier from a local run record."""
    result = _safe_json_object(run_record.get("result"))
    remote_id = run_record.get("remote_id") or result.get("hub_job_id") or ""
    return str(remote_id)


def _normalize_remote_status(status_value: str) -> str:
    """Map provider-specific job states onto the local run vocabulary."""
    return {
        "queued": "provisioning",
        "running": "training",
        "completed": "complete",
        "failed": "failed",
        "error": "failed",
        "canceled": "canceled",
    }.get(status_value, status_value)


def _persist_training_run(training_config: Any, run: Any, mode: str) -> None:
    """Best-effort persistence for train/pipeline runs in the local ledger."""
    from carl_studio.db import LocalDB

    now = _now_iso()
    db = LocalDB()
    existing = db.get_run(run.id)
    row = {
        "id": run.id,
        "model_id": training_config.base_model,
        "mode": mode,
        "status": run.phase.value,
        "hardware": training_config.compute_target.value,
        "config": training_config.model_dump(mode="json"),
        "result": {
            "phase": run.phase.value,
            "hub_job_id": run.hub_job_id,
            "error_message": run.error_message,
            "checkpoint_steps": list(getattr(run, "checkpoint_steps", [])),
            "output_repo": training_config.output_repo,
            "method": training_config.method.value,
        },
        "started_at": existing.get("started_at")
        if existing and existing.get("started_at")
        else now,
        "completed_at": now if run.phase.value in {"complete", "failed"} else None,
        "remote_id": run.hub_job_id,
    }

    if existing:
        db.update_run(run.id, {k: v for k, v in row.items() if k != "id"})
    else:
        db.insert_run(row)


def _lookup_local_run(run_or_job_id: str) -> dict[str, Any] | None:
    """Return a local run record if the identifier matches a stored run."""
    from carl_studio.db import LocalDB

    return LocalDB().get_run(run_or_job_id)


def _render_local_run_record(
    c: CampConsole, run_record: dict[str, Any], title: str = "Local Run"
) -> None:
    """Render a stored run record in a compact, human-readable way."""
    config = _safe_json_object(run_record.get("config"))
    result = _safe_json_object(run_record.get("result"))
    remote_id = _run_remote_id(run_record)

    pairs = [
        ("Run ID", run_record.get("id", "")),
        ("Mode", run_record.get("mode", "")),
        ("Status", run_record.get("status", "")),
        ("Model", run_record.get("model_id", "")),
        ("Hardware", run_record.get("hardware", "")),
    ]
    if remote_id:
        pairs.append(("Remote Job", remote_id))
    if config.get("output_repo"):
        pairs.append(("Output Repo", str(config["output_repo"])))
    if run_record.get("created_at"):
        pairs.append(("Created", str(run_record["created_at"])))
    if run_record.get("completed_at"):
        pairs.append(("Completed", str(run_record["completed_at"])))

    c.blank()
    c.config_block(pairs, title=title)
    if result.get("error_message"):
        c.warn(str(result["error_message"]))
    c.blank()


def _resolve_run_reference(run_or_job_id: str) -> tuple[dict[str, Any] | None, str]:
    """Resolve a local run ID to its remote job ID when available."""
    local_run = _lookup_local_run(run_or_job_id)
    if local_run is None:
        return None, run_or_job_id

    remote_id = _run_remote_id(local_run)
    return local_run, remote_id


def _project_status_label(project: dict[str, Any]) -> str:
    """Return a compact status label for the current project file."""
    if project["error"]:
        return f"invalid ({project['error']})"
    if not project["path"]:
        return "none found"
    if project["missing_fields"]:
        return "missing: " + ", ".join(project["missing_fields"])
    return "ready for train"


def _camp_status_label(summary: dict[str, Any]) -> str:
    """Return a compact carl.camp connection label."""
    camp = summary["camp"]
    if not camp["connected"]:
        return "local-only"
    return f"connected ({summary['effective_tier']})"


