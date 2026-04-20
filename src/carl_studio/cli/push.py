"""``carl push`` — ship a completed run's checkpoint to Hugging Face Hub.

First-class verb for "ship the model". Complements ``carl train`` (which
optionally pushes as a side-effect when ``output_repo`` is set in
``carl.yaml``). Call sites that want explicit control use this verb.

Accepts either:
- a local run id (looked up in :class:`LocalDB`), or
- a literal local checkpoint directory path.

The repo is resolved from the positional ``<repo>`` argument, falling
back to the ``--repo`` flag, falling back to the run record's
``config.output_repo`` when both are absent. This keeps the new
signature (positional ``run_id repo``) first-class while preserving
backwards compatibility with callers that used ``--repo``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, cast

import typer

from carl_studio.console import get_console

from .apps import app
from .shared import _render_extra_install_hint  # pyright: ignore[reportPrivateUsage]


@app.command(name="push")
def push_cmd(
    run_id_or_path: Annotated[
        str,
        typer.Argument(
            help="Run id (from `carl run list`) OR local checkpoint directory / file path",
        ),
    ],
    repo: Annotated[
        str,
        typer.Argument(
            help=(
                "Target HF repo (e.g. 'your-org/your-adapter'). "
                "Optional: if omitted, falls back to --repo, then to the run's output_repo."
            ),
        ),
    ] = "",
    repo_option: Annotated[
        str,
        typer.Option(
            "--repo",
            "-r",
            help="HF repo id (backwards-compat alias for the second positional arg)",
        ),
    ] = "",
    base_model: Annotated[
        str,
        typer.Option(
            "--base-model",
            "-b",
            help="Base model ID to embed in the generated model card",
        ),
    ] = "",
    method: Annotated[
        str,
        typer.Option(
            "--method",
            help="Training method recorded in the model card (sft, grpo, dpo, ...)",
        ),
    ] = "grpo",
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            "-d",
            help="Training dataset id recorded in the model card",
        ),
    ] = "",
    private: Annotated[
        bool,
        typer.Option(
            "--private/--public",
            help="Create a private repo (default: private)",
        ),
    ] = True,
    commit_message: Annotated[
        str,
        typer.Option(
            "--message",
            "-m",
            help="Commit message (default: auto-generated from run id)",
        ),
    ] = "",
) -> None:
    """Ship a completed run's checkpoint to Hugging Face Hub.

    Examples:

      carl push <run_id> your-org/your-adapter

      carl push ./carl-grpo-abc/checkpoint-100 your-org/your-adapter --private

      carl push <run_id> --repo your-org/your-adapter --method grpo
    """
    c = get_console()

    # --- Resolve repo (positional > flag > run record's output_repo) ---
    run_record = _lookup_run_record(run_id_or_path)

    resolved_repo = repo or repo_option or ""
    if not resolved_repo and run_record is not None:
        config_any: Any = run_record.get("config") or {}
        if isinstance(config_any, dict):
            candidate = cast("dict[str, Any]", config_any).get("output_repo")
            if candidate:
                resolved_repo = str(candidate)

    if not resolved_repo:
        c.error_with_hint(
            "no target repo resolved",
            detail=(
                "Provide <repo> as the second positional argument, pass --repo, "
                "or ensure the run's config.output_repo is set."
            ),
            hint="carl push <run_id> your-org/your-adapter",
            code="carl.push.repo_missing",
        )
        raise typer.Exit(2)

    if "/" not in resolved_repo:
        c.error_with_hint(
            f"invalid repo id: {resolved_repo!r}",
            detail="Repo ids must follow the 'namespace/name' convention.",
            hint="e.g. 'your-org/your-adapter'",
            code="carl.push.repo_invalid",
        )
        raise typer.Exit(2)

    # --- Resolve local path ---
    local_path = _resolve_run_path(run_id_or_path, run_record)
    if local_path is None:
        c.error_with_hint(
            f"no checkpoint found for {run_id_or_path!r}",
            detail=(
                "Neither a literal filesystem path nor a local run record with "
                "a resolvable output_dir could be found."
            ),
            hint="Inspect with: carl run show <run_id>",
            code="carl.push.path_missing",
        )
        raise typer.Exit(2)

    if not local_path.exists():
        c.error(f"path does not exist: {local_path}")
        raise typer.Exit(2)

    # --- Import HF (optional extra) ---
    try:
        from huggingface_hub import HfApi, get_token
    except ImportError as exc:
        _render_extra_install_hint(
            c,
            "hf",
            "Hugging Face Hub support is not installed.",
            exc,
        )
        raise typer.Exit(1)

    # --- Optional metadata path (legacy behavior): if all metadata flags
    # are provided we delegate to the existing model-card builder so the
    # `carl push` UX that ships a README with coherence metadata is
    # preserved. Otherwise we do a plain upload_folder / upload_file.
    use_metadata = bool(base_model or dataset)

    message = commit_message or f"carl push — {run_id_or_path}"

    c.info(f"Uploading {local_path} -> {resolved_repo}")

    if use_metadata and local_path.is_dir():
        try:
            from carl_studio.hub.models import (
                push_with_metadata,  # pyright: ignore[reportUnknownVariableType]
            )
        except ImportError as exc:
            _render_extra_install_hint(
                c,
                "hf",
                "Hub publishing support is not installed.",
                exc,
            )
            raise typer.Exit(1)

        import anyio

        push_fn: Any = push_with_metadata  # pyright: ignore[reportUnknownVariableType]
        try:
            url = anyio.run(
                push_fn,
                str(local_path),
                resolved_repo,
                base_model,
                method or "grpo",
                dataset,
                None,  # coherence_metrics
                private,
            )
        except Exception as exc:
            c.error(f"push failed: {exc}")
            raise typer.Exit(3) from exc
        c.ok(f"pushed {resolved_repo}  (private={private})")
        c.info(f"View: {url}")
        return

    # --- Plain path: upload folder or single file ---
    try:
        api = HfApi(token=get_token())
        api.create_repo(resolved_repo, private=private, exist_ok=True)
        if local_path.is_dir():
            api.upload_folder(
                folder_path=str(local_path),
                repo_id=resolved_repo,
                commit_message=message,
                ignore_patterns=["**/__pycache__/**", ".DS_Store"],
            )
        else:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=local_path.name,
                repo_id=resolved_repo,
                commit_message=message,
            )
    except Exception as exc:
        c.error(f"push failed: {exc}")
        raise typer.Exit(3) from exc

    c.ok(f"pushed {resolved_repo}  (private={private})")
    c.info(f"View: https://huggingface.co/{resolved_repo}")


def _lookup_run_record(run_id_or_path: str) -> dict[str, Any] | None:
    """Return a LocalDB run record for ``run_id_or_path``, or None.

    Best-effort: a literal filesystem path will typically miss the DB
    lookup, which is fine — the caller just treats it as a path.
    """
    try:
        from carl_studio.db import LocalDB

        db = LocalDB()
        # `LocalDB.get_run` returns untyped `dict | None`; narrow at the boundary.
        row_any: Any = db.get_run(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            run_id_or_path,
        )
        if not row_any:
            return None
        return cast("dict[str, Any]", row_any)
    except Exception:
        return None


def _resolve_run_path(
    run_id_or_path: str,
    run_record: dict[str, Any] | None,
) -> Path | None:
    """Resolve a run id / path to a local checkpoint ``Path``.

    Priority:
    1. Literal filesystem path (``.`` / ``..`` / absolute) if it exists.
    2. Run record → ``result.output_dir`` (from pipeline/trainer).
    3. Run record → ``config.output_dir`` if the trainer stashes one.
    4. Conventional defaults derived from run id (``carl-sft-<id>``,
       ``carl-grpo-<id>``) relative to the current working directory.
    """
    # 1. Literal path.
    candidate = Path(run_id_or_path).expanduser()
    if candidate.exists():
        return candidate

    # 2 / 3. Run record fields.
    if run_record is not None:
        result_any: Any = run_record.get("result")
        if isinstance(result_any, dict):
            out_dir_any: Any = cast("dict[str, Any]", result_any).get("output_dir")
            if isinstance(out_dir_any, str) and out_dir_any:
                p = Path(out_dir_any).expanduser()
                if p.exists():
                    return p

        config_any: Any = run_record.get("config")
        if isinstance(config_any, dict):
            out_dir_any = cast("dict[str, Any]", config_any).get("output_dir")
            if isinstance(out_dir_any, str) and out_dir_any:
                p = Path(out_dir_any).expanduser()
                if p.exists():
                    return p

    # 4. Conventional defaults. The trainer writes to
    # ``carl-sft-<run_id>`` / ``carl-grpo-<run_id>`` next to the process
    # cwd, so fall back to those when the DB row does not spell it out.
    for prefix in ("carl-grpo-", "carl-sft-"):
        p = Path(f"{prefix}{run_id_or_path}")
        if p.exists():
            return p

    return None


__all__ = ["push_cmd"]
