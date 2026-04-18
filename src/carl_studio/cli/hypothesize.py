"""`carl hypothesize` -- turn a natural-language hypothesis into a carl.yaml config.

Uses CARLAgent with a constrained tool surface (read only; no tool execution)
to translate statements like "coherence rewards beat entropy on GSM8K for Qwen 2.5"
into a runnable config: dataset, model, method (SFT/GRPO), reward composition,
threshold prediction.
"""
from __future__ import annotations

from pathlib import Path

import typer
import yaml

from carl_studio.console import get_console


HYPOTHESIZE_PROMPT = """You translate research hypotheses into runnable CARL training configs.

INPUT: a one-sentence hypothesis that names:
- Outcome claim ("X is better than Y")
- Measurement ("on <task>")
- Model family (optional)

OUTPUT: a YAML document with these keys:
  name: short slug
  hypothesis: the original statement
  prediction: a measurable threshold ("task_completion_rate > 0.75")
  model:
    base: hf model id (pick sensible default if not stated)
    method: sft | grpo | cascade
  dataset:
    id: hf dataset id
    split: train|test
  rewards:
    - name: coherence
      weight: 1.0
    # additional rewards if justified by the hypothesis
  eval:
    phase: 1 | 2 | 2prime
    threshold: float between 0.1 and 0.9

Respond ONLY with the YAML content inside a ```yaml code fence. No preface."""


def hypothesize_cmd(
    statement: str = typer.Argument(..., help="Hypothesis in plain English"),
    output: str = typer.Option("carl.yaml", "-o", "--output", help="Output config path"),
    model: str = typer.Option("", "--model", "-m", help="Claude model for translation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print config but don't write"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing output"),
) -> None:
    """Translate a hypothesis into a runnable carl.yaml experiment."""
    c = get_console()
    stmt = (statement or "").strip()
    if not stmt:
        c.error_with_hint(
            "Hypothesis is empty",
            detail="Pass a one-sentence hypothesis as the first argument.",
            hint='Example: carl hypothesize "coherence rewards beat entropy on GSM8K"',
            code="carl.hypothesize.empty",
        )
        raise typer.Exit(2)

    out_path = Path(output)
    if out_path.exists() and not force and not dry_run:
        c.error_with_hint(
            f"{out_path} already exists",
            detail="Refusing to overwrite without --force.",
            hint="Pass --force to overwrite, or -o <path> to write elsewhere.",
            code="carl.hypothesize.exists",
        )
        raise typer.Exit(2)

    from carl_studio.chat_agent import CARLAgent

    try:
        agent = CARLAgent(api_key="", model=model, frame=None, max_budget_usd=0.50)
    except ImportError as exc:
        c.error_with_hint(
            "CARLAgent unavailable",
            detail=str(exc),
            hint="Install chat extras: pip install 'carl-studio[observe]'",
            code="carl.hypothesize.no_agent",
        )
        raise typer.Exit(1) from exc
    # Disable all tools for this invocation -- it's a translation task only
    agent._tools = []  # type: ignore[attr-defined]

    c.blank()
    c.header("Hypothesize")
    c.kv("Hypothesis", stmt, key_width=12)
    c.info("Translating to carl.yaml ...")

    prompt = f"{HYPOTHESIZE_PROMPT}\n\nHYPOTHESIS: {stmt}"
    collected: list[str] = []
    try:
        for event in agent.chat(prompt):
            kind = getattr(event, "kind", "") or ""
            if kind in ("text", "text_delta"):
                collected.append(getattr(event, "content", "") or "")
    except Exception as exc:
        c.error_with_hint(
            "Agent call failed",
            detail=str(exc),
            hint="Set ANTHROPIC_API_KEY or run `carl camp login`.",
            code="carl.hypothesize.agent_error",
        )
        raise typer.Exit(1) from exc

    blob = "".join(collected).strip()
    yaml_text = _extract_yaml(blob)
    try:
        parsed = yaml.safe_load(yaml_text)
        if not isinstance(parsed, dict):
            raise ValueError("expected mapping")
    except Exception as exc:
        c.error_with_hint(
            "Model did not return valid YAML",
            detail=f"Got:\n{blob[:400]}",
            hint="Try rephrasing the hypothesis more concretely, or pass --model claude-opus-4-7.",
            code="carl.hypothesize.bad_output",
        )
        raise typer.Exit(1) from exc

    c.blank()
    c.print(yaml_text)
    c.blank()

    if dry_run:
        c.info("--dry-run: not writing file")
        return

    out_path.write_text(yaml_text, encoding="utf-8")
    c.ok(f"Wrote {out_path}")
    c.info(f"Next: carl train --config {out_path}")


def _extract_yaml(blob: str) -> str:
    """Strip ```yaml ... ``` fences if present; accept raw YAML otherwise."""
    s = blob.strip()
    for fence in ("```yaml", "```yml", "```"):
        if s.startswith(fence):
            s = s[len(fence) :]
            break
    if s.endswith("```"):
        s = s[:-3]
    return s.strip() + "\n"


__all__ = ["hypothesize_cmd", "HYPOTHESIZE_PROMPT"]
