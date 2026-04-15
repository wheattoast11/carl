"""Skills CLI sub-app. Register in cli.py via: app.add_typer(skills_app, name='skill')"""
from __future__ import annotations

import json
from typing import Any

import typer

from carl_studio.console import get_console
from carl_studio.skills.builtins import BUILTIN_SKILLS
from carl_studio.skills.runner import SkillRunner

skills_app = typer.Typer(
    name="skill",
    help="CARL skills and merit badges",
    no_args_is_help=True,
)


def _get_runner() -> SkillRunner:
    r = SkillRunner()
    for s in BUILTIN_SKILLS:
        r.register(s)
    return r


@skills_app.command(name="list")
def skill_list() -> None:
    """List all available skills and which badges are earned."""
    console = get_console()
    runner = _get_runner()
    earned = set(runner.badges_earned())
    skills = runner.list_skills()

    table = console.make_table("Skill", "Badge", "Tier", "Earned", "Description", title="CARL Skills")
    for skill in sorted(skills, key=lambda s: s.name):
        icon = console.theme.icons.badge if skill.name in earned else console.theme.icons.progress
        earned_label = f"{icon} YES" if skill.name in earned else "no"
        table.add_row(
            skill.name,
            skill.badge,
            skill.requires_tier,
            earned_label,
            skill.description[:60] + ("..." if len(skill.description) > 60 else ""),
        )
    console.print(table)


@skills_app.command(name="run")
def skill_run(
    name: str = typer.Argument(..., help="Skill name (e.g. observer, grader, trainer)"),
    inputs_json: str = typer.Option("{}", "--inputs", "-i", help="Inputs as JSON object"),
) -> None:
    """Run a skill by name. Pass inputs as JSON."""
    console = get_console()

    try:
        inputs: dict[str, Any] = json.loads(inputs_json)
    except json.JSONDecodeError as exc:
        console.error(f"Invalid JSON for --inputs: {exc}")
        raise typer.Exit(1) from exc

    if not isinstance(inputs, dict):
        console.error("--inputs must be a JSON object (dict), not a list or scalar")
        raise typer.Exit(1)

    runner = _get_runner()
    skill = runner.get(name)
    if skill is None:
        console.error(f"Skill '{name}' not found. Run 'carl skill list' to see available skills.")
        raise typer.Exit(1)

    console.info(f"Running skill: {name}")
    result = runner.run(name, **inputs)

    if result.success:
        console.ok(result.message)
    else:
        console.warn(result.message)

    if result.metrics:
        for k, v in result.metrics.items():
            console.kv(k, v)

    if not result.success:
        raise typer.Exit(1)


@skills_app.command(name="badges")
def skill_badges() -> None:
    """Show all earned merit badges."""
    console = get_console()
    runner = _get_runner()
    earned = runner.badges_earned()

    if not earned:
        console.info("No badges earned yet. Run 'carl skill run <name>' to start.")
        return

    console.print()
    console.print(f"  [camp.header]Merit Badges Earned ({len(earned)})[/]")
    console.blank()
    for skill_name in earned:
        skill = runner.get(skill_name)
        badge_name = skill.badge if skill else skill_name
        console.badge_award(badge_name)
    console.blank()


@skills_app.command(name="history")
def skill_history(
    skill: str = typer.Option("", "--skill", "-s", help="Filter by skill name"),
    limit: int = typer.Option(20, help="Max rows to show"),
) -> None:
    """Show skill run history."""
    console = get_console()
    runner = _get_runner()
    rows = runner.get_history(skill_name=skill or None, limit=limit)

    if not rows:
        msg = f"No history for skill '{skill}'." if skill else "No skill history yet."
        console.info(msg)
        return

    table = console.make_table(
        "ID", "Skill", "Success", "Badge", "Message", "Started",
        title="Skill Run History",
    )
    for row in rows:
        success_icon = console.theme.icons.ok if row["success"] else console.theme.icons.fail
        badge_icon = console.theme.icons.badge if row["badge_earned"] else ""
        started = str(row.get("started_at", ""))[:19]
        msg = str(row.get("message", ""))
        msg_short = msg[:50] + ("..." if len(msg) > 50 else "")
        table.add_row(
            str(row["id"]),
            str(row["skill_name"]),
            success_icon,
            badge_icon,
            msg_short,
            started,
        )
    console.print(table)
