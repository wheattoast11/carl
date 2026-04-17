# Skills

Composable workflows that earn merit badges. Each skill is a self-contained CARL workflow that can be executed, recorded, and awarded as a camp badge.

## Key Abstractions

- `BaseSkill` -- Abstract base class. Subclasses define `name`, `badge`, `description`, `requires_tier`, and implement `execute(**kwargs) -> SkillResult`. Pure execution; side effects happen in the runner.
- `SkillResult` -- Output model: success, badge_earned, metrics, message, artifact.
- `SkillRun` -- Historical record of a skill execution (hydrated from DB).
- `SkillRunner` -- Resolves, executes, records results in SQLite (`skill_runs` table), and awards badges. Manages its own schema independently from LocalDB.
- `SkillRegistry` -- In-memory name-to-skill mapping.

## Built-in Skills (`builtins/`)

- `ObserverSkill` -- coherence observation workflow
- `GraderSkill` -- evaluation and grading
- `TrainerSkill` -- training run orchestration
- `SynthesizerSkill` -- model synthesis/merging
- `DeployerSkill` -- model deployment

## Architecture

Skills are discovered and registered by `SkillRunner`. The A2A layer dispatches tasks that resolve to skills. The MCP server exposes skills as tools via `BaseSkill.to_mcp_schema()`. The CLI surfaces skills through `carl skills`.

## Usage

```python
from carl_studio.skills import SkillRunner
from carl_studio.skills.builtins import ObserverSkill

runner = SkillRunner()
runner.register(ObserverSkill())
result = runner.run("observer", model="my-model")
```
