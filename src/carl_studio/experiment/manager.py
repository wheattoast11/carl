"""Experiment manager — filesystem-backed experiment lifecycle.

Manages the experiment directory structure:
  experiments/
    E001_floor_ceiling_validation/
      hypothesis.json        # Pre-registered hypothesis
      config.json            # Training/eval config
      artifacts/             # Checkpoints, logs, plots
      witnesses.json         # Collected witnesses
      judgment.json           # Final verdict
      README.md              # Human-readable summary (auto-generated)

Memory tiering (mempalace-inspired):
  L0 — EXPERIMENTS.md index (~100 tokens, always loaded)
  L1 — hypothesis.json per experiment (~200 tokens each, loaded on demand)
  L2 — config + witnesses (loaded during analysis)
  L3 — raw artifacts (loaded only for deep inspection)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carl_studio.experiment.types import (
    Artifact,
    Experiment,
    ExperimentStatus,
    Hypothesis,
    Judgment,
    Witness,
)


class ExperimentManager:
    """Filesystem-backed experiment lifecycle manager.

    Usage:
        mgr = ExperimentManager("experiments/")
        exp = mgr.create(hypothesis)
        mgr.configure(exp.id, config_dict)
        mgr.start(exp.id, run_id="69d7...")
        mgr.add_artifact(exp.id, artifact)
        mgr.add_witness(exp.id, witness)
        judgment = mgr.judge(exp.id)
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.base_dir / "EXPERIMENTS.md"

    def create(self, hypothesis: Hypothesis, tags: list[str] | None = None) -> Experiment:
        """Create a new pre-registered experiment from a hypothesis."""
        n = len(list(self.base_dir.glob("E*"))) + 1
        exp_id = f"E{n:03d}_{hypothesis.id.replace('H', '').replace(' ', '_')}"

        exp = Experiment(
            id=exp_id,
            hypothesis=hypothesis,
            status=ExperimentStatus.PRE_REGISTERED,
            tags=tags or [],
        )

        exp_dir = self.base_dir / exp_id
        exp_dir.mkdir(exist_ok=True)
        (exp_dir / "artifacts").mkdir(exist_ok=True)

        # Save hypothesis
        (exp_dir / "hypothesis.json").write_text(
            hypothesis.model_dump_json(indent=2)
        )

        # Save experiment state
        self._save(exp)
        self._update_index()

        return exp

    def configure(self, exp_id: str, config: dict[str, Any]) -> Experiment:
        """Attach a training/eval config to the experiment."""
        exp = self.load(exp_id)
        exp.config = config
        exp.status = ExperimentStatus.CONFIGURED

        exp_dir = self.base_dir / exp_id
        (exp_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

        self._save(exp)
        return exp

    def start(self, exp_id: str, run_id: str) -> Experiment:
        """Mark experiment as running with a job/run ID."""
        exp = self.load(exp_id)
        exp.run_id = run_id
        exp.status = ExperimentStatus.RUNNING
        self._save(exp)
        self._update_index()
        return exp

    def add_artifact(self, exp_id: str, artifact: Artifact) -> None:
        """Add an artifact to the experiment."""
        exp = self.load(exp_id)
        exp.artifacts.append(artifact)
        self._save(exp)

    def add_witness(self, exp_id: str, witness: Witness) -> None:
        """Add a witness observation to the experiment."""
        exp = self.load(exp_id)
        exp.witnesses.append(witness)
        exp.status = ExperimentStatus.WITNESSING

        exp_dir = self.base_dir / exp_id
        witnesses_data = [w.model_dump() for w in exp.witnesses]
        (exp_dir / "witnesses.json").write_text(json.dumps(witnesses_data, indent=2, default=str))

        self._save(exp)

    def judge(self, exp_id: str) -> Judgment:
        """Render judgment on the experiment."""
        exp = self.load(exp_id)
        judgment = exp.judge()

        exp_dir = self.base_dir / exp_id
        (exp_dir / "judgment.json").write_text(judgment.model_dump_json(indent=2))

        # Auto-generate README
        self._write_readme(exp)
        self._save(exp)
        self._update_index()

        return judgment

    def load(self, exp_id: str) -> Experiment:
        """Load an experiment from disk."""
        exp_dir = self.base_dir / exp_id
        state_path = exp_dir / "experiment.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Experiment {exp_id} not found at {state_path}")
        return Experiment.model_validate_json(state_path.read_text())

    def list_experiments(self) -> list[dict[str, str]]:
        """List all experiments with their status."""
        results = []
        for d in sorted(self.base_dir.glob("E*")):
            if not d.is_dir():
                continue
            state = d / "experiment.json"
            if state.exists():
                exp = Experiment.model_validate_json(state.read_text())
                results.append({
                    "id": exp.id,
                    "title": exp.hypothesis.title,
                    "status": exp.status.value,
                    "verdict": exp.judgment.verdict.value if exp.judgment else "-",
                })
        return results

    def _save(self, exp: Experiment) -> None:
        exp_dir = self.base_dir / exp.id
        exp_dir.mkdir(exist_ok=True)
        (exp_dir / "experiment.json").write_text(exp.model_dump_json(indent=2))

    def _write_readme(self, exp: Experiment) -> None:
        """Auto-generate a human-readable experiment summary."""
        lines = [
            f"# {exp.id}: {exp.hypothesis.title}",
            "",
            f"**Status:** {exp.status.value}",
            f"**Created:** {exp.created_at.isoformat()}",
            f"**Run ID:** {exp.run_id or 'not started'}",
            "",
            "## Hypothesis",
            "",
            f"**Observation:** {exp.hypothesis.observation}",
            "",
            f"**Statement:** {exp.hypothesis.statement}",
            "",
            "## Predictions",
            "",
        ]
        for p in exp.hypothesis.predictions:
            witnessed = any(w.prediction_id == p.id and w.supports for w in exp.witnesses)
            refuted = any(w.prediction_id == p.id and not w.supports for w in exp.witnesses)
            icon = "REALIZED" if witnessed else "REFUTED" if refuted else "pending"
            lines.append(f"- **{p.id}** [{icon}]: {p.claim}")

        if exp.judgment:
            lines.extend([
                "",
                "## Judgment",
                "",
                f"**Verdict:** {exp.judgment.verdict.value}",
                f"**Confidence:** {exp.judgment.confidence:.2f}",
                f"**Notes:** {exp.judgment.notes}",
            ])

        exp_dir = self.base_dir / exp.id
        (exp_dir / "README.md").write_text("\n".join(lines))

    def _update_index(self) -> None:
        """Update the L0 index file (always-loaded experiment summary)."""
        experiments = self.list_experiments()
        lines = ["# Experiments", ""]
        for e in experiments:
            lines.append(f"- [{e['id']}]({e['id']}/) — {e['title']} [{e['status']}] {e['verdict']}")
        self._index_path.write_text("\n".join(lines))
