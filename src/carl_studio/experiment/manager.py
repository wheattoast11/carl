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
import os
import re
import uuid
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
        (exp_dir / "hypothesis.json").write_text(hypothesis.model_dump_json(indent=2))

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
                results.append(
                    {
                        "id": exp.id,
                        "title": exp.hypothesis.title,
                        "status": exp.status.value,
                        "verdict": exp.judgment.verdict.value if exp.judgment else "-",
                    }
                )
        return results

    def save_preparation(self, preparation: Any) -> None:
        """Retain prepared inputs as an artifact of the existing experiment."""
        exp_dir = self.base_dir / preparation.plan_id
        exp_dir.mkdir(mode=0o700, exist_ok=True)
        os.chmod(exp_dir, 0o700)
        payload = preparation.model_dump_json(indent=2)
        target = exp_dir / "preparation.json"
        if target.exists():
            existing = self.load_preparation(preparation.plan_id)
            if existing.config != preparation.config or existing.sources != preparation.sources:
                raise ValueError("Prepared experiment identity conflict")
            if not (exp_dir / "experiment.json").is_file():
                self._save(
                    Experiment(
                        id=existing.plan_id,
                        hypothesis=existing.hypothesis,
                        config=existing.config,
                        tags=["prepared-training"],
                    )
                )
            return
        temporary = exp_dir / ("preparation." + uuid.uuid4().hex + ".tmp")
        with temporary.open("x") as stream:
            os.chmod(temporary, 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        exp = Experiment(
            id=preparation.plan_id,
            hypothesis=preparation.hypothesis,
            config=preparation.config,
            tags=["prepared-training"],
        )
        self._save(exp)
        self._update_index()

    def load_preparation(self, plan_id: str) -> Any:
        """Read the typed preparation artifact without loading training code."""
        from carl_studio.types.preparation import TrainingPreparation

        if not re.fullmatch(r"Eprep_[0-9a-f]{24}", plan_id):
            raise ValueError("Invalid prepared experiment ID")
        return TrainingPreparation.model_validate_json(
            (self.base_dir / plan_id / "preparation.json").read_text()
        )

    def claim_training(self, plan_id: str, *, resume: bool = False) -> None:
        """Claim a prepared experiment once before starting its effects."""
        self.load_preparation(plan_id)
        exp_dir = self.base_dir / plan_id
        admission = exp_dir / "execution.admission"
        descriptor = os.open(admission, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.close(descriptor)
            exp = self.load(plan_id)
            if exp.config.get("execution_active"):
                raise ValueError("Execution custody requires reconciliation before resume")
            claim = exp_dir / "execution.claim"
            if resume and claim.exists():
                run = self.load_training_result(plan_id)
                if run is None or run.phase.value not in {"paused", "failed"}:
                    raise ValueError("Execution custody requires reconciliation before resume")
                claim.rename(claim.with_name("execution." + uuid.uuid4().hex + ".claim"))
            with claim.open("x") as stream:
                os.chmod(claim, 0o600)
                stream.write(json.dumps({"pid": os.getpid(), "plan_id": plan_id}))
            exp.config["execution_active"] = True
            self._save(exp)
        finally:
            admission.unlink()

    def save_training_result(self, plan_id: str, run: Any) -> None:
        """Retain the training result at its original experiment owner."""
        exp = self.load(plan_id)
        exp.run_id = run.id
        exp.artifacts = list(run.artifacts)
        exp.config["training_result"] = run.model_dump(mode="json")
        exp.config["execution_active"] = False
        self._save(exp)

    def load_training_result(self, plan_id: str) -> Any:
        """Return an existing result instead of repeating a training effect."""
        from carl_studio.types.run import TrainingRun

        exp = self.load(plan_id)
        data = exp.config.get("training_result")
        return TrainingRun.model_validate(data) if data is not None else None

    def _save(self, exp: Experiment) -> None:
        exp_dir = self.base_dir / exp.id
        exp_dir.mkdir(exist_ok=True)
        temporary = exp_dir / ("experiment." + uuid.uuid4().hex + ".tmp")
        with temporary.open("x") as stream:
            os.chmod(temporary, 0o600)
            stream.write(exp.model_dump_json(indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(exp_dir / "experiment.json")

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
            lines.extend(
                [
                    "",
                    "## Judgment",
                    "",
                    f"**Verdict:** {exp.judgment.verdict.value}",
                    f"**Confidence:** {exp.judgment.confidence:.2f}",
                    f"**Notes:** {exp.judgment.notes}",
                ]
            )

        exp_dir = self.base_dir / exp.id
        (exp_dir / "README.md").write_text("\n".join(lines))

    def _update_index(self) -> None:
        """Update the L0 index file (always-loaded experiment summary)."""
        experiments = self.list_experiments()
        lines = ["# Experiments", ""]
        for e in experiments:
            lines.append(f"- [{e['id']}]({e['id']}/) — {e['title']} [{e['status']}] {e['verdict']}")
        self._index_path.write_text("\n".join(lines))
