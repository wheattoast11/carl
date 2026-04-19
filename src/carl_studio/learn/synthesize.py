"""Synthesize pipeline -- codebase to graded training samples.

Carl-as-agent: uses LLM to analyze codebase, generate samples
from templates, validate in sandbox, output JSONL.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from carl_studio.console import get_console
from carl_studio.data.templates import ALL_TEMPLATES, ProblemTemplate
from carl_studio.data.validator import validate_sample
from carl_studio.learn.ingest import SourceIngester
from carl_studio.llm import LLMProvider, parse_llm_json
from carl_studio.tools import ensure_tools

__all__ = ["SynthesizeConfig", "SynthesizePipeline", "SynthesizeResult"]

SYNTH_SYSTEM = """You are generating a coding problem for ML training. You will receive:
1. A codebase to draw inspiration from
2. A problem template describing the type of problem to generate

Generate a self-contained project with a graded test harness that prints SCORE:X/Y.

Output strict JSON only:
{
  "task_description": "2-4 sentence problem description",
  "initial_files": {"filename.py": "content", "test_solution.py": "graded harness printing SCORE:X/Y"},
  "fix_target": "which file the fix replaces",
  "intended_fix": "correct fix (full file replacement)",
  "trivial_bypass": "lazy fix that passes T1 but fails T2+"
}

REQUIREMENTS:
- test_solution.py prints exactly one SCORE:X/Y line
- Buggy code scores low, fix scores high, bypass scores low
- Standard library only in test_solution.py
- Output ONLY JSON"""


class SynthesizeConfig(BaseModel):
    source: str
    count: int = 10
    output: str = ""

    @property
    def output_path(self) -> Path:
        if self.output:
            return Path(self.output)
        return Path(self.source) / "carl_synthesized.jsonl"


class SynthesizeResult(BaseModel):
    total_generated: int
    total_valid: int
    output_path: str
    coverage: dict[str, int]


class SynthesizePipeline:
    def __init__(self, config: SynthesizeConfig, console=None):
        self.config = config
        self.console = console or get_console()
        self.llm = LLMProvider.auto()

    def run(self) -> SynthesizeResult:
        c = self.console

        # Check tools
        ensure_tools(c)

        # Phase 1: Ingest
        c.kv("Source", self.config.source)
        c.kv("Provider", self.llm.provider_name)
        c.kv("Model", self.llm._model)

        ingester = SourceIngester()
        chunks = ingester.ingest(self.config.source)
        c.kv("Files ingested", len(chunks))

        # Build codebase context
        file_contents: dict[str, str] = {}
        for chunk in chunks:
            src = chunk.source
            if src not in file_contents:
                file_contents[src] = chunk.text
            else:
                file_contents[src] += "\n" + chunk.text

        codebase_ctx = "\n\n".join(
            f"### {path}\n```\n{content[:3000]}\n```"
            for path, content in list(file_contents.items())[:20]
        )

        # Phase 2: Generate + Validate
        c.rule("Generating")
        samples: list[dict] = []
        attempts = 0
        coverage: dict[str, int] = {}
        per_template = max(1, self.config.count // len(ALL_TEMPLATES))
        output_path = self.config.output_path

        for template in ALL_TEMPLATES:
            for seed in range(per_template * 3):  # up to 3x attempts per slot
                if sum(1 for s in samples if s["metadata"]["template_id"] == template.id) >= per_template:
                    break

                attempts += 1
                sample_id = f"synth-{template.id}-{seed:03d}"

                # Generate
                prompt = self._generation_prompt(template, codebase_ctx, seed)
                try:
                    response = self.llm.complete(
                        [{"role": "system", "content": SYNTH_SYSTEM},
                         {"role": "user", "content": prompt}],
                    )
                    generated = parse_llm_json(response)
                    if not generated:
                        c.warn(f"{sample_id}: JSON parse failed")
                        continue

                    # Validate
                    vr = validate_sample(generated)
                    if vr.valid:
                        c.ok(f"{sample_id} (red={vr.red_score:.2f} green={vr.green_score:.2f} ag={vr.antigame_score:.2f})")
                        sample = self._make_sample(generated, template, sample_id)
                        samples.append(sample)
                        for d in template.dims:
                            coverage[d] = coverage.get(d, 0) + 1
                        # Stream to disk
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(str(output_path) + ".partial", "a") as f:
                            f.write(json.dumps(sample) + "\n")
                    else:
                        # Two-pass fix
                        fix_response = self.llm.complete(
                            [{"role": "user", "content": self._fix_prompt(generated, vr)}],
                        )
                        fixed = parse_llm_json(fix_response)
                        if fixed:
                            vr2 = validate_sample(fixed)
                            if vr2.valid:
                                c.ok(f"{sample_id} fixed (red={vr2.red_score:.2f} green={vr2.green_score:.2f} ag={vr2.antigame_score:.2f})")
                                sample = self._make_sample(fixed, template, sample_id)
                                samples.append(sample)
                                for d in template.dims:
                                    coverage[d] = coverage.get(d, 0) + 1
                                with open(str(output_path) + ".partial", "a") as f:
                                    f.write(json.dumps(sample) + "\n")
                            else:
                                c.warn(f"{sample_id}: still invalid after fix")
                        else:
                            c.warn(f"{sample_id}: fix JSON parse failed")

                except Exception as e:
                    c.warn(f"{sample_id}: {e}")

                if len(samples) >= self.config.count:
                    break
            if len(samples) >= self.config.count:
                break

        # Phase 3: Write final output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        # Clean up partial
        partial = Path(str(output_path) + ".partial")
        if partial.exists():
            partial.unlink()

        # Register in ~/.carl/datasets/
        datasets_dir = Path.home() / ".carl" / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        index_path = datasets_dir / "index.yaml"
        index: dict = {}
        if index_path.exists():
            try:
                import yaml
                with open(index_path) as f:
                    index = yaml.safe_load(f) or {}
            except ImportError:
                pass
        index[str(output_path.resolve())] = {
            "samples": len(samples), "coverage": coverage, "source": self.config.source,
        }
        try:
            import yaml
            with open(index_path, "w") as f:
                yaml.dump(index, f)
        except ImportError:
            with open(datasets_dir / "index.json", "w") as f:
                json.dump(index, f, indent=2)

        # Summary
        c.rule("Complete")
        c.kv("Valid samples", len(samples))
        c.kv("Attempts", attempts)
        c.kv("Pass rate", f"{len(samples)/max(attempts,1):.0%}")
        c.kv("Output", str(output_path))
        for dim, n in sorted(coverage.items()):
            c.kv(f"  {dim}", n)
        c.info(f"Next: carl train --dataset {output_path}")

        return SynthesizeResult(
            total_generated=attempts,
            total_valid=len(samples),
            output_path=str(output_path),
            coverage=coverage,
        )

    def _generation_prompt(self, template: ProblemTemplate, codebase_ctx: str, seed: int) -> str:
        tiers = "\n".join(f"  T{i+1} ({t.name}): {t.description}" for i, t in enumerate(template.tiers))
        n = len(template.tiers)
        return f"""TEMPLATE: {template.name}
CAPABILITY: {', '.join(template.dims)}
BUG PATTERN: {template.bug_pattern}
VARIATION SEED: {seed}
VARIATION AXES: {', '.join(template.variation_axes)}

TEST TIERS:
{tiers}

CODEBASE CONTEXT (for inspiration):
{codebase_ctx}

Buggy code must score <= 1/{n}. Fix must score >= {n-1}/{n}. Bypass must score <= 1/{n}.
Verify mentally before outputting."""

    def _fix_prompt(self, sample: dict, vr) -> str:
        issues: list[str] = []
        if vr.red_score >= 0.5:
            issues.append(f"RED={vr.red_score:.2f}: test doesn't catch bug. Strengthen it.")
        if vr.green_score < 0.9:
            issues.append(f"GREEN={vr.green_score:.2f}: fix doesn't pass. Fix the intended_fix or test.")
        if vr.antigame_score >= 0.5:
            issues.append(f"ANTIGAME={vr.antigame_score:.2f}: bypass passes. Add edge cases.")
        return f"""FAILED VALIDATION. Fix the sample.

FILES: {json.dumps(sample.get('initial_files', {}), indent=2)}
FIX TARGET: {sample.get('fix_target', '')}
INTENDED FIX: {sample.get('intended_fix', '')}
TRIVIAL BYPASS: {sample.get('trivial_bypass', '')}

ISSUES: {'; '.join(issues)}

Output COMPLETE fixed JSON (same schema). Verify before outputting. Output ONLY JSON."""

    @staticmethod
    def _make_sample(generated: dict, template: ProblemTemplate, sample_id: str) -> dict:
        return {
            "id": sample_id,
            "messages": [
                {"role": "system", "content": "You are CARL, an AI assistant that solves problems through planning and action.\n\nAvailable tools: read_file, write_file, execute_code, run_shell\n\nBe direct. Plan concisely. Act decisively."},
                {"role": "user", "content": generated.get("task_description", "")},
            ],
            "tools": [
                {"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
                {"type": "function", "function": {"name": "write_file", "description": "Write a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
                {"type": "function", "function": {"name": "execute_code", "description": "Execute Python code.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
                {"type": "function", "function": {"name": "run_shell", "description": "Run shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
            ],
            "initial_files": generated.get("initial_files", {}),
            "metadata": {
                "capability_dims": template.dims,
                "template_id": template.id,
                "test_tiers": len(template.tiers),
            },
        }
