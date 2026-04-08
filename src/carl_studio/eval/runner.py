"""Eval runner -- load checkpoint, run dataset, compute metrics, gate PASS/FAIL.

Supports three phases:
  Phase 1:  tool-call format/selection/chain metrics
  Phase 2:  VLM click accuracy / coordinate precision
  Phase 2': environment task completion + CARL coherence
"""

from __future__ import annotations

import logging
import re
import statistics
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / Report / Gate
# ---------------------------------------------------------------------------

class EvalConfig(BaseModel):
    """Evaluation configuration."""

    checkpoint: str = Field(description="HF model ID or local path")
    dataset: str = Field(description="HF dataset ID or local path")
    dataset_split: str = Field(default="test", description="Dataset split to load")
    phase: str = Field(
        default="auto",
        description="Eval phase: '1' (tool-call), '2' (vision), '2prime' (env), 'auto'",
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Pass threshold for primary metric")
    max_samples: int | None = Field(default=None, ge=1, description="Cap number of eval samples")
    batch_size: int = Field(default=1, ge=1, description="Inference batch size")
    device: str = Field(default="auto", description="Device: 'auto', 'cpu', 'cuda', 'cuda:0', etc.")

    @field_validator("phase")
    @classmethod
    def validate_phase(cls, v: str) -> str:
        allowed = {"auto", "1", "2", "2prime"}
        if v not in allowed:
            raise ValueError(f"phase must be one of {allowed}, got {v!r}")
        return v


class EvalReport(BaseModel):
    """Structured evaluation results."""

    checkpoint: str
    phase: str
    n_samples: int
    metrics: dict[str, float] = Field(default_factory=dict)
    primary_metric: str
    primary_value: float
    threshold: float
    passed: bool
    coherence: dict[str, float] | None = None


class EvalGate:
    """Applies pass/fail threshold to an EvalReport."""

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold

    def check(self, report: EvalReport) -> bool:
        """Return True if the report's primary metric meets the threshold."""
        return report.primary_value >= self.threshold


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------

_PHASE_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)phase\s*2\s*prime|phase2prime|2prime|env.?grpo", "2prime"),
    (r"(?i)phase\s*2|phase2|vision|vlm|click", "2"),
    (r"(?i)phase\s*1|phase1|tool.?call|sft|grpo", "1"),
]


def _detect_phase(checkpoint: str) -> str:
    """Infer eval phase from checkpoint name. Falls back to '1'."""
    for pattern, phase in _PHASE_PATTERNS:
        if re.search(pattern, checkpoint):
            return phase
    return "1"


# ---------------------------------------------------------------------------
# Per-phase metric computation (no GPU required)
# ---------------------------------------------------------------------------

def _compute_phase1_metrics(
    completions: list[str],
    samples: list[dict[str, Any]],
) -> dict[str, float]:
    """Phase 1 metrics: format validity, tool selection accuracy, chain completion."""
    from carl_studio.training.rewards.base import extract_json
    from carl_studio.training.rewards.task import (
        tool_call_format_reward,
        tool_selection_reward,
        chain_completion_reward,
    )

    format_scores = tool_call_format_reward(completions)

    # Build expected_tools and chain_length from sample metadata
    expected_tools: list[list[str]] = []
    chain_lengths: list[int] = []
    for s in samples:
        et = s.get("expected_tools") or s.get("tools") or []
        if isinstance(et, str):
            et = [et]
        expected_tools.append(et)
        chain_lengths.append(int(s.get("chain_length", 1)))

    selection_scores = tool_selection_reward(
        completions, expected_tools=expected_tools,
    )
    chain_scores = chain_completion_reward(
        completions, expected_tools=expected_tools, chain_length=chain_lengths,
    )

    format_validity = statistics.mean(format_scores) if format_scores else 0.0
    selection_acc = statistics.mean(selection_scores) if selection_scores else 0.0
    chain_rate = statistics.mean(chain_scores) if chain_scores else 0.0

    return {
        "format_validity": round(format_validity, 4),
        "tool_selection_accuracy": round(selection_acc, 4),
        "chain_completion_rate": round(chain_rate, 4),
    }


def _compute_phase2_metrics(
    completions: list[str],
    samples: list[dict[str, Any]],
) -> dict[str, float]:
    """Phase 2 metrics: format compliance, click accuracy, precision, distance."""
    import math

    from carl_studio.training.rewards.vlm import (
        coordinate_format_reward,
        click_accuracy_reward,
        precision_reward,
        _parse_coordinates,
    )

    format_scores = coordinate_format_reward(completions)
    bboxes = [s.get("bbox") or s.get("bounding_box") for s in samples]

    # Filter out samples without bboxes for accuracy/precision
    valid_completions: list[str] = []
    valid_bboxes: list[list[float]] = []
    for c, b in zip(completions, bboxes):
        if b is not None and isinstance(b, (list, tuple)) and len(b) >= 4:
            valid_completions.append(c)
            valid_bboxes.append(b)

    if valid_completions:
        click_scores = click_accuracy_reward(valid_completions, bbox=valid_bboxes)
        prec_scores = precision_reward(valid_completions, bbox=valid_bboxes)

        # Mean distance in pixels
        distances: list[float] = []
        for c, box in zip(valid_completions, valid_bboxes):
            coords = _parse_coordinates(c)
            if coords is not None:
                cx = (float(box[0]) + float(box[2])) / 2
                cy = (float(box[1]) + float(box[3])) / 2
                distances.append(math.sqrt((coords[0] - cx) ** 2 + (coords[1] - cy) ** 2))

        mean_click = statistics.mean(click_scores)
        mean_prec = statistics.mean(prec_scores)
        mean_dist = statistics.mean(distances) if distances else float("nan")
    else:
        mean_click = 0.0
        mean_prec = 0.0
        mean_dist = float("nan")

    return {
        "format_compliance": round(statistics.mean(format_scores) if format_scores else 0.0, 4),
        "click_accuracy": round(mean_click, 4),
        "mean_precision": round(mean_prec, 4),
        "mean_distance_px": round(mean_dist, 2) if not math.isnan(mean_dist) else 0.0,
    }


def _compute_phase2prime_metrics(
    completions: list[str],
    samples: list[dict[str, Any]],
) -> dict[str, float]:
    """Phase 2' metrics: task completion rate and mean CARL score.

    Task completion is determined by the 'completed' or 'success' field in sample
    metadata. CARL score comes from 'carl_score' if present, else 0.0.
    """
    completed = 0
    carl_scores: list[float] = []

    for i, s in enumerate(samples):
        # Task completion: check metadata or heuristic (non-empty completion with tool call)
        if s.get("completed") is not None or s.get("success") is not None:
            if s.get("completed", s.get("success", False)):
                completed += 1
        else:
            # Heuristic: completion contains at least one tool call structure
            from carl_studio.training.rewards.base import extract_json
            if extract_json(completions[i]) is not None:
                completed += 1

        carl_scores.append(float(s.get("carl_score", 0.0)))

    n = len(samples) or 1
    return {
        "task_completion_rate": round(completed / n, 4),
        "mean_carl_score": round(statistics.mean(carl_scores) if carl_scores else 0.0, 4),
    }


# Primary metric per phase
_PRIMARY_METRIC: dict[str, str] = {
    "1": "chain_completion_rate",
    "2": "click_accuracy",
    "2prime": "task_completion_rate",
}


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------

class EvalRunner:
    """Runs evaluation for a checkpoint against a dataset.

    Usage::

        config = EvalConfig(
            checkpoint="wheattoast11/OmniCoder-9B-Zero-Phase2",
            dataset="wheattoast11/zero-rl-tool-calling-data",
            phase="auto",
            threshold=0.5,
        )
        runner = EvalRunner(config)
        report = runner.run()
        print(report.passed)
    """

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.phase = config.phase if config.phase != "auto" else _detect_phase(config.checkpoint)

    def run(self) -> EvalReport:
        """Load model, run dataset, compute metrics, return report."""
        samples = self._load_dataset()
        model, tokenizer = self._load_model()
        completions = self._generate(model, tokenizer, samples)
        metrics = self._compute_metrics(completions, samples)
        coherence = self._compute_coherence(model, tokenizer, completions)

        primary_metric = _PRIMARY_METRIC.get(self.phase, "chain_completion_rate")
        primary_value = metrics.get(primary_metric, 0.0)

        return EvalReport(
            checkpoint=self.config.checkpoint,
            phase=self.phase,
            n_samples=len(samples),
            metrics=metrics,
            primary_metric=primary_metric,
            primary_value=primary_value,
            threshold=self.config.threshold,
            passed=primary_value >= self.config.threshold,
            coherence=coherence,
        )

    # -- Internal methods --

    def _load_dataset(self) -> list[dict[str, Any]]:
        """Load eval dataset from HF hub or local path."""
        import json
        import os

        path = self.config.dataset

        # Local JSONL file
        if os.path.isfile(path) and path.endswith((".jsonl", ".json")):
            samples: list[dict[str, Any]] = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
            if self.config.max_samples is not None:
                samples = samples[: self.config.max_samples]
            return samples

        # HF datasets
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required to load HF datasets. "
                "Install it with: pip install datasets"
            ) from exc

        ds = load_dataset(path, split=self.config.dataset_split)
        samples = [dict(row) for row in ds]
        if self.config.max_samples is not None:
            samples = samples[: self.config.max_samples]
        return samples

    def _load_model(self) -> tuple[Any, Any]:
        """Load model and tokenizer with lazy torch/transformers imports."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for model evaluation. Install it with: pip install torch"
            ) from exc

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for model evaluation. "
                "Install it with: pip install transformers"
            ) from exc

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # VLM phases use image-text model class
        if self.phase == "2":
            try:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(
                    self.config.checkpoint,
                    torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
                    device_map=device if device.startswith("cuda") else None,
                )
            except Exception:
                logger.warning(
                    "AutoModelForImageTextToText failed, falling back to AutoModelForCausalLM"
                )
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.checkpoint,
                    torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
                    device_map=device if device.startswith("cuda") else None,
                )
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                self.config.checkpoint,
                torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
                device_map=device if device.startswith("cuda") else None,
            )

        if not device.startswith("cuda") and model.device.type != device:
            model = model.to(device)

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _generate(
        self,
        model: Any,
        tokenizer: Any,
        samples: list[dict[str, Any]],
    ) -> list[str]:
        """Generate completions for each sample."""
        import torch

        completions: list[str] = []

        for i in range(0, len(samples), self.config.batch_size):
            batch = samples[i : i + self.config.batch_size]
            for sample in batch:
                prompt = sample.get("prompt") or sample.get("query") or sample.get("text", "")
                if isinstance(prompt, list):
                    # Conversational format -- use chat template if available
                    if hasattr(tokenizer, "apply_chat_template"):
                        text = tokenizer.apply_chat_template(
                            prompt, tokenize=False, add_generation_prompt=True,
                        )
                    else:
                        text = "\n".join(
                            m.get("content", "") for m in prompt if isinstance(m, dict)
                        )
                else:
                    text = str(prompt)

                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=2048,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                # Decode only new tokens
                new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
                completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
                completions.append(completion)

            if (i + self.config.batch_size) % 50 == 0:
                logger.info("Generated %d / %d", min(i + self.config.batch_size, len(samples)), len(samples))

        return completions

    def _compute_metrics(
        self,
        completions: list[str],
        samples: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Dispatch to phase-specific metric computation."""
        if self.phase == "1":
            return _compute_phase1_metrics(completions, samples)
        elif self.phase == "2":
            return _compute_phase2_metrics(completions, samples)
        elif self.phase == "2prime":
            return _compute_phase2prime_metrics(completions, samples)
        else:
            return _compute_phase1_metrics(completions, samples)

    def _compute_coherence(
        self,
        model: Any,
        tokenizer: Any,
        completions: list[str],
    ) -> dict[str, float] | None:
        """Optionally compute CARL coherence metrics. Returns None if no GPU."""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
        except ImportError:
            return None

        try:
            from carl_studio.primitives import CoherenceProbe

            vocab_size = model.config.vocab_size
            probe = CoherenceProbe(vocab_size=vocab_size)

            phi_values: list[float] = []
            cloud_values: list[float] = []
            disc_values: list[float] = []

            # Sample up to 20 completions for coherence (forward pass is expensive)
            subset = completions[:20]
            for text in subset:
                if not text.strip() or len(text) < 10:
                    continue
                try:
                    inputs = tokenizer(
                        text, return_tensors="pt", truncation=True, max_length=512,
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        outputs = model(**inputs)
                    if was_training:
                        model.train()

                    logits = outputs.logits[0]
                    token_ids = inputs["input_ids"][0]

                    snapshot = probe.measure(logits, token_ids)
                    phi_values.append(snapshot.phi_mean)
                    cloud_values.append(snapshot.cloud_quality_mean)
                    disc_values.append(snapshot.discontinuity_score)

                    del outputs, logits, inputs
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning("Coherence probe failed on sample: %s", e)
                    continue

            if not phi_values:
                return None

            return {
                "phi_mean": round(statistics.mean(phi_values), 4),
                "cloud_quality_mean": round(statistics.mean(cloud_values), 4),
                "discontinuity_score": round(statistics.mean(disc_values), 4),
            }
        except Exception as e:
            logger.warning("Coherence computation skipped: %s", e)
            return None
