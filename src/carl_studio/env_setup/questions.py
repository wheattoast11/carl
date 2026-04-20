"""Question registry for ``carl env``.

Each question is a pure function of current state that returns either:
- ``None`` — the question is already satisfied (skip)
- a ``Question`` — prompt + handler that mutates state on answer

Composition is associative by design: applying Q_m then Q_n produces
the same state as Q_n then Q_m when the fields they touch are disjoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from carl_studio.env_setup.state import EnvState


@dataclass(frozen=True)
class Choice:
    """One selectable option in a question."""

    key: str
    value: str
    label: str


@dataclass
class Question:
    """A single prompt + answer handler."""

    id: str
    prompt: str
    choices: list[Choice]
    explainer: str = ""
    free_form: bool = False  # accept arbitrary text rather than choice indices

    def handle(self, state: EnvState, raw_answer: str) -> EnvState:
        """Apply the answer to state and return the new (mutated) state."""
        return _HANDLERS[self.id](state, raw_answer, self.choices)


# ---------------------------------------------------------------------------
# Q1 — mode
# ---------------------------------------------------------------------------


def _q_mode(state: EnvState) -> Question | None:
    if state.mode:
        return None
    return Question(
        id="mode",
        prompt="What are you optimizing for?",
        explainer=(
            "Training fine-tunes a model on data. Inference serves a model "
            "for use. Both runs training and then serves the result."
        ),
        choices=[
            Choice("1", "train", "Pure training (SFT / GRPO / DPO)"),
            Choice("2", "infer", "Inference only"),
            Choice("3", "both", "Both training + inference"),
        ],
    )


def _handle_mode(state: EnvState, raw: str, choices: list[Choice]) -> EnvState:
    value = _resolve_choice(raw, choices)
    return state.model_copy(update={"mode": value})


# ---------------------------------------------------------------------------
# Q2 — method (only when training)
# ---------------------------------------------------------------------------


def _q_method(state: EnvState) -> Question | None:
    if state.method or state.mode == "infer":
        return None
    return Question(
        id="method",
        prompt="Training objective?",
        explainer=(
            "SFT learns from labeled examples. GRPO optimizes a policy "
            "against rewards. Cascade runs SFT first, then GRPO on top."
        ),
        choices=[
            Choice("1", "sft", "Supervised fine-tuning (SFT)"),
            Choice("2", "grpo", "GRPO reinforcement learning"),
            Choice("3", "dpo", "DPO preference optimization"),
            Choice("4", "cascade", "Multi-stage cascade (SFT → GRPO)"),
        ],
    )


def _handle_method(state: EnvState, raw: str, choices: list[Choice]) -> EnvState:
    value = _resolve_choice(raw, choices)
    return state.model_copy(update={"method": value})


# ---------------------------------------------------------------------------
# Q3 — dataset (only when training)
# ---------------------------------------------------------------------------


def _q_dataset(state: EnvState) -> Question | None:
    if state.dataset or state.mode == "infer":
        return None
    return Question(
        id="dataset",
        prompt="Dataset repo id or JSONL path:",
        explainer=(
            "Enter a HuggingFace dataset id (e.g. 'trl-lib/tool-use-suite') "
            "or a local path to a JSONL file. Prefix with ./ for local files."
        ),
        choices=[],
        free_form=True,
    )


def _handle_dataset(state: EnvState, raw: str, choices: list[Choice]) -> EnvState:
    raw = raw.strip()
    kind = "local" if raw.startswith(("./", "/", "~")) or raw.endswith(".jsonl") else "hf"
    return state.model_copy(update={"dataset": raw, "dataset_kind": kind})


# ---------------------------------------------------------------------------
# Q4 — compute (only when training)
# ---------------------------------------------------------------------------


def _q_compute(state: EnvState) -> Question | None:
    if state.compute or state.mode == "infer":
        return None
    return Question(
        id="compute",
        prompt="Compute target?",
        explainer=(
            "Local runs on one GPU on your machine. Cloud presets dispatch "
            "via RunPod / Modal / Lambda. Choose based on data size + budget."
        ),
        choices=[
            Choice("1", "local", "Local single GPU (A100/H100/L40S)"),
            Choice("2", "runpod-a100x4", "RunPod A100 x4 (fastest)"),
            Choice("3", "runpod-l40sx4", "RunPod L40S x4 (cost-optimized)"),
            Choice("4", "modal-a100x2", "Modal A100 x2 (middle ground)"),
        ],
    )


def _handle_compute(state: EnvState, raw: str, choices: list[Choice]) -> EnvState:
    value = _resolve_choice(raw, choices)
    return state.model_copy(update={"compute": value})


# ---------------------------------------------------------------------------
# Q5 — reward shape (only when method is grpo or cascade)
# ---------------------------------------------------------------------------


def _q_reward(state: EnvState) -> Question | None:
    if state.reward or state.mode == "infer":
        return None
    if state.method not in ("grpo", "cascade"):
        return None
    return Question(
        id="reward",
        prompt="GRPO reward shape?",
        explainer=(
            "Static is the canonical CARL composite (coherence + format + "
            "discontinuity). Phase-adaptive shifts weights by Kuramoto R. "
            "Custom leaves it for you to plug in later."
        ),
        choices=[
            Choice("1", "static", "Static CARL composite (50/30/20)"),
            Choice("2", "phase_adaptive", "Phase-adaptive (shift weights by R)"),
            Choice("3", "custom", "Custom reward fn (I'll plug it in later)"),
            Choice("4", "none", "No reward shaping (pure supervised loss)"),
        ],
    )


def _handle_reward(state: EnvState, raw: str, choices: list[Choice]) -> EnvState:
    value = _resolve_choice(raw, choices)
    return state.model_copy(update={"reward": value})


# ---------------------------------------------------------------------------
# Q6 — cascade stages (only when method is cascade)
# ---------------------------------------------------------------------------


def _q_cascade(state: EnvState) -> Question | None:
    if state.cascade_stages is not None or state.method != "cascade":
        return None
    return Question(
        id="cascade_stages",
        prompt="How many cascade stages?",
        explainer=(
            "2 = SFT → GRPO. 3 = SFT → DPO → GRPO. More stages = deeper "
            "curriculum but longer runtime."
        ),
        choices=[
            Choice("1", "2", "2 stages: SFT → GRPO"),
            Choice("2", "3", "3 stages: SFT → DPO → GRPO"),
        ],
    )


def _handle_cascade(state: EnvState, raw: str, choices: list[Choice]) -> EnvState:
    value_str = _resolve_choice(raw, choices)
    return state.model_copy(update={"cascade_stages": int(value_str)})


# ---------------------------------------------------------------------------
# Q7 — eval gate
# ---------------------------------------------------------------------------


def _q_eval(state: EnvState) -> Question | None:
    if state.eval_gate or state.mode == "infer":
        return None
    return Question(
        id="eval_gate",
        prompt="Eval admission policy?",
        explainer=(
            "none = no eval gate (run anyway). metric = gate on a numeric "
            "threshold (loss, accuracy). crystallization = gate on BITC "
            "crystallization events (v0.11+ primitive; empirical)."
        ),
        choices=[
            Choice("1", "none", "None — always admit"),
            Choice("2", "metric", "Metric threshold (loss/accuracy)"),
            Choice("3", "crystallization", "BITC crystallization events"),
        ],
    )


def _handle_eval(state: EnvState, raw: str, choices: list[Choice]) -> EnvState:
    value = _resolve_choice(raw, choices)
    return state.model_copy(update={"eval_gate": value})


# ---------------------------------------------------------------------------
# Resolution + registry
# ---------------------------------------------------------------------------


def _resolve_choice(raw: str, choices: list[Choice]) -> str:
    """Accept either a choice index (1/2/...) or the value directly."""
    raw = raw.strip()
    for c in choices:
        if raw == c.key or raw.lower() == c.value.lower():
            return c.value
    # Fallback: accept the first-letter shortcut
    for c in choices:
        if raw and c.value.startswith(raw.lower()):
            return c.value
    raise ValueError(f"unrecognized answer {raw!r}; choices={[c.value for c in choices]}")


_HANDLERS: dict[str, Callable[[EnvState, str, list[Choice]], EnvState]] = {
    "mode": _handle_mode,
    "method": _handle_method,
    "dataset": _handle_dataset,
    "compute": _handle_compute,
    "reward": _handle_reward,
    "cascade_stages": _handle_cascade,
    "eval_gate": _handle_eval,
}


# Public: ordered list of question factories. Composition respects order;
# later questions may inspect earlier answers. Each factory returns None
# when the question is auto-satisfied (based on mode, prior answers, etc.).
QUESTION_REGISTRY: list[Callable[[EnvState], Question | None]] = [
    _q_mode,
    _q_method,
    _q_dataset,
    _q_compute,
    _q_reward,
    _q_cascade,
    _q_eval,
]


def next_question(state: EnvState) -> Question | None:
    """Return the next unanswered question, or ``None`` when complete."""
    for factory in QUESTION_REGISTRY:
        q = factory(state)
        if q is not None:
            return q
    return None


__all__ = ["Question", "Choice", "QUESTION_REGISTRY", "next_question"]
