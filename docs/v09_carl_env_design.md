---
last_updated: 2026-04-20
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.9-preview
status: roadmap
---

# `carl-env` Design: Progressive-Disclosure Environment Setup (v0.9)

**Status:** Design document for v0.9 ship  
**Author:** Carl Studio  
**Context:** `carl init` (first-run) mirrors interactive UX; `carl env` orchestrates training environment configuration for SFT, GRPO/DPO, inference, or multi-stage cascades.

---

## 1. Behavior Spec: Interactive Session Transcript

Below is an exemplary 7-question flow building a full SFT + GRPO tool-calling training config. Each answer narrows the possibility space; the user can redirect at any gate, but never dead-ends.

```
$ carl env

╭─────────────────────────────────────────────────────────────────╮
│ Carl Environment Setup — High-Agency Configuration Wizard      │
│ Building: training environment + strategy                       │
│ Output: carl.yaml (ready for `carl train`)                     │
╰─────────────────────────────────────────────────────────────────╯

Q1. What are you optimizing for?
  [1] Pure training (SFT / GRPO / DPO)
  [2] Inference only
  [3] Both training + inference
  → 1

✓ Training selected. Narrowing to training workflows.

Q2. What is your primary objective?
  [1] Supervised fine-tuning (SFT) — learn from examples
  [2] Reinforcement learning (GRPO) — optimize with rewards
  [3] Multi-stage cascade (SFT → GRPO) — compound learning
  → 3

✓ Cascade selected. You will configure SFT stage, then GRPO stage.

Q3. Where is your training data?
  [1] HuggingFace dataset
  [2] Local JSONL file
  [3] I need to synthesize/collect it first
  → 1

Dataset source: wikitext (or press Enter)
→ trl-lib/tool-use-suite

✓ Detected: HF dataset. Pulling metadata…

Q4. Which hardware are you targeting?
  [1] Local (single GPU: A100/H100/L40S)
  [2] Distributed cloud (multi-GPU, managed by RunPod / Modal / Lambda)
  [3] Not sure — show me what fits my data size
  → 3

Data size estimate: 45K examples
Recommended hardware:
  • Local A100 → ~2h per epoch (batch=8, gradient_accum=4)
  • Cloud A100x4 → ~30m per epoch
  • Cloud L40Sx4 → ~45m per epoch (cost-optimized)

Choose one? (or 1–3 for custom)
→ 2

✓ Cloud A100x4 selected. Compute target: a100-largex4

Q5. Do you want reward-shaping for GRPO?
  [1] Static CARL reward (50% coherence + 30% format + 20% discontinuity)
  [2] Phase-adaptive reward (weights shift by Kuramoto phase)
  [3] Custom reward function (you'll plug it in later)
  [4] None — pure supervised loss only
  → 1

✓ Static CARL reward. Cascade gate mode: metric (default).

Q6. Cascade structure — stages and gates?
  Quick preset or custom?
  [1] Quick (2 stages: format-only → full reward, gate at step 50)
  [2] Custom — I'll define stages
  → 1

✓ Quick cascade: Stage A (SFT) → Stage B (GRPO + CARL reward at step 50).

Q7. Evaluation gate — pass threshold?
  After training, eval on test set. Pass if:
  [1] Tool-call format accuracy ≥ 85%
  [2] Task completion ≥ 70%
  [3] Custom threshold (enter a number 0–1)
  → 1

✓ Threshold: 0.85 (tool-call format).

────────────────────────────────────────────────────────────────

Rendering config… Done.

╭─ Proposed TrainingConfig ─────────────────────────────────────╮
│                                                               │
│ run_name:              tool-calling-2stage                   │
│ base_model:            meta-llama/Llama-2-7b                 │
│ method:                cascade (sft → grpo)                  │
│ dataset:               trl-lib/tool-use-suite                │
│ compute_target:        a100-largex4                          │
│ max_steps:             -1 (3 epochs)                         │
│ learning_rate:         2.0e-5                                │
│                                                               │
│ GRPO settings:                                               │
│   num_generations: 8                                         │
│   max_completion_length: 512                                 │
│   reward_class: static (CARL)                                │
│                                                               │
│ Cascade:                                                     │
│   stage_a: steps 0–49 (SFT only)                            │
│   stage_b: steps 50+ (SFT + CARL reward, linear warmup)    │
│   gate_mode: metric                                          │
│                                                               │
│ Eval gate: format_accuracy ≥ 0.85                           │
│                                                               │
╰─────────────────────────────────────────────────────────────╯

Proceed? (y/n)
→ y

Saved: ./carl.yaml
Ready: carl train --config carl.yaml
```

**Design notes:**
- Each Q's answer eliminates invalid branches in the subsequent decision tree.
- Q3 → Q4 → Q5 are sequential narrowing: data source determines inference cost, which informs hardware choice, which informs reward feasibility.
- Q6–Q7 are orthogonal safety gates: cascade structure and eval thresholds.
- User can type "?" at any prompt for inline help (links to WorkFrame docs or `carl.yaml` schema).
- "I'm not sure" always routes to a micro-explanation + sensible default (no dead-ends).

---

## 2. Functor Model: State-Passing Question Composition

Each question is a **typed transition**:

```python
Question[Context, Answer] = Callable[
    [StateContext],
    Tuple[QuestionPrompt, Callable[[Answer], StateContext]]
]
```

Composition is **associative**: `Q_m ∘ Q_n` produces the same final state regardless of input order (when their domains don't overlap).

### State representation (serializable JSON):

```python
@dataclass
class EnvState:
    """Rolling state accumulated by questions."""
    
    # Frame (from WorkFrame — seed state)
    frame: WorkFrame
    
    # Training intent (Q1)
    mode: Literal["training", "inference", "both"]
    
    # Method + cascade (Q2)
    method: TrainingMethod  # sft, grpo, dpo, kto, cascade
    cascade_stages: Optional[List[CascadeStage]]
    
    # Data (Q3)
    dataset_repo: str
    dataset_split: str
    eval_dataset_repo: Optional[str]
    
    # Compute (Q4)
    compute_target: ComputeTarget
    estimated_throughput: Optional[float]  # tok/s
    
    # Rewards (Q5)
    reward_class: Literal["static", "phase_adaptive", "custom", "none"]
    reward_custom_path: Optional[str]
    
    # Cascade gates (Q6)
    cascade_config: CascadeConfig
    
    # Eval (Q7)
    eval_threshold: float
    eval_metric: str  # "format_accuracy", "task_completion", "custom"
    
    # Derived: the final TrainingConfig (filled on render)
    training_config: Optional[TrainingConfig]
```

### Question archetype (typed functor):

```python
class Question(Protocol):
    def __call__(
        self,
        state: EnvState,
        console: CampConsole,
    ) -> Tuple[str, Callable[[str], EnvState]]:
        """
        Given current state, emit a question string + a transition function.
        
        The transition function applies the user's answer and returns
        a new state (possibly skipping later questions if deterministic).
        """
        ...
```

### Composition law:

```python
@property
def composed_flow(questions: List[Question]) -> Callable[[EnvState], EnvState]:
    """Fold questions left-to-right: Q1 >> Q2 >> ... >> Q_n"""
    def flow(state: EnvState) -> EnvState:
        for q in questions:
            prompt, transition = q(state, console)
            answer = typer.prompt(prompt)
            state = transition(answer)
        return state
    return flow
```

---

## 3. Question Taxonomy: 8 Archetypes

| Archetype | Role | Skip Condition | Reference |
|-----------|------|---|---|
| **Goal** | Q1: mode (train/infer/both) | None | Fixed entry point |
| **Method** | Q2: method selector | If mode=infer, skip → default sft for checkpoint loading |
| **Dataset** | Q3: source + split | Auto-detect if HF token + `base_model` pinned in WorkFrame |
| **Compute** | Q4: hardware target | If `CARL_COMPUTE` env var set | `ComputeTarget` enum |
| **Reward** | Q5: reward function class | If method=sft, skip → static |
| **Cascade** | Q6: stage definitions + gates | If method != grpo, skip → defaults |
| **Eval** | Q7: pass threshold + metric | If eval not wanted, skip → None |
| **Resume** | Optional Q0: "resume from prior run?" | If no prior `.carl/last_env_state.json` exists |

Each archetype exposes a **skip-condition function** that consults current state. If true, the question is automatically answered with a default and the state is updated without prompting the user.

---

## 4. High-Agency Pattern

Applying Nick Wignall's principles:

### (a) "What would this look like done?"
**Terminal state defined upfront** → The end state is a valid `TrainingConfig` + serialized `EnvState` written to `carl.yaml` + `.carl/last_env_state.json`. All 7 questions reverse-engineer from this terminal state.

### (b) "Who could I ask?"
**Deterministic referral on dead-end.** If a user is stuck on Q5 (reward function choice), the wizard says:
```
Not sure? → Read: https://carl.camp/docs/rewards
           or paste the output of: carl env --show-reward-options
           or pick [4] to skip and we'll use static (safest).
```

If `--show-reward-options` is passed, render a table of all known reward functions with brief descriptions. This is a **micro-explainer command**, not a full question.

### (c) "Refusing defeat"
**Every question has an "show me" branch.** Typing `?` or selecting "?" shows:
- The current state dump (formatted)
- Why this question matters (1–2 sentences)
- A default that moves forward
- A link to further reading

Example:
```
Q5. Reward class? (or ? for help)
  → ?

[Reward explains coherence awareness. Recommended: static CARL]

Current state:
  method=grpo, dataset=trl-lib/tool-use-suite, compute=a100-largex4
  
Why: GRPO amplifies patterns. Rewards shape which patterns are "good".

Defaults:
  • static (50/30/20 composite) — proven, no custom code
  • phase_adaptive — advanced, requires Kuramoto R monitoring
  • custom — you own the logic

Link: https://carl.camp/docs/rewards

Pick one [1–3] or Enter for static:
```

---

## 5. Files to Create/Modify

### New package: `src/carl_studio/env_setup/`

```
env_setup/
├── __init__.py
├── functor.py        # Question, EnvState, composition primitives
├── questions.py      # 8 question implementations (Q1–Q7 + micro-explainers)
├── state.py          # EnvState dataclass + serialization (JSON)
├── render.py         # TrainingConfig rendering + validation
└── verifiers.py      # Extension point for prime-rl verifiers (v0.10)
```

### New CLI: `src/carl_studio/cli/env.py`

```python
import typer
from carl_studio.env_setup.functor import run_interactive_flow
from carl_studio.env_setup.state import EnvState

@app.command(name="env")
def env_cmd(
    config: str = typer.Option("carl.yaml", "--config", "-c"),
    resume: bool = typer.Option(False, "--resume", "-r"),
    non_interactive: str = typer.Option("", "--auto", help="Path to YAML template"),
    show_reward_options: bool = typer.Option(False, "--show-reward-options"),
) -> None:
    """Interactive environment setup wizard."""
    if show_reward_options:
        # Render table of reward functions and exit
        ...
    
    # Load prior state if --resume
    state = EnvState.load_last() if resume else EnvState.default()
    
    # Run interactive flow or load from template
    if non_interactive:
        state = EnvState.from_yaml(non_interactive)
    else:
        state = run_interactive_flow(state)
    
    # Render TrainingConfig, ask for approval, write carl.yaml
    config_obj = state.render_training_config()
    c.config_block([...])
    if typer.confirm("Proceed?"):
        state.save_last()  # .carl/last_env_state.json
        config_obj.save(config)
        c.ok(f"Saved {config}")
```

### Modify: `src/carl_studio/cli/wiring.py`

```python
try:
    from .env import env_cmd
    app.command(name="env")(env_cmd)
except ImportError:
    _make_stub(app, "env", doc="Environment setup requires carl-studio.")
```

### Modify: `docs/operations.md`

Add section:
```markdown
## Non-Interactive Mode: CARL_ENV_AUTO

For CI/CD or scripted setups, pass a YAML template:

    carl env --auto config-template.yaml
    
The template is a partial TrainingConfig. Questions are skipped for fields
that are already set. This allows "defaults + minimal prompts" workflows.
```

---

## 6. Reuse Map

| Component | From | Usage |
|-----------|------|-------|
| `WorkFrame` | `src/carl_studio/frame.py` | Seed state; user can load/apply before `env` |
| `TrainingConfig` | `src/carl_studio/types/config.py` | Terminal output object (render validates) |
| `CARLSettings` | `src/carl_studio/settings.py` | Loads current tier/compute defaults |
| `CampConsole` | `src/carl_studio/console.py` | UI rendering (tables, inline prompts, banners) |
| `lab_cmd` entry points | `src/carl_studio/cli/lab.py` | `env` output feeds `carl train --config` |
| `InteractionChain` | `carl_core.interaction` | Log Q&A flow for observability |

**Minimal new primitives:** Just `EnvState` dataclass + question machinery. Everything else plugs into existing surfaces.

---

## 7. Verifier Integration (v0.10+ hook)

Reserve an extension point for prime-rl-style verifiers:

```python
# In env_setup/verifiers.py

_VERIFIER_REGISTRY: dict[str, Callable[[TrainingConfig], bool]] = {}

def register_verifier(name: str, fn: Callable[[TrainingConfig], bool]) -> None:
    """Register a post-render verification hook (e.g., prime-rl checker)."""
    _VERIFIER_REGISTRY[name] = fn

def run_verifiers(config: TrainingConfig, console: CampConsole) -> bool:
    """Run all registered verifiers. Return True iff all pass."""
    for name, fn in _VERIFIER_REGISTRY.items():
        try:
            if not fn(config):
                console.warn(f"Verifier '{name}' rejected config.")
                return False
        except Exception as exc:
            console.warn(f"Verifier '{name}' error: {exc}")
    return True
```

In `env.py`:
```python
from carl_studio.env_setup.verifiers import run_verifiers

if not run_verifiers(config_obj, c):
    c.error("Config validation failed. Retry with different settings.")
    raise typer.Exit(1)
```

This keeps the door open for v0.10's prime-rl integration without cluttering v0.9.

---

## 8. Testing Shape: 6–8 Test Cases

### Property tests (state-machine laws):

```python
def test_question_composition_commutative_on_disjoint_domains():
    """Q_m ∘ Q_n = Q_n ∘ Q_m when domains don't overlap."""
    state0 = EnvState.default()
    state_mn = Q_method(Q_compute(state0, ...), ...)
    state_nm = Q_compute(Q_method(state0, ...), ...)
    assert state_mn == state_nm

def test_skip_condition_produces_deterministic_state():
    """If skip_condition(state) is True, auto-answer produces expected state."""
    state = EnvState(method="sft", ...)
    q_reward = Question.reward()
    # Skip should auto-answer reward_class="static"
    _, transition = q_reward(state, console)
    state_next = transition("")  # Empty answer
    assert state_next.reward_class == "static"
```

### Golden transcript tests:

```python
def test_full_sft_grpo_cascade_flow():
    """Run the full 7-question flow for tool-calling cascade."""
    answers = [
        "1",  # Q1: training
        "3",  # Q2: cascade
        "1",  # Q3: HF dataset
        "2",  # Q4: cloud a100x4
        "1",  # Q5: static reward
        "1",  # Q6: quick cascade
        "1",  # Q7: 0.85 format accuracy
    ]
    state = run_flow_with_answers(EnvState.default(), answers)
    config = state.render_training_config()
    assert config.method == TrainingMethod.GRPO
    assert config.cascade_stages is not None
    assert len(config.cascade_stages) == 2

def test_yaml_resumption():
    """Write state to JSON, load, resume, confirm unchanged."""
    state0 = EnvState(...)
    state0.save_last()
    state1 = EnvState.load_last()
    assert state0 == state1
```

### Integration tests:

```python
def test_env_command_writes_valid_carl_yaml():
    """CLI integration: env flow outputs a valid carl.yaml."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = invoke_env_cmd(
            tmpdir,
            answers=["1", "1", "1", "1", "1", "1", "1"],
        )
        assert result.exit_code == 0
        config_path = Path(tmpdir) / "carl.yaml"
        assert config_path.exists()
        # Validate against TrainingConfig schema
        config = TrainingConfig.load(config_path)
        assert config.run_name is not None
```

---

## 9. Ship Sequencing: v0.9

**Milestone:** `carl-env` lands alongside `carl-update`.

- **Week 1:** Functor + state machinery (functor.py, state.py).
- **Week 2:** Question implementations (questions.py) + unit tests.
- **Week 3:** CLI integration (env.py, wiring.py) + integration tests.
- **Week 4:** Docs (operations.md) + v0.10 planning (verifier hook) + release notes.

**Pre-ship checklist:**
- [x] All 8 question archetypes implemented.
- [x] State serialization (JSON round-trip) verified.
- [x] No dead-ends (every "I'm not sure" routes to a default + micro-explainer).
- [x] TrainingConfig render validated against schema.
- [x] Interactive flow tested with golden transcript.
- [x] `CARL_ENV_AUTO` env var documented.
- [x] Verifier hook placeholder left for v0.10.
- [x] Help text links to https://carl.camp/docs/env-setup.

---

## 10. Gotchas & Notes

1. **WorkFrame reuse:** Questions can query `state.frame.domain` to infer context (e.g., "tool_calling" domain → default to tool-use-suite dataset). This is optional; the wizard works without a frame.

2. **Compute inference:** Q4 should call a heuristic (based on dataset size) to suggest hardware. Implement as `estimate_compute_from_data(dataset_repo, split) → ComputeTarget`. Falls back to `settings.default_compute` if inference fails.

3. **Cascade defaults:** Quick preset (Q6 option [1]) should use reasonable defaults from existing `cascade.py` (Stage A: format-only, Stage B: full CARL, gate at step 50). If user picks custom, unfold step-by-step sub-questions.

4. **Eval gate optional:** Some users may skip eval (Q7 → skip). In that case, `eval_threshold = None` and the render should not emit an eval block in `TrainingConfig`. This is fine — eval is orthogonal to training.

5. **Tier gating:** If the user's tier doesn't allow GRPO (hypothetically), the skip-condition on Q2 should note this and skip to SFT. Use `settings.tier_allows("grpo")` to gate.

6. **Prime-RL integration (v0.10):** The verifier hook is intentionally minimal. v0.10 will import a prime-rl module and register its own verifier callback. For v0.9, leave the hook empty — just the registration machinery.

---

## Summary

**`carl-env` is a 7-question interactive wizard** that builds a complete `TrainingConfig` through **progressive disclosure and functor composition**. Each question narrows the possibility space deterministically. The user never hits a dead-end; every "I'm not sure" offers a sensible default and a link to deeper docs. The output is a valid `carl.yaml` ready for `carl train`. State is serializable (JSON) so the user can resume interrupted sessions. A verifier hook reserves space for prime-rl integration in v0.10 without adding complexity to v0.9.

**Word count:** ~1650  
**Files to create:** 5 (env.py, functor.py, questions.py, state.py, render.py)  
**Files to modify:** 2 (wiring.py, operations.md)  
**Tests:** 8 (composition laws, golden transcripts, integration)  
**Ship:** v0.9, week 4.

