"""
Coherence probes for the CARL Trainability Index.

Each probe measures one dimension of a model's capacity for coherence-aware
learning. Probes accept a model + tokenizer (or run in CPU-only degraded mode)
and return a ProbeResult with a letter grade and raw value.

Lazy-imports torch/transformers so the module is importable without GPU.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from carl_core.constants import DEFECT_THRESHOLD
from carl_core.math import compute_phi


# ---------------------------------------------------------------------------
# Grade map
# ---------------------------------------------------------------------------

GRADE_MAP: Dict[str, float] = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25, "F": 0.0}


def _grade_to_score(grade: str) -> float:
    return GRADE_MAP.get(grade, 0.0)


# ---------------------------------------------------------------------------
# ProbeResult
# ---------------------------------------------------------------------------


class ProbeResult(BaseModel):
    """Result of a single coherence probe."""

    probe_name: str
    grade: str  # A/B/C/D/F
    score: float  # 1.0/0.75/0.5/0.25/0.0
    value: float  # raw measured value
    detail: str  # human-readable explanation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_device(device: str) -> str:
    """Resolve 'auto' to cuda/mps/cpu."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _generate_logits(
    model: Any,
    tokenizer: Any,
    text: str,
    device: str,
    max_new_tokens: int = 64,
) -> np.ndarray:
    """
    Generate output logits from a prompt.

    Returns logits as numpy array [T, V] for the generated tokens.
    Raises RuntimeError if torch or model is unavailable.
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )

    # outputs.logits is a tuple of [batch, vocab] tensors, one per generated step
    logits_list = outputs.logits
    logits = torch.stack(logits_list, dim=0).squeeze(1)  # [T, V]
    return logits.float().cpu().numpy()


def _compute_mean_phi(logits: np.ndarray) -> float:
    """Mean Phi from logits array [T, V]."""
    if logits.shape[0] == 0:
        return 0.0
    phi, _, _ = compute_phi(logits)
    return float(np.mean(phi))


def _compute_phi_array(logits: np.ndarray) -> np.ndarray:
    """Phi trajectory from logits array [T, V]."""
    if logits.shape[0] == 0:
        return np.array([0.0])
    phi, _, _ = compute_phi(logits)
    return phi


# ---------------------------------------------------------------------------
# Domain text collections for probes
# ---------------------------------------------------------------------------

# Short representative texts per domain -- used by stability and pressure probes
_DOMAIN_TEXTS: Dict[str, List[str]] = {
    "code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "class BinaryTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None",
        "async def fetch_data(url: str) -> dict:\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as resp:\n            return await resp.json()",
    ],
    "natural_language": [
        "The process of photosynthesis converts light energy into chemical energy that can later be released to fuel the organism's activities.",
        "In quantum mechanics, the uncertainty principle states that certain pairs of physical properties cannot be simultaneously known to arbitrary precision.",
        "The Treaty of Westphalia in 1648 established the concept of state sovereignty as a foundational principle of international relations.",
    ],
    "math": [
        "Let f: R -> R be a continuous function. By the intermediate value theorem, if f(a) < 0 and f(b) > 0, then there exists c in (a,b) such that f(c) = 0.",
        "The eigenvalues of a symmetric matrix A are real. Proof: If Ax = lambda*x, then x^T A x = lambda * x^T x.",
        "Consider the integral of e^(-x^2) from -infinity to infinity. By the Gaussian integral formula, this equals sqrt(pi).",
    ],
    "instructions": [
        "Step 1: Open the terminal. Step 2: Navigate to the project directory. Step 3: Run pip install -r requirements.txt. Step 4: Execute python main.py.",
        "To configure the database: First, set the DATABASE_URL environment variable. Then run the migration script with --apply flag. Finally, verify with a SELECT 1 query.",
        "Troubleshooting guide: If the service fails to start, check the log file at /var/log/app.log. If the port is in use, find the process with lsof -i :8080.",
    ],
}

# Prompts of increasing complexity for the pressure probe
_COMPLEXITY_PROMPTS: List[Tuple[int, str]] = [
    (1, "What is 2+2?"),
    (2, "Explain why the sky appears blue in one paragraph."),
    (
        3,
        "Write a Python function that takes a list of integers and returns "
        "the longest increasing subsequence. Include the time complexity.",
    ),
    (
        4,
        "A company has three products with the following profit margins and "
        "demand elasticities. Product A: 40% margin, elasticity -1.2. "
        "Product B: 25% margin, elasticity -0.8. Product C: 60% margin, "
        "elasticity -2.1. If the company must raise prices by 10% across all "
        "products, which product should receive the smallest increase? "
        "Show your reasoning step by step.",
    ),
    (
        5,
        "Design a distributed consensus protocol for a network of 7 nodes "
        "where up to 2 nodes may exhibit Byzantine faults. The protocol must "
        "guarantee safety under asynchrony and liveness under partial synchrony. "
        "Provide pseudocode, prove the safety property, and analyze message "
        "complexity. Compare your design to PBFT and Tendermint.",
    ),
]

# Adaptation prompts -- simple pattern-matching tasks for TTT simulation
_ADAPTATION_TASKS: List[Dict[str, str]] = [
    {"prompt": "Translate to French: Hello", "expected_prefix": "Bonjour"},
    {"prompt": "Translate to French: Thank you", "expected_prefix": "Merci"},
    {"prompt": "Translate to French: Good morning", "expected_prefix": "Bon"},
    {"prompt": "Translate to French: Goodbye", "expected_prefix": "Au revoir"},
    {"prompt": "Translate to French: Please", "expected_prefix": "S'il"},
    {"prompt": "Translate to French: Yes", "expected_prefix": "Oui"},
    {"prompt": "Translate to French: No", "expected_prefix": "Non"},
    {"prompt": "Translate to French: Water", "expected_prefix": "Eau"},
    {"prompt": "Translate to French: Book", "expected_prefix": "Livre"},
    {"prompt": "Translate to French: Sun", "expected_prefix": "Soleil"},
]


# ---------------------------------------------------------------------------
# 1. PhaseTransitionProbe
# ---------------------------------------------------------------------------


class PhaseTransitionProbe:
    """
    Measures steps to crystallize a new capability.

    Approach: Generate responses to a fixed set of prompts, tracking how Phi
    evolves as prompt complexity increases. The probe measures how quickly
    the model reaches a high-Phi regime -- a proxy for the speed at which
    coherence crystallizes under new input pressure.

    In full-training mode this would fine-tune for N steps. The simplified
    version here estimates crystallization speed from the Phi response curve
    across increasingly challenging prompts, scaled to equivalent training
    steps.

    Grade thresholds (equivalent steps to Phi >= 0.5):
        A: < 30     B: 30-50     C: 50-100     D: 100-200     F: > 200
    """

    PHI_THRESHOLD = 0.5

    def run(
        self,
        model: Any = None,
        tokenizer: Any = None,
        *,
        device: str = "auto",
        prompts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ProbeResult:
        device = _get_device(device)

        if model is None or tokenizer is None:
            return ProbeResult(
                probe_name="phase_transition",
                grade="F",
                score=0.0,
                value=999.0,
                detail="Model/tokenizer not provided -- cannot measure phase transition speed.",
            )

        try:
            import torch  # noqa: F401
        except ImportError:
            return ProbeResult(
                probe_name="phase_transition",
                grade="F",
                score=0.0,
                value=999.0,
                detail="GPU not available (torch not installed).",
            )

        if prompts is None:
            # Use complexity prompts -- each level represents ~40 equivalent steps
            prompts = [text for _, text in _COMPLEXITY_PROMPTS]

        phi_values: List[float] = []
        for prompt in prompts:
            try:
                logits = _generate_logits(model, tokenizer, prompt, device)
                phi_values.append(_compute_mean_phi(logits))
            except Exception as e:
                phi_values.append(0.0)

        # Find first prompt level where Phi crosses threshold
        crossed_at: Optional[int] = None
        for i, phi in enumerate(phi_values):
            if phi >= self.PHI_THRESHOLD:
                crossed_at = i
                break

        # Map prompt index to equivalent training steps
        # Each complexity level ~ 40 equivalent steps of fine-tuning pressure
        steps_per_level = 40
        if crossed_at is not None:
            equiv_steps = crossed_at * steps_per_level
        else:
            # Never crossed -- estimate from best phi achieved
            best_phi = max(phi_values) if phi_values else 0.0
            if best_phi > 0:
                # Extrapolate: how many more levels to reach threshold
                ratio = self.PHI_THRESHOLD / best_phi
                equiv_steps = int(len(prompts) * steps_per_level * ratio)
            else:
                equiv_steps = 999

        # Grade
        if equiv_steps < 30:
            grade = "A"
        elif equiv_steps < 50:
            grade = "B"
        elif equiv_steps < 100:
            grade = "C"
        elif equiv_steps < 200:
            grade = "D"
        else:
            grade = "F"

        phi_summary = ", ".join(f"{p:.3f}" for p in phi_values)
        return ProbeResult(
            probe_name="phase_transition",
            grade=grade,
            score=_grade_to_score(grade),
            value=float(equiv_steps),
            detail=(
                f"Equivalent steps to Phi>={self.PHI_THRESHOLD}: {equiv_steps}. "
                f"Phi per level: [{phi_summary}]."
            ),
        )


# ---------------------------------------------------------------------------
# 2. PhiStabilityProbe
# ---------------------------------------------------------------------------


class PhiStabilityProbe:
    """
    Measures Phi robustness across domain shift.

    Computes mean Phi on texts from domain A (code) and domain B (natural
    language), then measures |delta_phi|. A model with stable coherence
    maintains similar Phi across domains.

    Grade thresholds (|delta_phi|):
        A: < 0.05     B: 0.05-0.1     C: 0.1-0.2     D: 0.2-0.5     F: > 0.5
    """

    def run(
        self,
        model: Any = None,
        tokenizer: Any = None,
        *,
        device: str = "auto",
        domain_a: str = "code",
        domain_b: str = "natural_language",
        **kwargs: Any,
    ) -> ProbeResult:
        device = _get_device(device)

        if model is None or tokenizer is None:
            return ProbeResult(
                probe_name="phi_stability",
                grade="F",
                score=0.0,
                value=1.0,
                detail="Model/tokenizer not provided -- cannot measure Phi stability.",
            )

        try:
            import torch  # noqa: F401
        except ImportError:
            return ProbeResult(
                probe_name="phi_stability",
                grade="F",
                score=0.0,
                value=1.0,
                detail="GPU not available (torch not installed).",
            )

        texts_a = _DOMAIN_TEXTS.get(domain_a, _DOMAIN_TEXTS["code"])
        texts_b = _DOMAIN_TEXTS.get(domain_b, _DOMAIN_TEXTS["natural_language"])

        def _mean_phi_for_texts(texts: List[str]) -> float:
            phis: List[float] = []
            for text in texts:
                try:
                    logits = _generate_logits(model, tokenizer, text, device)
                    phis.append(_compute_mean_phi(logits))
                except Exception:
                    phis.append(0.0)
            return float(np.mean(phis)) if phis else 0.0

        phi_a = _mean_phi_for_texts(texts_a)
        phi_b = _mean_phi_for_texts(texts_b)
        delta = abs(phi_a - phi_b)

        if delta < 0.05:
            grade = "A"
        elif delta < 0.1:
            grade = "B"
        elif delta < 0.2:
            grade = "C"
        elif delta < 0.5:
            grade = "D"
        else:
            grade = "F"

        return ProbeResult(
            probe_name="phi_stability",
            grade=grade,
            score=_grade_to_score(grade),
            value=delta,
            detail=(
                f"|delta_phi| = {delta:.4f} "
                f"(domain '{domain_a}' Phi={phi_a:.4f}, "
                f"domain '{domain_b}' Phi={phi_b:.4f})."
            ),
        )


# ---------------------------------------------------------------------------
# 3. CoherenceUnderPressureProbe
# ---------------------------------------------------------------------------


class CoherenceUnderPressureProbe:
    """
    Measures how Phi degrades as task complexity increases.

    Computes Phi on 5 prompts of escalating complexity and classifies the
    degradation profile:

    - plateau:  Phi stays within 0.1 of peak across all levels
    - graceful: Phi drops monotonically but final >= 50% of peak
    - cliff:    Phi drops sharply (>0.3) at one transition
    - chaotic:  Phi is non-monotonic with variance > 0.05
    - collapse: Final Phi < 25% of peak

    Grade: A(plateau), B(graceful), C(cliff), D(chaotic), F(collapse)
    """

    def run(
        self,
        model: Any = None,
        tokenizer: Any = None,
        *,
        device: str = "auto",
        **kwargs: Any,
    ) -> ProbeResult:
        device = _get_device(device)

        if model is None or tokenizer is None:
            return ProbeResult(
                probe_name="coherence_pressure",
                grade="F",
                score=0.0,
                value=0.0,
                detail="Model/tokenizer not provided -- cannot measure coherence under pressure.",
            )

        try:
            import torch  # noqa: F401
        except ImportError:
            return ProbeResult(
                probe_name="coherence_pressure",
                grade="F",
                score=0.0,
                value=0.0,
                detail="GPU not available (torch not installed).",
            )

        phi_levels: List[float] = []
        for _, prompt in _COMPLEXITY_PROMPTS:
            try:
                logits = _generate_logits(model, tokenizer, prompt, device)
                phi_levels.append(_compute_mean_phi(logits))
            except Exception:
                phi_levels.append(0.0)

        profile, profile_value = self._classify_profile(phi_levels)

        grade_map = {
            "plateau": "A",
            "graceful": "B",
            "cliff": "C",
            "chaotic": "D",
            "collapse": "F",
        }
        grade = grade_map.get(profile, "F")

        phi_summary = ", ".join(f"L{i+1}={p:.3f}" for i, p in enumerate(phi_levels))
        return ProbeResult(
            probe_name="coherence_pressure",
            grade=grade,
            score=_grade_to_score(grade),
            value=profile_value,
            detail=f"Profile: {profile}. Phi by complexity: [{phi_summary}].",
        )

    @staticmethod
    def _classify_profile(phi_levels: List[float]) -> Tuple[str, float]:
        """
        Classify the Phi-vs-complexity profile.

        Returns (profile_name, summary_value) where summary_value is
        a normalized degradation metric in [0, 1]:
          0.0 = perfect plateau
          1.0 = total collapse
        """
        if not phi_levels:
            return "collapse", 1.0

        arr = np.array(phi_levels)
        peak = float(np.max(arr))

        if peak < 1e-6:
            return "collapse", 1.0

        normalized = arr / peak
        final_ratio = normalized[-1]

        # Check collapse first
        if final_ratio < 0.25:
            return "collapse", 1.0 - final_ratio

        # Check plateau: all within 0.1 of peak
        if float(np.max(np.abs(normalized - 1.0))) < 0.1:
            return "plateau", float(1.0 - np.mean(normalized))

        # Check cliff: any single drop > 0.3
        diffs = np.diff(normalized)
        if float(np.min(diffs)) < -0.3:
            return "cliff", float(-np.min(diffs))

        # Check chaotic: non-monotonic with high variance
        if float(np.var(normalized)) > 0.05:
            sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
            if sign_changes >= 2:
                return "chaotic", float(np.var(normalized))

        # Graceful degradation
        if final_ratio >= 0.5:
            return "graceful", float(1.0 - final_ratio)

        # Default to cliff if we get here
        return "cliff", float(1.0 - final_ratio)


# ---------------------------------------------------------------------------
# 4. AdaptationRateProbe
# ---------------------------------------------------------------------------


class AdaptationRateProbe:
    """
    Measures test-time training iterations to 80% success.

    Simulates TTT by measuring how Phi responds across a series of related
    prompts. The probe tracks whether the model's coherence (Phi) on later
    prompts in a task sequence improves relative to the first -- a proxy for
    within-context adaptation.

    Full TTT mode (when SLOT/LoRA is available) would do actual gradient
    steps. Simplified mode measures Phi improvement across the task sequence
    as a proxy.

    Grade (iterations to 80% accuracy proxy):
        A: < 5     B: 5-10     C: 10-20     D: 20-50     F: > 50 or never
    """

    SUCCESS_THRESHOLD = 0.80

    def run(
        self,
        model: Any = None,
        tokenizer: Any = None,
        *,
        device: str = "auto",
        tasks: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> ProbeResult:
        device = _get_device(device)

        if model is None or tokenizer is None:
            return ProbeResult(
                probe_name="adaptation_rate",
                grade="F",
                score=0.0,
                value=999.0,
                detail="Model/tokenizer not provided -- cannot measure adaptation rate.",
            )

        try:
            import torch  # noqa: F401
        except ImportError:
            return ProbeResult(
                probe_name="adaptation_rate",
                grade="F",
                score=0.0,
                value=999.0,
                detail="GPU not available (torch not installed).",
            )

        if tasks is None:
            tasks = _ADAPTATION_TASKS

        # Build cumulative context: each successive prompt includes prior
        # prompt-response pairs as context, simulating TTT memory
        successes = 0
        total = len(tasks)
        context_prefix = ""
        iterations_to_threshold: Optional[int] = None

        for i, task in enumerate(tasks):
            prompt_with_context = context_prefix + task["prompt"]

            try:
                logits = _generate_logits(
                    model, tokenizer, prompt_with_context, device, max_new_tokens=32
                )
                # Decode generated tokens to check for expected prefix
                import torch

                inputs = tokenizer(
                    prompt_with_context,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    gen_output = model.generate(
                        **inputs, max_new_tokens=32, do_sample=False
                    )

                # Decode only the generated part
                input_len = inputs["input_ids"].shape[1]
                generated_ids = gen_output[0][input_len:]
                generated_text = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()

                if generated_text.startswith(task["expected_prefix"]):
                    successes += 1
                    # Add successful pair to context for subsequent prompts
                    context_prefix += (
                        f"{task['prompt']}\n{generated_text}\n\n"
                    )
                else:
                    # Still add the attempt to context (TTT learns from errors too)
                    context_prefix += (
                        f"{task['prompt']}\n{generated_text}\n\n"
                    )

            except Exception:
                pass

            # Check if we've reached threshold
            current_rate = successes / (i + 1)
            if (
                current_rate >= self.SUCCESS_THRESHOLD
                and (i + 1) >= 3
                and iterations_to_threshold is None
            ):
                iterations_to_threshold = i + 1

        if iterations_to_threshold is None:
            final_rate = successes / max(total, 1)
            if final_rate >= self.SUCCESS_THRESHOLD:
                iterations_to_threshold = total
            else:
                # Extrapolate if partial success
                if successes > 0:
                    estimated = int(
                        total * (self.SUCCESS_THRESHOLD / (final_rate + 1e-6))
                    )
                    iterations_to_threshold = min(estimated, 999)
                else:
                    iterations_to_threshold = 999

        # Grade
        iters = iterations_to_threshold
        if iters < 5:
            grade = "A"
        elif iters < 10:
            grade = "B"
        elif iters < 20:
            grade = "C"
        elif iters < 50:
            grade = "D"
        else:
            grade = "F"

        return ProbeResult(
            probe_name="adaptation_rate",
            grade=grade,
            score=_grade_to_score(grade),
            value=float(iters),
            detail=(
                f"Iterations to {self.SUCCESS_THRESHOLD:.0%} success: {iters}. "
                f"Final accuracy: {successes}/{total} "
                f"({successes/max(total,1):.0%})."
            ),
        )
