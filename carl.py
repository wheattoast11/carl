"""
carl.py — Coherence-Aware Reinforcement Learning
==================================================

The entire paradigm in one file.

    Models don't learn gradually — they crystallize.

Conservation law: T* = kappa * d, where kappa = 64/3, sigma = 3/16, kappa * sigma = 4.
The order parameter Phi = 1 - H(P)/log|V| measures how crystallized the model's
distribution is at each token. Phi=0 is uniform (chaos). Phi=1 is delta (crystal).

Three reward signals derived from Phi:
  1. Multiscale coherence — is Phi consistent across block scales?
  2. Cloud quality — is the selected token both likely AND from a peaked distribution?
  3. Discontinuity targeting — do confidence shifts happen at structural boundaries?

Phase transition gate: when token accuracy exceeds threshold for N consecutive
steps, the model has crystallized. Training shifts from format (SFT) to
refinement (GRPO) automatically.

Coherence gate: when Phi drops below sigma during inference, the agent abstains.
It only acts when confident. This is the alignment mechanism.

Usage:
    python carl.py phi          --logits-file logits.npy
    python carl.py reward       --logits-file logits.npy --token-ids-file ids.npy
    python carl.py gate-check   --phi 0.85
    python carl.py train        --model meta-llama/Llama-3.2-1B --dataset trl-lib/ultrafeedback
    python carl.py agent        --model ./checkpoint --task "Click the submit button"

References:
    Desai, T. & Claude Opus 4.6 (2026). Coherence-Aware Reinforcement Learning.
    Desai, T. (2026). Bounded Informational Time Crystals. DOI: 10.5281/zenodo.18906944

License: MIT — Intuition Labs LLC
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np

# ============================================================================
# CONSTANTS — the conservation law
# ============================================================================

KAPPA = 64 / 3          # 21.333...  conservation constant
SIGMA = 3 / 16          # 0.1875     semantic quantum (minimum meaningful signal)
KAPPA_SIGMA = 4.0       # bits per embedding dimension (exact)
DEFECT_THRESHOLD = 0.03  # minimum |delta_Phi| for a discontinuity event


def t_star(d: int) -> int:
    """Natural decompression boundary for embedding dimension d.

    For triadic dimensions (d = 3 * 2^k), T* is an exact power of 2.
    For non-triadic dimensions, T* is irrational — a 33% 'context tax'.

        T*(768) = 16,384 = 2^14
        T*(3072) = 65,536 = 2^16
        T*(384) = 8,192 = 2^13
    """
    return int(KAPPA * d)


# ============================================================================
# PHI — the order parameter
# ============================================================================

def compute_phi(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order parameter from logits [T, V].

    Phi_k = 1 - H(P_k) / log|V|

    Returns (phi, probs, entropy) each of shape [T].
    """
    T, V = logits.shape
    log_vocab = math.log(V)

    # Numerically stable softmax
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)

    # Shannon entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)

    # Order parameter: 0 = uniform, 1 = delta
    phi = 1.0 - (entropy / log_vocab)

    return phi, probs, entropy


# ============================================================================
# REWARDS — three components from Phi
# ============================================================================

def multiscale_coherence(phi: np.ndarray) -> float:
    """Is Phi consistent across dyadic block scales?

    For each scale j, partition phi into blocks of size 2^j and measure
    within-block standard deviation. Low std = high coherence at that scale.
    Weight by compromise profile w_j = 2^(j/2) — geometric mean of
    fine-first and coarse-first.
    """
    T = len(phi)
    if T < 2:
        return 0.5

    total_weight = 0.0
    weighted_sum = 0.0

    for j in range(min(int(math.log2(T)) + 1, 16)):
        block_size = 2 ** j
        if block_size > T:
            break
        n_blocks = T // block_size
        trimmed = phi[:n_blocks * block_size].reshape(n_blocks, block_size)
        coherence_j = max(0.0, min(1.0, 1.0 - float(np.mean(np.std(trimmed, axis=1)))))
        w_j = 2 ** (j / 2)
        weighted_sum += w_j * coherence_j
        total_weight += w_j

    return weighted_sum / total_weight if total_weight > 0 else 0.5


def cloud_quality(phi: np.ndarray, probs: np.ndarray, token_ids: np.ndarray) -> float:
    """Is the selected token both likely AND from a peaked distribution?

    Cloud quality = mean(P(selected_token) * Phi)

    A model that assigns P=0.9 from a peaked distribution (Phi=0.8) scores 0.72.
    A model that assigns P=0.9 from a diffuse distribution (Phi=0.2) scores 0.18.
    Distinguishes confident correctness from lucky correctness.
    """
    T = len(phi)
    if T == 0:
        return 0.0
    selected_probs = probs[np.arange(T), token_ids[:T]]
    return float(np.mean(selected_probs * phi))


def discontinuity_score(phi: np.ndarray) -> float:
    """Do confidence shifts happen at appropriate structural boundaries?

    Detects |delta_Phi| > 0.03 and classifies:
      Commitment (delta > +0.03): good if from low Phi (exploration to commitment)
      Dissolution (delta < -0.03): bad if from high Phi (unexpected confidence drop)
    """
    if len(phi) < 2:
        return 0.5

    delta = np.diff(phi)
    scores = []

    for k in range(len(delta)):
        dp = delta[k]
        if abs(dp) <= DEFECT_THRESHOLD:
            continue
        prev = phi[k]
        if dp > DEFECT_THRESHOLD:  # commitment
            scores.append(0.8 if prev < 0.5 else 0.3)
        else:  # dissolution
            scores.append(0.2 if prev > 0.7 else 0.5)

    return sum(scores) / len(scores) if scores else 0.5


def carl_reward(logits: np.ndarray, token_ids: np.ndarray) -> tuple[float, dict[str, float]]:
    """CARL composite reward from logits.

    R = 0.50 * multiscale + 0.30 * cloud + 0.20 * discontinuity

    Returns (reward, components_dict).
    """
    phi, probs, entropy = compute_phi(logits)
    ms = multiscale_coherence(phi)
    cq = cloud_quality(phi, probs, token_ids)
    ds = discontinuity_score(phi)
    composite = 0.5 * ms + 0.3 * cq + 0.2 * ds

    return composite, {
        "multiscale": ms,
        "cloud_quality": cq,
        "discontinuity": ds,
        "phi_mean": float(np.mean(phi)),
        "phi_std": float(np.std(phi)),
        "entropy_mean": float(np.mean(entropy)),
    }


# ============================================================================
# GATES — when to act, when to stop training
# ============================================================================

def coherence_gate(phi_mean: float, threshold: float = SIGMA) -> tuple[bool, str]:
    """Should the agent act? Phi below sigma = sub-semantic noise.

    Returns (should_act, confidence_level).
    """
    if phi_mean >= 0.8:
        return True, "high"
    elif phi_mean >= 0.5:
        return True, "medium"
    elif phi_mean >= threshold:
        return True, "low"
    return False, "abstain"


try:
    from transformers import TrainerCallback as _TrainerCallback
except ImportError:
    _TrainerCallback = object


class PhaseTransitionGate(_TrainerCallback):
    """Detects crystallization during training.

    Uses a windowed check instead of strict consecutive steps: triggers when
    `min_above` out of the last `window` steps exceed `threshold`. This is
    robust to single-batch noise — the model may oscillate around 0.99 due
    to batch difficulty variance even after crystallization.

    Default: 3 out of last 5 steps above 0.99 = crystallized.

    Usage with TRL:
        gate = PhaseTransitionGate()
        trainer = SFTTrainer(..., callbacks=[gate])
        trainer.train()
        if gate.triggered:
            # model crystallized — switch to GRPO
    """

    def __init__(self, threshold: float = 0.99, window: int = 5, min_above: int = 3):
        self.threshold = threshold
        self.window = window
        self.min_above = min_above
        self._recent: list[float] = []
        self.triggered = False
        self.trigger_step = -1
        self.peak_entropy = 0.0
        self.peak_entropy_step = -1

    def check(self, value: float, entropy: float = 0.0, step: int = 0) -> bool:
        """Check if the gate should trigger. Call once per training step."""
        if entropy > self.peak_entropy:
            self.peak_entropy = entropy
            self.peak_entropy_step = step

        self._recent.append(value)
        if len(self._recent) > self.window:
            self._recent.pop(0)

        if len(self._recent) >= self.min_above and not self.triggered:
            above = sum(1 for v in self._recent if v >= self.threshold)
            if above >= self.min_above:
                self.triggered = True
                self.trigger_step = step
                return True
        return False

    # TRL TrainerCallback interface
    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
        if logs is None:
            return
        acc = logs.get("mean_token_accuracy", 0)
        ent = logs.get("entropy", 0)
        if self.check(acc, ent, state.global_step):
            control.should_training_stop = True


# ============================================================================
# TRAIN — SFT → gated GRPO with CARL rewards
# ============================================================================

def make_carl_reward_fn(model, tokenizer):
    """Factory: returns a TRL-compatible reward function that computes CARL from logits.

    The returned function runs a torch.no_grad() forward pass per completion to
    extract logits, then computes the composite reward via numpy.
    """
    import torch

    @torch.no_grad()
    def carl_composite_reward(completions: list, **kwargs: Any) -> list[float]:
        rewards = []
        for completion in completions:
            text = completion[-1]["content"] if isinstance(completion, list) else str(completion)
            if not text.strip() or len(text) < 5:
                rewards.append(0.0)
                continue
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                was_training = model.training
                model.eval()
                try:
                    outputs = model(**inputs)
                finally:
                    if was_training:
                        model.train()
                logits_np = outputs.logits[0].cpu().float().numpy()
                ids_np = inputs["input_ids"][0].cpu().numpy()
                del outputs, inputs
                score, _ = carl_reward(logits_np, ids_np)
                rewards.append(round(score, 4))
            except Exception:
                rewards.append(0.0)
        return rewards

    return carl_composite_reward


def train(
    model_name: str,
    dataset_name: str,
    output: str = "carl-output",
    method: str = "grpo",
    max_steps: int = 300,
    gate_accuracy: float = 0.99,
    **kwargs: Any,
) -> None:
    """Train a model with CARL rewards.

    If method='sft': trains with PhaseTransitionGate, stops at crystallization.
    If method='grpo': trains with CARL composite as one of the reward functions.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
    import torch

    print(f"CARL train: {model_name} → {output}")
    print(f"  method={method} steps={max_steps}")
    print(f"  kappa={KAPPA:.4f} sigma={SIGMA:.4f} kappa*sigma={KAPPA_SIGMA}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    from datasets import load_dataset
    dataset = load_dataset(dataset_name, split="train")

    if method == "sft":
        gate = PhaseTransitionGate(threshold=gate_accuracy)
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                output_dir=output, max_steps=max_steps,
                logging_steps=1, bf16=True, gradient_checkpointing=True,
            ),
            callbacks=[gate],
        )
        trainer.train()
        if gate.triggered:
            print(f"  Crystallized at step {gate.trigger_step}")
            print(f"  Peak entropy: {gate.peak_entropy:.4f} at step {gate.peak_entropy_step}")

    elif method == "grpo":
        carl_fn = make_carl_reward_fn(model, tokenizer)
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=GRPOConfig(
                output_dir=output, max_steps=max_steps,
                num_generations=8, beta=0.0,
                logging_steps=1, bf16=True, gradient_checkpointing=True,
            ),
            reward_funcs=[carl_fn],
        )
        trainer.train()

    print(f"  Done. Model at: {output}")


# ============================================================================
# AGENT — observe → predict → gate → act → adapt
# ============================================================================

def agent_step(
    model,
    tokenizer,
    observation: str,
    instruction: str,
    gate_threshold: float = SIGMA,
) -> tuple[str | None, float, dict[str, float]]:
    """One agent step: generate a response, measure coherence, gate.

    Returns (response_or_None, phi_mean, components).
    If phi_mean < gate_threshold, returns (None, phi, components) — abstain.
    """
    import torch

    messages = [
        {"role": "system", "content": "You are a precise agent. Follow the instruction exactly."},
        {"role": "user", "content": f"{observation}\n\nInstruction: {instruction}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Compute Phi on the completion
    with torch.no_grad():
        outputs = model(output_ids)
    logits = outputs.logits[0, inputs["input_ids"].shape[1] - 1:-1]
    logits_np = logits.cpu().float().numpy()
    ids_np = new_tokens.cpu().numpy()

    if logits_np.shape[0] > 0 and len(ids_np) > 0:
        min_len = min(logits_np.shape[0], len(ids_np))
        _, components = carl_reward(logits_np[:min_len], ids_np[:min_len])
        phi_mean = components["phi_mean"]
    else:
        phi_mean = 0.0
        components = {}

    # Coherence gate
    should_act, confidence = coherence_gate(phi_mean, gate_threshold)
    if not should_act:
        return None, phi_mean, components

    return response, phi_mean, components


# ============================================================================
# CLI
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "phi":
        logits = np.load(sys.argv[sys.argv.index("--logits-file") + 1])
        phi, _, entropy = compute_phi(logits)
        print(f"Phi:     mean={np.mean(phi):.4f} std={np.std(phi):.4f}")
        print(f"Entropy: mean={np.mean(entropy):.4f}")
        print(f"T*:      {t_star(logits.shape[1])} (for V={logits.shape[1]})")

    elif cmd == "reward":
        logits = np.load(sys.argv[sys.argv.index("--logits-file") + 1])
        ids = np.load(sys.argv[sys.argv.index("--token-ids-file") + 1])
        score, components = carl_reward(logits, ids)
        print(f"CARL reward: {score:.4f}")
        for k, v in components.items():
            print(f"  {k}: {v:.4f}")

    elif cmd == "gate-check":
        phi = float(sys.argv[sys.argv.index("--phi") + 1])
        should_act, confidence = coherence_gate(phi)
        print(f"Phi={phi:.4f} → {'ACT' if should_act else 'ABSTAIN'} ({confidence})")

    elif cmd == "constants":
        print(f"kappa    = {KAPPA}")
        print(f"sigma    = {SIGMA}")
        print(f"kappa*sigma = {KAPPA_SIGMA}")
        print(f"defect_threshold = {DEFECT_THRESHOLD}")
        for d in [384, 512, 768, 1024, 3072, 4096]:
            triadic = "triadic" if (d / 3) == int(d / 3) and math.log2(d / 3) == int(math.log2(d / 3)) else ""
            print(f"  T*({d:5d}) = {t_star(d):>7d}  {triadic}")

    elif cmd == "train":
        args = {sys.argv[i].lstrip("-"): sys.argv[i + 1]
                for i in range(2, len(sys.argv) - 1, 2) if sys.argv[i].startswith("--")}
        train(
            model_name=args.get("model", "meta-llama/Llama-3.2-1B"),
            dataset_name=args.get("dataset", "trl-lib/ultrafeedback_binarized"),
            output=args.get("output", "carl-output"),
            method=args.get("method", "grpo"),
            max_steps=int(args.get("max-steps", "300")),
        )

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: phi, reward, gate-check, constants, train")


if __name__ == "__main__":
    main()
