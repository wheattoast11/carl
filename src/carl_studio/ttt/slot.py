"""
SLOT & LoRA Micro-Update — Test-Time Training Mechanisms
=========================================================
SLOT: Per-sample hidden delta optimization (non-parametric, ephemeral).
LoRA Micro-Update: Rank-1 per-interaction weight adaptation (parametric, persistent).

Architecture-agnostic: auto-discovers transformer layers for Llama, Qwen,
Mistral, Gemma, GPT2, GPT-J, LLaVA, Idefics, and encoder-decoder models.

Part of the CARL Studio TTT (test-time training) subpackage.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _find_layers(model: torch.nn.Module) -> list:
    """Find the transformer layer list regardless of model architecture.

    Tries common paths in order of specificity. Works for Llama, Qwen, Mistral,
    Gemma, GPT2, GPT-J, LLaVA, Idefics, encoder-decoder models, etc.
    """
    candidates = [
        "model.layers",                # Llama, Qwen, Mistral, Gemma
        "language_model.model.layers",  # Some VLMs (LLaVA, Idefics)
        "transformer.h",               # GPT2, GPT-J
        "transformer.layers",          # Some custom models
        "encoder.layers",              # Encoder models
        "model.decoder.layers",        # Encoder-decoder
    ]

    for path in candidates:
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__") and len(obj) > 0:
            return obj

    raise ValueError(
        f"Cannot find transformer layers on {type(model).__name__}. "
        f"Tried: {candidates}. Pass inject_layer_idx and override "
        f"_find_layers() for custom architectures."
    )


@dataclass
class SLOTResult:
    """Result of a SLOT optimization pass."""
    delta: torch.Tensor
    losses: list[float]
    initial_loss: float
    final_loss: float


class SLOTOptimizer:
    """Self-correcting Layer Optimization at Test-time.

    Optimizes a per-sample delta vector added to hidden states at a middle
    transformer layer. The delta steers the prompt representation so the model
    produces better coordinate predictions for this specific screenshot.

    The delta is:
      - Non-parametric: optimized per input, does not touch model weights
      - Ephemeral: discarded after each interaction (or kept for re-use)
      - Cheap: 8 forward-backward passes through frozen model
    """

    def __init__(
        self,
        model: torch.nn.Module,
        d_model: int,
        inject_layer_idx: int | None = None,
        steps: int = 8,
        lr: float = 0.002,
    ):
        self.model = model
        self.d_model = d_model
        self.steps = steps
        self.lr = lr

        # Inject at the middle transformer layer by default
        layers = _find_layers(model)
        n_layers = len(layers)
        self.inject_idx = inject_layer_idx if inject_layer_idx is not None else n_layers // 2
        self._inject_layer = layers[self.inject_idx]  # Cache to avoid re-walking model graph

    def optimize(
        self,
        processor,
        screenshot,
        instruction: str,
        target_text: str,
        system_prompt: str,
        chat_template_kwargs: dict | None = None,
    ) -> SLOTResult:
        """Optimize delta to push model toward target_text for this input.

        Args:
            processor: AutoProcessor for the VLM
            screenshot: PIL.Image of the current screen
            instruction: Task instruction
            target_text: Target coordinate string, e.g. "(123, 456)"
            system_prompt: System prompt used in training
            chat_template_kwargs: Extra kwargs for apply_chat_template

        Returns:
            SLOTResult with optimized delta and loss trajectory
        """
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

        # Build full conversation including assistant response (target)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {"type": "text", "text": instruction},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
        ]

        # Process through the VLM processor (handles image tokens)
        _ct_kwargs = chat_template_kwargs if chat_template_kwargs is not None else {}
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **_ct_kwargs,
        ).to(self.model.device)

        # Tokenize just the target to find its length
        target_ids = tokenizer.encode(target_text, add_special_tokens=False, return_tensors="pt")
        target_ids = target_ids.to(self.model.device)
        target_len = target_ids.shape[1]

        # Probe forward pass to get expanded sequence length
        # (image tokens expand inside the model)
        with torch.no_grad():
            probe_out = self.model(**inputs)
        total_seq_len = probe_out.logits.shape[1]

        # Target tokens are at the END of the sequence (plain text, no expansion)
        target_start = total_seq_len - target_len

        # Initialize delta: single vector broadcast across prompt positions
        delta = torch.zeros(
            1, 1, self.d_model,
            device=self.model.device,
            dtype=next(self.model.parameters()).dtype,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        inject_layer = self._inject_layer
        losses = []

        for step in range(self.steps):
            # Register hook to inject delta at prompt positions
            def make_hook(d):
                def hook_fn(module, inp, output):
                    hs = output[0]
                    # Add delta to all prompt positions (before target)
                    modified = hs.clone()
                    modified[:, :target_start, :] = modified[:, :target_start, :] + d
                    return (modified,) + output[1:]
                return hook_fn

            handle = inject_layer.register_forward_hook(make_hook(delta))

            outputs = self.model(**inputs)

            # Loss on target token positions
            # logits at [target_start-1 : target_start+target_len-1] predict target_ids
            target_logits = outputs.logits[0, target_start - 1 : target_start + target_len - 1]
            loss = F.cross_entropy(target_logits, target_ids[0])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            handle.remove()
            losses.append(loss.item())

        return SLOTResult(
            delta=delta.detach(),
            losses=losses,
            initial_loss=losses[0] if losses else 0.0,
            final_loss=losses[-1] if losses else 0.0,
        )

    def predict_with_delta(
        self,
        processor,
        screenshot,
        instruction: str,
        system_prompt: str,
        delta: torch.Tensor,
        max_new_tokens: int = 64,
        chat_template_kwargs: dict | None = None,
    ) -> str:
        """Generate coordinates with SLOT delta injected into hidden states."""
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {"type": "text", "text": instruction},
                ],
            },
        ]

        _ct_kwargs = chat_template_kwargs if chat_template_kwargs is not None else {}
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **_ct_kwargs,
        ).to(self.model.device)

        inject_layer = self._inject_layer

        def hook_fn(module, inp, output):
            hs = output[0]
            modified = hs.clone()
            modified = modified + delta  # broadcast delta across all positions
            return (modified,) + output[1:]

        handle = inject_layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        handle.remove()

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class LoRAMicroUpdate:
    """Rank-1 LoRA update after each interaction.

    Persistent weight adaptation: each interaction contributes a tiny gradient
    step to rank-1 LoRA weights. Over many interactions, the model adapts to
    the deployment environment.

    This is the novel contribution of Paper 2 -- neither ephemeral (like SLOT)
    nor batch-dependent (like online GRPO).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_modules: list[str] | None = None,
        rank: int = 1,
        lr: float = 1e-5,
    ):
        from peft import LoraConfig, get_peft_model

        self.lr = lr
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
        ]

        config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=self.target_modules,
            task_type="CAUSAL_LM",
            lora_dropout=0,
        )
        self.model = get_peft_model(model, config)
        self.model.eval()

        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
        )
        self.update_count = 0
        self.loss_history: list[float] = []

    def update(
        self,
        inputs: dict,
        target_ids: torch.Tensor,
        total_seq_len: int,
        target_len: int,
    ) -> float:
        """Single gradient step on LoRA weights.

        Args:
            inputs: Full model inputs (prompt + target)
            target_ids: Token ids for the target response
            total_seq_len: Expanded sequence length from probe pass
            target_len: Length of target tokens

        Returns:
            Loss value
        """
        target_start = total_seq_len - target_len

        self.model.train()
        outputs = self.model(**inputs)

        target_logits = outputs.logits[0, target_start - 1 : target_start + target_len - 1]
        loss = F.cross_entropy(target_logits, target_ids[0])

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model.eval()

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        self.update_count += 1
        return loss_val

    def get_stats(self) -> dict:
        """Return summary statistics for LoRA micro-updates."""
        return {
            "updates": self.update_count,
            "mean_loss": sum(self.loss_history) / max(len(self.loss_history), 1),
            "recent_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
