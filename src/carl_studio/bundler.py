"""Generate self-contained training scripts for HF Jobs.

HF Jobs ``run_uv_job()`` uploads a single .py file.  Any
``from carl_studio import ...`` would fail on the remote worker.

The Bundler takes a ``TrainingConfig`` and emits a complete script with:
  - PEP 723 inline dependency metadata
  - All reward functions inlined
  - All CARL crystal math inlined
  - Cascade manager inlined
  - Model loading + dataset formatting + training loop
"""

from __future__ import annotations

import textwrap

from carl_studio.types.config import TrainingConfig


class Bundler:
    """Generate self-contained Python scripts from TrainingConfig.

    The generated script includes:
    - PEP 723 inline dependencies
    - All reward functions inlined
    - All CARL crystal math inlined
    - Cascade manager inlined
    - Model loading + dataset formatting + training loop
    """

    def generate(self, config: TrainingConfig) -> str:
        """Generate a self-contained training script."""
        if config.method.value in ("grpo",):
            return self._generate_grpo(config)
        elif config.method.value == "sft":
            return self._generate_sft(config)
        else:
            raise ValueError(f"Bundler does not yet support method: {config.method}")

    def _generate_sft(self, config: TrainingConfig) -> str:
        return textwrap.dedent(f'''\
            # /// script
            # requires-python = ">=3.10"
            # dependencies = ["trl>=1.0", "peft>=0.13", "transformers>=5.0",
            #     "datasets>=3.0", "accelerate>=0.24.0", "trackio", "bitsandbytes", "numpy"]
            # ///
            """CARL Studio — Bundled SFT Script (auto-generated)."""
            import os, json, trackio
            from datasets import load_dataset
            from peft import LoraConfig
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from trl import SFTTrainer, SFTConfig

            hf_token = os.environ.get("HF_TOKEN")
            tokenizer = AutoTokenizer.from_pretrained(
                "{config.tokenizer_source or config.base_model}",
                token=hf_token, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                "{config.base_model}",
                quantization_config=BitsAndBytesConfig(load_in_8bit={config.quantization.load_in_8bit}),
                device_map="auto", trust_remote_code=True, token=hf_token)

            dataset = load_dataset("{config.dataset_repo}", split="{config.dataset_split}", token=hf_token)

            peft_config = LoraConfig(
                r={config.lora.r}, lora_alpha={config.lora.alpha},
                target_modules={config.lora.target_modules},
                task_type="CAUSAL_LM", lora_dropout={config.lora.dropout})

            training_config = SFTConfig(
                output_dir="carl-sft-output",
                push_to_hub={config.push_to_hub},
                hub_model_id="{config.output_repo}",
                hub_strategy="{config.hub_strategy}",
                hub_token=hf_token,
                num_train_epochs={config.num_train_epochs},
                per_device_train_batch_size={config.per_device_train_batch_size},
                gradient_accumulation_steps={config.gradient_accumulation_steps},
                learning_rate={config.learning_rate},
                max_length={config.max_length},
                logging_steps=1, logging_first_step=True,
                save_strategy="steps", save_steps=50, save_total_limit=4,
                warmup_ratio={config.warmup_ratio}, lr_scheduler_type="{config.lr_scheduler_type}",
                bf16={config.bf16}, gradient_checkpointing=True, optim="adamw_8bit",
                report_to="trackio", project="carl-studio", run_name="{config.run_name}")

            trainer = SFTTrainer(
                model=model, processing_class=tokenizer,
                train_dataset=dataset, args=training_config, peft_config=peft_config)

            trainer.train()
            trainer.push_to_hub()
            trackio.finish()
            print(f"Done! Model at: https://huggingface.co/{config.output_repo}")
        ''')

    def _generate_grpo(self, config: TrainingConfig) -> str:
        # Compute cascade boundaries from config or use 2-stage defaults
        carl_start = 50
        total_steps = 1000
        use_3_stage = False
        stage_b_start = 100  # only used in 3-stage mode
        stage_c_start = 200  # only used in 3-stage mode
        if config.cascade_stages:
            if len(config.cascade_stages) >= 3:
                use_3_stage = True
                cumulative = 0
                for i, stage in enumerate(config.cascade_stages):
                    cumulative += stage.steps
                    if i == 0:
                        stage_b_start = cumulative
                    elif i == 1:
                        stage_c_start = cumulative
                total_steps = cumulative
            elif len(config.cascade_stages) == 2:
                carl_start = config.cascade_stages[0].steps
                total_steps = config.cascade_stages[0].steps + config.cascade_stages[1].steps
        if config.max_steps > 0:
            total_steps = config.max_steps

        return textwrap.dedent(f'''\
            # /// script
            # requires-python = ">=3.10"
            # dependencies = ["trl>=1.0", "peft>=0.13", "transformers>=5.0",
            #     "datasets>=3.0", "accelerate>=0.24.0", "trackio", "bitsandbytes", "numpy"]
            # ///
            """CARL Studio — Bundled GRPO+CARL Script (auto-generated).

            Self-contained for HF Jobs. All reward functions, crystal math,
            cascade manager, and callbacks are inlined.
            """
            import gc
            import json
            import math
            import os
            import re
            import sys
            import threading
            from typing import Any

            import numpy as np
            import torch
            import trackio
            from datasets import load_dataset
            from peft import LoraConfig
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                TrainerCallback,
            )
            from trl import GRPOConfig, GRPOTrainer


            # ===================================================================
            # Crystal constants
            # ===================================================================

            KAPPA = 64 / 3        # = 21.333... conservation constant
            SIGMA = 3 / 16        # = 0.1875    semantic quantum
            DEFECT_THRESHOLD = 0.03


            # ===================================================================
            # Text extraction helper
            # ===================================================================

            def _extract_text(completion: str | list[dict[str, Any]]) -> str:
                """Extract plain text from a completion (string or conversational)."""
                if isinstance(completion, str):
                    return completion
                if isinstance(completion, list):
                    for msg in reversed(completion):
                        if isinstance(msg, dict) and msg.get("content"):
                            return str(msg["content"])
                    return ""
                return str(completion)


            def _extract_json(text: str) -> dict[str, Any] | None:
                """Extract a tool-call dict from text (fenced or bare JSON)."""
                fenced = re.search(r"```(?:json)?\\s*\\n?(.*?)\\n?\\s*```", text, re.DOTALL)
                if fenced:
                    parsed = _try_parse_tool_json(fenced.group(1).strip())
                    if parsed is not None:
                        return parsed
                start = text.find("{{")
                if start == -1:
                    return None
                depth, end = 0, -1
                for i in range(start, len(text)):
                    if text[i] == "{{":
                        depth += 1
                    elif text[i] == "}}":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                if end == -1:
                    return None
                return _try_parse_tool_json(text[start : end + 1])


            def _try_parse_tool_json(raw: str) -> dict[str, Any] | None:
                try:
                    obj = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    return None
                if not isinstance(obj, dict) or "name" not in obj:
                    return None
                args = obj.get("arguments")
                if isinstance(args, str):
                    try:
                        obj["arguments"] = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        pass
                return obj


            # ===================================================================
            # Crystal math: compute_phi
            # ===================================================================

            def compute_phi(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                """Order parameter Phi from logits [T, V].
                Returns (phi [T], probs [T,V], entropy [T])."""
                T, V = logits.shape
                log_vocab = math.log(V)
                logits_shifted = logits - logits.max(axis=-1, keepdims=True)
                exp_logits = np.exp(logits_shifted)
                probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
                log_probs = np.log(probs + 1e-10)
                entropy = -np.sum(probs * log_probs, axis=-1)
                phi = 1.0 - (entropy / log_vocab)
                return phi, probs, entropy


            # ===================================================================
            # _logits_to_crystal_metrics (composite CARL scoring)
            # ===================================================================

            def _cloud_quality(logits: np.ndarray, token_ids: np.ndarray) -> float:
                T, V = logits.shape
                if T == 0:
                    return 0.0
                phi, probs, entropy = compute_phi(logits)
                selected_probs = probs[np.arange(T), token_ids]
                return float(np.mean(selected_probs * phi))


            def _multiscale_coherence(logits: np.ndarray, token_ids: np.ndarray) -> float:
                T, V = logits.shape
                if T < 1:
                    return 0.5
                phi, probs, entropy = compute_phi(logits)
                N_max = int(math.log2(max(T, 1)))
                total_weight, weighted_sum = 0.0, 0.0
                for j in range(min(N_max + 1, 16)):
                    block_size = 2 ** j
                    if block_size > T:
                        break
                    n_blocks = T // block_size
                    trimmed = phi[:n_blocks * block_size].reshape(n_blocks, block_size)
                    block_stds = np.std(trimmed, axis=1)
                    coherence_j = float(max(0.0, min(1.0, 1.0 - float(np.mean(block_stds)))))
                    w_j = 2 ** (j / 2)
                    weighted_sum += w_j * coherence_j
                    total_weight += w_j
                return float(weighted_sum / total_weight) if total_weight > 0 else 0.5


            def _discontinuity(logits: np.ndarray, token_ids: np.ndarray) -> float:
                T, V = logits.shape
                if T < 2:
                    return 0.5
                phi, probs, entropy = compute_phi(logits)
                delta_phi = np.diff(phi)
                defect_scores: list[float] = []
                for k in range(len(delta_phi)):
                    dp = delta_phi[k]
                    if abs(dp) <= DEFECT_THRESHOLD:
                        continue
                    prev_order = phi[k]
                    if dp > DEFECT_THRESHOLD:
                        defect_scores.append(0.8 if prev_order < 0.5 else 0.3)
                    else:
                        defect_scores.append(0.2 if prev_order > 0.7 else 0.5)
                return float(sum(defect_scores) / len(defect_scores)) if defect_scores else 0.5


            def _logits_to_crystal_metrics(
                logits: np.ndarray, token_ids: np.ndarray,
            ) -> tuple[float, dict[str, float]]:
                """CARL composite: 50% multiscale + 30% cloud + 20% discontinuity."""
                ms = _multiscale_coherence(logits, token_ids)
                cq = _cloud_quality(logits, token_ids)
                disc = _discontinuity(logits, token_ids)
                composite = 0.5 * ms + 0.3 * cq + 0.2 * disc
                return float(composite), {{"multiscale": ms, "cloud_quality": cq, "discontinuity": disc}}


            # ===================================================================
            # make_carl_reward factory (L1, L2, L8, L9 fixes)
            # ===================================================================

            def make_carl_reward(
                model, tokenizer, vocab_size: int = 128000,
                active_after_step: int = 0, max_length: int = {config.max_length},
            ):
                """TRL-compatible CARL reward closure with all production fixes."""
                _step_counter = [0]
                _last_metrics: list[Any] = [None]
                _metrics_lock = threading.Lock()  # Fix L2

                @torch.no_grad()
                def carl_composite_reward(completions: list, **kwargs: Any) -> list[float]:
                    _step_counter[0] += 1
                    if _step_counter[0] < active_after_step:
                        return [0.0] * len(completions)
                    rewards: list[float] = []
                    batch_metrics: list[dict[str, float]] = []
                    for completion in completions:
                        text = _extract_text(completion)
                        if not text.strip() or len(text) < 10:
                            rewards.append(0.0)
                            continue
                        try:
                            inputs = tokenizer(
                                text, return_tensors="pt", truncation=True,
                                max_length=max_length, padding=False,
                            )
                            inputs = {{k: v.to(model.device) for k, v in inputs.items()}}
                            # Fix L1: eval/train toggle
                            was_training = model.training
                            model.eval()
                            try:
                                outputs = model(**inputs)
                            finally:
                                if was_training:
                                    model.train()
                            logits_t = outputs.logits[0]
                            token_ids_t = inputs["input_ids"][0]
                            logits_np = logits_t.cpu().float().numpy()
                            token_ids_np = token_ids_t.cpu().numpy()
                            # Fix L8: VRAM cleanup
                            del outputs, logits_t, inputs
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            score, components = _logits_to_crystal_metrics(logits_np, token_ids_np)
                            batch_metrics.append(components)
                            rewards.append(round(float(score), 4))
                        except Exception as e:
                            # Fix L9: log OOM to stderr
                            print(f"[CARL] Forward pass failed: {{e}}", file=sys.stderr)
                            rewards.append(0.0)
                    if batch_metrics:
                        with _metrics_lock:
                            _last_metrics[0] = (_step_counter[0], batch_metrics)
                    return rewards

                carl_composite_reward._last_metrics = _last_metrics
                carl_composite_reward._metrics_lock = _metrics_lock
                carl_composite_reward._step = _step_counter
                return carl_composite_reward


            # ===================================================================
            # Tool registry + Task rewards R1-R5
            # ===================================================================

            TOOL_REGISTRY = {{
                "read_file": {{"required": ["path"]}},
                "write_file": {{"required": ["path", "content"]}},
                "list_directory": {{"required": ["path"]}},
                "web_search": {{"required": ["query"]}},
                "execute_code": {{"required": ["code"]}},
                "search_codebase": {{"required": ["pattern"]}},
                "run_shell": {{"required": ["command"]}},
                "git_operation": {{"required": ["operation"]}},
            }}
            TOOL_NAMES = set(TOOL_REGISTRY.keys())

            _FENCED_BLOCK_RE = re.compile(r"```[\\s\\S]*?```")


            def _get_required_args(name: str) -> list[str]:
                t = TOOL_REGISTRY.get(name)
                return t["required"] if t else []


            def _score_format(text: str) -> float:
                if "{{" not in text:
                    return 0.0
                tool_call = _extract_json(text)
                if tool_call is None:
                    start = text.find("{{")
                    if start != -1:
                        depth, end = 0, -1
                        for i in range(start, len(text)):
                            if text[i] == "{{":
                                depth += 1
                            elif text[i] == "}}":
                                depth -= 1
                                if depth == 0:
                                    end = i
                                    break
                        if end != -1:
                            try:
                                json.loads(text[start : end + 1])
                                return 0.3
                            except (json.JSONDecodeError, ValueError):
                                pass
                    return 0.0
                name = tool_call.get("name", "")
                args = tool_call.get("arguments")
                if args is None or not isinstance(args, dict):
                    return 0.6
                if name in TOOL_NAMES:
                    required = _get_required_args(name)
                    if required and all(k in args for k in required):
                        return 1.0
                return 0.8


            def tool_call_format_reward(completions: list, **kwargs: Any) -> list[float]:
                return [_score_format(_extract_text(c)) for c in completions]


            def tool_selection_reward(
                completions: list, expected_tools: list[list[str]] | None = None, **kwargs: Any,
            ) -> list[float]:
                if expected_tools is None:
                    return [0.0] * len(completions)
                scores: list[float] = []
                for i, completion in enumerate(completions):
                    text = _extract_text(completion)
                    tool_call = _extract_json(text)
                    if tool_call is None:
                        scores.append(0.0)
                        continue
                    name = tool_call.get("name", "")
                    if name not in TOOL_NAMES:
                        scores.append(0.0)
                        continue
                    expected = expected_tools[i] if i < len(expected_tools) else []
                    exp_list = expected if isinstance(expected, list) else [expected]
                    scores.append(1.0 if name in exp_list else 0.5)
                return scores


            def chain_completion_reward(
                completions: list,
                expected_tools: list[list[str]] | None = None,
                chain_length: list[int] | None = None,
                **kwargs: Any,
            ) -> list[float]:
                if expected_tools is None or chain_length is None:
                    return [0.0] * len(completions)
                scores: list[float] = []
                for i, completion in enumerate(completions):
                    text = _extract_text(completion)
                    cl = chain_length[i] if i < len(chain_length) else 0
                    expected = expected_tools[i] if i < len(expected_tools) else []
                    exp_list = expected if isinstance(expected, list) else [expected]
                    tool_call = _extract_json(text)
                    if cl == 0:
                        scores.append(1.0 if tool_call is None else 0.0)
                        continue
                    if tool_call is None:
                        scores.append(0.0)
                        continue
                    name = tool_call.get("name", "")
                    if name in exp_list:
                        scores.append(1.0 if cl == 1 else 0.6)
                    else:
                        scores.append(0.3 if any(n in text for n in exp_list) else 0.0)
                return scores


            def neuralese_v2_reward(completions: list, **kwargs: Any) -> list[float]:
                scores: list[float] = []
                for completion in completions:
                    text = _extract_text(completion)
                    if not text:
                        scores.append(0.0)
                        continue
                    total = len(text)
                    if total < 30:
                        scores.append(1.0)
                        continue
                    structured = 0
                    for m in _FENCED_BLOCK_RE.finditer(text):
                        structured += len(m.group(0))
                    remaining = _FENCED_BLOCK_RE.sub("", text)
                    i = 0
                    while i < len(remaining):
                        if remaining[i] == "{{":
                            depth, start = 0, i
                            for j in range(i, len(remaining)):
                                if remaining[j] == "{{":
                                    depth += 1
                                elif remaining[j] == "}}":
                                    depth -= 1
                                    if depth == 0:
                                        structured += j - start + 1
                                        i = j + 1
                                        break
                            else:
                                i += 1
                        else:
                            i += 1
                    density = (structured / total) * 2.0
                    scores.append(min(1.0, round(density, 4)))
                return scores


            def conciseness_reward(
                completions: list, max_completion_length: int = {config.max_length}, **kwargs: Any,
            ) -> list[float]:
                scores: list[float] = []
                for completion in completions:
                    text = _extract_text(completion)
                    score = max(0.0, 1.0 - (len(text) / max_completion_length))
                    scores.append(round(score, 4))
                return scores


            # ===================================================================
            # CascadeRewardManager (Nemotron Cascade 2)
            # ===================================================================

            class CascadeRewardManager:
                def __init__(self, carl_start: int = {carl_start}, warmup_steps: int = 10) -> None:
                    self.carl_start = carl_start
                    self.warmup_steps = warmup_steps
                    self._step: int = 0

                def get_stage(self) -> str:
                    return "B" if self._step >= self.carl_start else "A"

                def get_stage_weight(self, active_in_stages: set[str]) -> float:
                    stage = self.get_stage()
                    if stage not in active_in_stages:
                        return 0.0
                    warmup = self.warmup_steps
                    if stage == "B" and self._step < self.carl_start + warmup:
                        return (self._step - self.carl_start + 1) / warmup
                    return 1.0

                def wrap_reward(self, reward_fn, active_in_stages: set[str]):
                    manager = self
                    def wrapped(completions: list, **kwargs: Any) -> list[float]:
                        weight = manager.get_stage_weight(active_in_stages)
                        if weight <= 0.0:
                            return [0.0] * len(completions)
                        raw = reward_fn(completions, **kwargs)
                        if weight >= 1.0:
                            return raw
                        return [r * weight for r in raw]
                    wrapped.__name__ = getattr(reward_fn, "__name__", "wrapped_reward")
                    return wrapped


            # ===================================================================
            # CascadeCallback
            # ===================================================================

            class CascadeCallback(TrainerCallback):
                def __init__(self, cascade_manager: CascadeRewardManager) -> None:
                    self.cascade = cascade_manager

                def on_step_begin(self, args, state, control, **kwargs):
                    self.cascade._step = state.global_step

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs is not None:
                        stage = self.cascade.get_stage()
                        logs["cascade/stage"] = {{"A": 0, "B": 1}}[stage]
                        logs["cascade/step"] = state.global_step


            # ===================================================================
            # CoherenceMonitorCallback
            # ===================================================================

            class CoherenceMonitorCallback(TrainerCallback):
                def __init__(self, carl_reward_fn) -> None:
                    self.carl_fn = carl_reward_fn

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs is None:
                        return
                    if state.global_step <= 1:
                        logs["coherence/kappa"] = KAPPA
                        logs["coherence/sigma"] = SIGMA
                        logs["coherence/kappa_x_sigma"] = KAPPA * SIGMA
                    lock = getattr(self.carl_fn, "_metrics_lock", None)
                    metrics_ref = getattr(self.carl_fn, "_last_metrics", None)
                    if metrics_ref is None:
                        return
                    if lock is not None:
                        lock.acquire()
                        try:
                            data = metrics_ref[0]
                        finally:
                            lock.release()
                    else:
                        data = metrics_ref[0]
                    if data is None:
                        return
                    stored_step, batch_metrics = data
                    if abs(stored_step - state.global_step) > 1:
                        return
                    if not batch_metrics:
                        return
                    ms_vals = [m.get("multiscale", 0.0) for m in batch_metrics]
                    cq_vals = [m.get("cloud_quality", 0.0) for m in batch_metrics]
                    disc_vals = [m.get("discontinuity", 0.0) for m in batch_metrics]
                    logs["coherence/phi_mean"] = sum(ms_vals) / len(ms_vals)
                    logs["coherence/cloud_quality"] = sum(cq_vals) / len(cq_vals)
                    logs["coherence/discontinuity_density"] = sum(disc_vals) / len(disc_vals)
                    composite_vals = [
                        0.5 * m.get("multiscale", 0.0)
                        + 0.3 * m.get("cloud_quality", 0.0)
                        + 0.2 * m.get("discontinuity", 0.0)
                        for m in batch_metrics
                    ]
                    logs["coherence/cryst_to_melt_ratio"] = sum(composite_vals) / len(composite_vals)


            # ===================================================================
            # Main: Model loading, dataset, training
            # ===================================================================

            def main():
                hf_token = os.environ.get("HF_TOKEN")

                # --- Tokenizer ---
                tokenizer = AutoTokenizer.from_pretrained(
                    "{config.tokenizer_source or config.base_model}",
                    token=hf_token, trust_remote_code=True,
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # --- Model ---
                quant_config = BitsAndBytesConfig(load_in_8bit={config.quantization.load_in_8bit})
                model = AutoModelForCausalLM.from_pretrained(
                    "{config.base_model}",
                    quantization_config=quant_config,
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token,
                )

                # --- LoRA ---
                peft_config = LoraConfig(
                    r={config.lora.r}, lora_alpha={config.lora.alpha},
                    target_modules={config.lora.target_modules},
                    task_type="CAUSAL_LM", lora_dropout={config.lora.dropout},
                )

                # --- Dataset ---
                dataset = load_dataset("{config.dataset_repo}", split="{config.dataset_split}", token=hf_token)

                # --- Cascade Manager (2-stage: task from step 0, CARL at carl_start) ---
                cascade = CascadeRewardManager(carl_start={carl_start}, warmup_steps=10)

                # --- CARL reward ---
                carl_fn = make_carl_reward(
                    model, tokenizer, active_after_step=0,
                    max_length={config.max_length},
                )

                # --- Wrap rewards with cascade staging ---
                # R1 (format): active in all stages
                r1 = cascade.wrap_reward(tool_call_format_reward, {{"A", "B"}})
                # R2 (selection): active in all stages
                r2 = cascade.wrap_reward(tool_selection_reward, {{"A", "B"}})
                # R3 (chain): active in all stages
                r3 = cascade.wrap_reward(chain_completion_reward, {{"A", "B"}})
                # R4 (neuralese): active in all stages
                r4 = cascade.wrap_reward(neuralese_v2_reward, {{"A", "B"}})
                # R5 (conciseness): active in all stages
                r5 = cascade.wrap_reward(conciseness_reward, {{"A", "B"}})
                # R6 (CARL): active in B only (after carl_start)
                r6 = cascade.wrap_reward(carl_fn, {{"B"}})

                reward_funcs = [r1, r2, r3, r4, r5, r6]
                reward_weights = [2.0, 0.5, 2.0, 0.5, 0.5, 1.0]

                # --- GRPOConfig ---
                training_config = GRPOConfig(
                    output_dir="carl-grpo-output",
                    push_to_hub={config.push_to_hub},
                    hub_model_id="{config.output_repo}",
                    hub_strategy="{config.hub_strategy}",
                    hub_token=hf_token,
                    max_steps={total_steps},
                    per_device_train_batch_size={config.per_device_train_batch_size},
                    gradient_accumulation_steps={config.gradient_accumulation_steps},
                    learning_rate={config.learning_rate},
                    num_generations={config.num_generations},
                    beta={config.beta},
                    max_completion_length={config.max_length},
                    logging_steps=1,
                    logging_first_step=True,
                    save_strategy="steps",
                    save_steps=50,
                    save_total_limit=8,
                    scale_rewards="group",
                    mask_truncated_completions=True,
                    warmup_ratio={config.warmup_ratio},
                    lr_scheduler_type="{config.lr_scheduler_type}",
                    bf16={config.bf16},
                    gradient_checkpointing=True,
                    optim="adamw_8bit",
                    max_grad_norm={config.max_grad_norm},
                    weight_decay={config.weight_decay},
                    seed={config.seed},
                    report_to="trackio",
                )
                trackio.init(project="carl-studio", run_name="{config.run_name}")

                # --- Callbacks ---
                coherence_cb = CoherenceMonitorCallback(carl_fn)
                cascade_cb = CascadeCallback(cascade)

                # --- Trainer ---
                trainer = GRPOTrainer(
                    model=model,
                    processing_class=tokenizer,
                    train_dataset=dataset,
                    args=training_config,
                    peft_config=peft_config,
                    reward_funcs=reward_funcs,
                    reward_weights=reward_weights,
                    callbacks=[coherence_cb, cascade_cb],
                )

                trainer.train()
                trainer.push_to_hub()
                trackio.finish()
                print(f"Done! Model at: https://huggingface.co/{config.output_repo}")


            if __name__ == "__main__":
                main()
        ''')
