"""CARLTrainer -- orchestrates SFT and GRPO training runs.

Two modes:
  Remote (any non-LOCAL compute target):
    Generate self-contained script via Bundler, submit to HF Jobs backend,
    return job ID.  Training runs asynchronously.
  Local (LOCAL compute target):
    Run TRL training loop in-process with CARL rewards + cascade + callbacks.

All heavy imports (torch, transformers, trl, peft, datasets) are LAZY --
they live inside the methods that need them.  Module-level imports are
stdlib + carl_studio types only.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Optional

from carl_core.interaction import ActionType, InteractionChain
from carl_core.retry import RetryPolicy

from carl_studio.types.config import ComputeTarget, TrainingConfig, TrainingMethod
from carl_studio.types.run import RunPhase, TrainingRun

logger = logging.getLogger(__name__)


TRAINING_EXTRA_HINT = "Install training dependencies with: pip install 'carl-studio[training]'"

CARL_CHECKPOINT_FILE = ".carl_checkpoint.pt"


# ---------------------------------------------------------------------------
# Compute target -> backend mapping
# ---------------------------------------------------------------------------

_REMOTE_BACKEND = "hf_jobs"

_COMPUTE_TO_FLAVOR: dict[ComputeTarget, str] = {
    ComputeTarget.L4X1: "l4x1",
    ComputeTarget.L4X4: "l4x4",
    ComputeTarget.A10G_LARGE: "a10g-large",
    ComputeTarget.A10G_LARGEX2: "a10g-largex2",
    ComputeTarget.A10G_LARGEX4: "a10g-largex4",
    ComputeTarget.A100_LARGE: "a100-large",
    ComputeTarget.A100_LARGEX2: "a100-largex2",
    ComputeTarget.A100_LARGEX4: "a100-largex4",
    ComputeTarget.A100_LARGEX8: "a100-largex8",
    ComputeTarget.L40SX1: "l40sx1",
    ComputeTarget.L40SX4: "l40sx4",
    ComputeTarget.L40SX8: "l40sx8",
}


class CARLTrainer:
    """Orchestrates a single CARL training run (SFT or GRPO).

    Usage::

        from carl_studio import CARLTrainer, TrainingConfig
        trainer = CARLTrainer(config)
        run = await trainer.train()
        print(run.id, run.phase, run.hub_job_id)
    """

    def __init__(
        self,
        config: TrainingConfig,
        *,
        skip_credits: bool = False,
        resume_from_checkpoint: bool | str | None = None,
        interaction_chain: InteractionChain | None = None,
        training_step_interval: int = 100,
    ) -> None:
        self.config = config
        self.skip_credits = bool(skip_credits)
        self.resume_from_checkpoint = resume_from_checkpoint
        self.run = TrainingRun(
            id=uuid.uuid4().hex[:12],
            config=config,
            phase=RunPhase.INITIALIZING,
        )
        self._model: Any = None
        self._tokenizer: Any = None
        self._cascade_manager: Any = None
        self._reward_fns: list[Any] = []
        # --- InteractionChain plumbing -------------------------------------
        # When a chain is provided, the trainer emits TRAINING_STEP (every
        # ``training_step_interval`` global steps), CHECKPOINT (on save) and
        # a bracketing ``training.start``/``training.complete`` step pair.
        self.chain: InteractionChain | None = interaction_chain
        if training_step_interval < 1:
            raise ValueError(
                f"training_step_interval must be >= 1, got {training_step_interval}"
            )
        self._step_interval = int(training_step_interval)

    def _record(
        self,
        action: ActionType,
        name: str,
        *,
        input: Any = None,
        output: Any = None,
        success: bool = True,
    ) -> None:
        """Append a step to self.chain iff one is attached. Never raises."""
        if self.chain is None:
            return
        try:
            self.chain.record(
                action,
                name,
                input=input,
                output=output,
                success=success,
                session_id=self.run.id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("interaction chain record failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_remote(self) -> bool:
        """True when the compute target is anything other than LOCAL."""
        return self.config.compute_target != ComputeTarget.LOCAL

    async def train(self) -> TrainingRun:
        """Dispatch training based on compute_target.

        Returns the TrainingRun with updated phase and (for remote) hub_job_id.
        """
        try:
            if self.is_remote:
                return await self._train_remote()
            else:
                return await self._train_local()
        except Exception as exc:
            self.run.phase = RunPhase.FAILED
            self.run.error_message = f"{type(exc).__name__}: {exc}"
            logger.error("Training failed: %s", exc, exc_info=True)
            return self.run

    async def watch(
        self,
        poll_interval: float = 60.0,
        *,
        max_consecutive_failures: int = 5,
        poll_interval_s: float | None = None,
        timeout_s: float | None = None,
    ) -> TrainingRun:
        """Monitor a submitted job until completion. Returns final state.

        Status checks are retried with exponential backoff via
        carl_core.retry. After ``max_consecutive_failures`` consecutive
        retry-exhausted failures, aborts with CARLError(code="carl.watch_exhausted").
        """
        if not self.run.hub_job_id:
            raise ValueError("No job to watch — call train() first")

        interval = float(poll_interval_s) if poll_interval_s is not None else float(poll_interval)
        if interval <= 0:
            raise ValueError(f"poll interval must be > 0, got {interval}")
        if max_consecutive_failures < 1:
            raise ValueError(
                f"max_consecutive_failures must be >= 1, got {max_consecutive_failures}"
            )

        if timeout_s is None:
            try:
                timeout_s = float(self._parse_timeout(self.config.timeout))
            except Exception:
                timeout_s = 86400.0
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")

        from carl_studio.compute import get_backend

        backend = get_backend(_REMOTE_BACKEND)
        self.run.phase = RunPhase.TRAINING
        logger.info(
            "Watching job: %s (interval=%.1fs, timeout=%.0fs, max_failures=%d)",
            self.run.hub_job_id,
            interval,
            timeout_s,
            max_consecutive_failures,
        )

        return await self._watch_loop(
            backend=backend,
            interval_s=interval,
            timeout_s=timeout_s,
            max_consecutive_failures=max_consecutive_failures,
        )

    async def _watch_loop(
        self,
        *,
        backend: Any,
        interval_s: float,
        timeout_s: float,
        max_consecutive_failures: int,
    ) -> TrainingRun:
        """Inner watch loop — retry + backoff + consecutive-failure abort."""
        import asyncio

        from carl_core.errors import CARLError, CARLTimeoutError, NetworkError
        from carl_core.retry import async_retry

        policy = RetryPolicy(
            max_attempts=3,
            backoff_base=2.0,
            max_delay=30.0,
            retryable=(NetworkError, IOError, ConnectionError, TimeoutError),
        )

        consecutive_failures = 0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise CARLTimeoutError(
                    f"watch timed out after {timeout_s:.0f}s",
                    context={
                        "job_id": self.run.hub_job_id,
                        "timeout_s": timeout_s,
                        "interval_s": interval_s,
                    },
                )

            async def _check() -> str:
                return await backend.status(self.run.hub_job_id)

            try:
                status = await async_retry(_check, policy=policy)
            except CARLError:
                raise
            except Exception as exc:
                retryable = isinstance(exc, policy.retryable)
                consecutive_failures += 1
                logger.warning(
                    "Status check %s (%d/%d consecutive): %s",
                    "failed" if retryable else "raised non-retryable",
                    consecutive_failures,
                    max_consecutive_failures,
                    exc,
                )
                if consecutive_failures >= max_consecutive_failures:
                    raise CARLError(
                        f"watch exhausted after {consecutive_failures} consecutive failures",
                        code="carl.watch_exhausted",
                        context={
                            "job_id": self.run.hub_job_id,
                            "consecutive_failures": consecutive_failures,
                            "last_error_type": type(exc).__name__,
                            "last_error_message": str(exc),
                            "retryable": retryable,
                        },
                        cause=exc,
                    ) from exc
                await asyncio.sleep(min(interval_s, max(0.0, deadline - loop.time())))
                continue

            consecutive_failures = 0

            status_lower = (status or "").lower()
            if status_lower == "completed":
                self.run.phase = RunPhase.COMPLETE
                logger.info("Job completed: %s", self.run.hub_job_id)
                return self.run
            if status_lower in ("error", "failed"):
                self.run.phase = RunPhase.FAILED
                self.run.error_message = f"Remote job failed: {status}"
                logger.error("Job failed: %s", self.run.hub_job_id)
                return self.run
            if status_lower == "canceled":
                self.run.phase = RunPhase.FAILED
                self.run.error_message = "Job canceled"
                return self.run

            await asyncio.sleep(min(interval_s, max(0.0, deadline - loop.time())))

    async def train_and_watch(self, poll_interval: float = 60.0) -> TrainingRun:
        """Submit job and monitor until completion. The full lifecycle."""
        run = await self.train()
        if run.phase == RunPhase.FAILED:
            return run
        if self.is_remote and run.hub_job_id:
            return await self.watch(poll_interval=poll_interval)
        return run

    @staticmethod
    def check_checkpoint(model_id: str, token: str | None = None) -> bool:
        """Gate: does a checkpoint exist on Hub with actual weights?"""
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
            info = api.model_info(model_id, token=token)
            files = {s.rfilename for s in info.siblings}
            return "adapter_model.safetensors" in files or "model.safetensors" in files
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Remote mode
    # ------------------------------------------------------------------

    async def _train_remote(self) -> TrainingRun:
        """Generate bundled script, submit to remote backend, return job ID."""
        from carl_studio.bundler import Bundler
        from carl_studio.compute import get_backend

        self.run.phase = RunPhase.PROVISIONING
        logger.info(
            "Remote training: target=%s method=%s",
            self.config.compute_target.value,
            self.config.method.value,
        )

        deducted_amount, credit_jwt, credit_supabase_url = self._prededuct_credits()
        submitted = False

        try:
            bundler = Bundler()
            script = bundler.generate(self.config)

            backend = get_backend(_REMOTE_BACKEND)
            flavor = _COMPUTE_TO_FLAVOR.get(self.config.compute_target)
            if flavor is None:
                raise ValueError(
                    f"No hardware flavor mapping for compute target: {self.config.compute_target}"
                )

            timeout_seconds = self._parse_timeout(self.config.timeout)

            await backend.provision(hardware=flavor, timeout=timeout_seconds)
            job_id = await backend.execute(
                script=script,
                flavor=flavor,
                timeout=self.config.timeout,
                secrets={"HF_TOKEN": "HF_TOKEN"},
            )
            submitted = True

            self.run.hub_job_id = job_id
            self.run.phase = RunPhase.TRAINING
            self._record(
                ActionType.TRAINING_STEP,
                "training.submitted",
                input={
                    "flavor": flavor,
                    "method": self.config.method.value,
                    "run_id": self.run.id,
                },
                output={"hub_job_id": job_id},
            )
            logger.info("Remote job submitted: %s", job_id)
            return self.run
        except BaseException:
            if not submitted:
                self._refund_credits(
                    credit_jwt,
                    credit_supabase_url,
                    deducted_amount,
                    reason="job_failed_pre_submit",
                )
            else:
                self._refund_credits(
                    credit_jwt,
                    credit_supabase_url,
                    deducted_amount,
                    reason="job_failed_post_submit",
                )
            raise

    def _prededuct_credits(self) -> tuple[int, str | None, str | None]:
        """Synchronous credit pre-deduction for managed compute.

        Returns (deducted_amount, jwt, supabase_url). Raises
        CARLError(code="carl.credits_failed") when a deduction was required
        but failed — unless ``skip_credits`` was set on the trainer, in which
        case a warning is logged and (0, None, None) is returned.
        """
        from carl_core.errors import CARLError

        try:
            from carl_studio.camp import (
                DEFAULT_CARL_CAMP_SUPABASE_URL,
                load_camp_session,
            )
        except ImportError as exc:
            if self.skip_credits:
                logger.info("Credits module unavailable; --skip-credits honored")
                return 0, None, None
            raise CARLError(
                "credits module unavailable — install carl-studio[camp] or pass skip_credits=True",
                code="carl.credits_failed",
                context={"run_id": self.run.id},
                cause=exc,
            ) from exc

        try:
            session = load_camp_session()
        except Exception as exc:
            if self.skip_credits:
                logger.warning("camp session unavailable; --skip-credits honored: %s", exc)
                return 0, None, None
            raise CARLError(
                "failed to load camp session for credit deduction",
                code="carl.credits_failed",
                context={"run_id": self.run.id},
                cause=exc,
            ) from exc

        jwt = session.jwt
        if not jwt:
            logger.debug("No camp JWT; skipping credit pre-deduction (BYOK mode)")
            return 0, None, None

        try:
            from carl_studio.credits.balance import deduct_credits
            from carl_studio.credits.estimate import (
                METHOD_STEP_SECONDS,
                estimate_job_cost,
            )
        except ImportError as exc:
            if self.skip_credits:
                logger.info("Credits submodule unavailable; --skip-credits honored")
                return 0, None, None
            raise CARLError(
                "credits submodule unavailable — install carl-studio[camp]",
                code="carl.credits_failed",
                context={"run_id": self.run.id},
                cause=exc,
            ) from exc

        supabase_url = session.supabase_url or DEFAULT_CARL_CAMP_SUPABASE_URL
        per_step = METHOD_STEP_SECONDS.get(self.config.method.value, 20.0)
        try:
            estimate = estimate_job_cost(
                hardware=self.config.compute_target.value,
                max_steps=self.config.max_steps or 1000,
                per_step_seconds=per_step,
            )
        except Exception as exc:
            if self.skip_credits:
                logger.warning("cost estimate failed; --skip-credits honored: %s", exc)
                return 0, None, None
            raise CARLError(
                "failed to estimate credit cost",
                code="carl.credits_failed",
                context={
                    "run_id": self.run.id,
                    "hardware": self.config.compute_target.value,
                    "max_steps": self.config.max_steps,
                },
                cause=exc,
            ) from exc

        if estimate.total_with_buffer <= 0:
            logger.debug("Credit estimate is zero; no deduction needed")
            return 0, None, None

        try:
            ok = deduct_credits(
                jwt,
                supabase_url,
                estimate.total_with_buffer,
                self.run.id,
                "training_job",
            )
        except Exception as exc:
            if self.skip_credits:
                logger.warning(
                    "credit deduction failed but --skip-credits is set; proceeding: %s",
                    exc,
                )
                return 0, None, None
            raise CARLError(
                f"credit pre-deduction failed: {exc}",
                code="carl.credits_failed",
                context={
                    "run_id": self.run.id,
                    "amount": estimate.total_with_buffer,
                    "hardware": self.config.compute_target.value,
                },
                cause=exc,
            ) from exc

        if not ok:
            if self.skip_credits:
                logger.warning("credit deduction returned False; --skip-credits honored")
                return 0, None, None
            raise CARLError(
                "credit pre-deduction rejected by server",
                code="carl.credits_failed",
                context={
                    "run_id": self.run.id,
                    "amount": estimate.total_with_buffer,
                },
            )

        logger.info(
            "Pre-deducted %d credits for run %s",
            estimate.total_with_buffer,
            self.run.id,
        )
        return estimate.total_with_buffer, jwt, supabase_url

    def _try_prededuct_credits(self) -> tuple[int, str | None, str | None]:
        """Deprecated — use _prededuct_credits. Kept for API stability."""
        return self._prededuct_credits()

    def _refund_credits(
        self,
        jwt: str | None,
        supabase_url: str | None,
        amount: int,
        *,
        reason: str = "job_failed",
    ) -> None:
        """Best-effort refund. Safe to call with zero/None inputs (no-op).

        Refund failures are logged but never raised — we do not want to
        mask the original exception that triggered the refund.
        """
        if amount <= 0 or not jwt or not supabase_url:
            return
        try:
            from carl_studio.credits.balance import refund_credits

            refund_credits(jwt, supabase_url, amount, self.run.id, reason)
            logger.info(
                "Refunded %d credits for run %s (reason=%s)",
                amount,
                self.run.id,
                reason,
            )
        except Exception as exc:
            logger.warning(
                "Credit refund failed for run %s (reason=%s): %s",
                self.run.id,
                reason,
                exc,
            )

    def _try_refund_credits(
        self,
        jwt: str | None,
        supabase_url: str | None,
        amount: int,
    ) -> None:
        """Deprecated alias — use _refund_credits."""
        self._refund_credits(jwt, supabase_url, amount)

    # ------------------------------------------------------------------
    # Local mode
    # ------------------------------------------------------------------

    async def _train_local(self) -> TrainingRun:
        """Run TRL training loop in-process."""
        logger.info(
            "Local training: method=%s model=%s",
            self.config.method.value,
            self.config.base_model,
        )

        self._record(
            ActionType.TRAINING_STEP,
            "training.start",
            input={
                "method": self.config.method.value,
                "base_model": self.config.base_model,
                "run_id": self.run.id,
                "max_steps": self.config.max_steps,
            },
        )

        self.run.phase = RunPhase.LOADING_MODEL
        self._load_model_and_tokenizer()

        self.run.phase = RunPhase.TRAINING
        try:
            if self.config.method == TrainingMethod.SFT:
                await self._run_sft()
            elif self.config.method == TrainingMethod.GRPO:
                await self._run_grpo()
            else:
                raise ValueError(
                    f"Training method '{self.config.method.value}' is not yet implemented. "
                    "Supported methods: sft, grpo."
                )
        except BaseException as exc:
            self._record(
                ActionType.TRAINING_STEP,
                "training.failed",
                input={"method": self.config.method.value},
                output={"error": str(exc), "type": type(exc).__name__},
                success=False,
            )
            raise

        self.run.phase = RunPhase.COMPLETE
        self._record(
            ActionType.TRAINING_STEP,
            "training.complete",
            input={"method": self.config.method.value},
            output={"run_id": self.run.id},
        )
        logger.info("Local training complete: run_id=%s", self.run.id)
        return self.run

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model_and_tokenizer(self) -> None:
        """Load model + tokenizer with quantization.  Tries Unsloth first.

        VLM mode (config.vlm_mode=True): uses AutoModelForImageTextToText
        to activate the vision encoder. Do NOT pass tokenizer to TRL trainers
        in VLM mode — let them auto-load AutoProcessor.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(TRAINING_EXTRA_HINT) from exc

        cfg = self.config
        hf_token = self._get_hf_token()

        # Tokenizer (may use override source for missing chat template)
        tokenizer_source = cfg.tokenizer_source or cfg.base_model
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            token=hf_token,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Quantization config
        quant_kwargs: dict[str, Any] = {}
        if cfg.quantization.load_in_8bit or cfg.quantization.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=cfg.quantization.load_in_8bit,
                load_in_4bit=cfg.quantization.load_in_4bit,
                bnb_4bit_compute_dtype=getattr(
                    torch, cfg.quantization.bnb_4bit_compute_dtype, torch.bfloat16
                ),
                bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
            )
            quant_kwargs["quantization_config"] = bnb_config

        # VLM mode: use AutoModelForImageTextToText to activate vision encoder
        if cfg.vlm_mode:
            from transformers import AutoModelForImageTextToText

            self._model = AutoModelForImageTextToText.from_pretrained(
                cfg.base_model,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token,
                torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
                **quant_kwargs,
            )
            logger.info("Model loaded via AutoModelForImageTextToText (VLM mode)")
        else:
            # Try Unsloth for memory-efficient loading
            try:
                from unsloth import FastLanguageModel

                unsloth_kwargs: dict[str, Any] = {}
                if cfg.fast_inference:
                    unsloth_kwargs["fast_inference"] = True
                    unsloth_kwargs["gpu_memory_utilization"] = 0.6

                self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                    model_name=cfg.base_model,
                    max_seq_length=cfg.max_length,
                    dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
                    load_in_4bit=cfg.quantization.load_in_4bit,
                    load_in_8bit=cfg.quantization.load_in_8bit,
                    token=hf_token,
                    **unsloth_kwargs,
                )
                logger.info("Model loaded via Unsloth FastLanguageModel")
            except ImportError:
                self._model = AutoModelForCausalLM.from_pretrained(
                    cfg.base_model,
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token,
                    torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
                    **quant_kwargs,
                )
                logger.info("Model loaded via AutoModelForCausalLM (Unsloth not available)")

        # Suppress Qwen3.5 thinking mode if requested — model generates
        # <think>...</think> prefix by default, burning completion budget.
        if cfg.disable_thinking and hasattr(self._model, "generation_config"):
            gen_config = self._model.generation_config
            if hasattr(gen_config, "enable_thinking"):
                gen_config.enable_thinking = False
                logger.info("Disabled Qwen3.5 thinking mode (enable_thinking=False)")
            gen_config.do_sample = True

    # ------------------------------------------------------------------
    # Checkpoint helpers (WS-T1)
    # ------------------------------------------------------------------

    def _resolve_resume_arg(self, output_dir: str) -> bool | str | None:
        """Resolve the resume_from_checkpoint argument for trainer.train()."""
        if self.resume_from_checkpoint in (False, None):
            return None
        return self.resume_from_checkpoint

    def _announce_existing_checkpoint(self, output_dir: str) -> None:
        """Log a warning if a previous crash-checkpoint exists."""
        try:
            from pathlib import Path

            ckpt_path = Path(output_dir) / CARL_CHECKPOINT_FILE
            if not ckpt_path.exists():
                return
        except Exception:
            return

        try:
            import torch

            payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.warning(
                "Found existing %s but could not parse: %s",
                CARL_CHECKPOINT_FILE,
                exc,
            )
            return

        step = payload.get("step") if isinstance(payload, dict) else None
        exc_type = payload.get("exception_type") if isinstance(payload, dict) else None
        exc_msg = payload.get("exception_message") if isinstance(payload, dict) else None
        saved_at = payload.get("saved_at") if isinstance(payload, dict) else None
        logger.warning(
            "Prior crash-checkpoint found in %s: step=%s exception=%s msg=%r saved_at=%s "
            "— pass --resume-from-checkpoint to resume, or delete the file to start fresh.",
            output_dir,
            step,
            exc_type,
            exc_msg,
            saved_at,
        )

    @staticmethod
    def _save_carl_checkpoint(trainer: Any, output_dir: str, exc: BaseException) -> None:
        """Persist a crash-checkpoint beside the HF Trainer checkpoints.

        Writes ``output_dir/.carl_checkpoint.pt`` with step, optimizer and
        scheduler state dicts, RNG states, recent metrics, exception_type,
        exception_message, saved_at.

        Guarantee: any failure inside this function is logged but NEVER
        raised — masking the caller's original exception would be worse
        than losing the crash-state snapshot.
        """
        def _safe_capture(label: str, fn: Any) -> Any:
            """Call fn() and return its result; log and return None on any failure."""
            try:
                return fn()
            except Exception as cap_exc:
                logger.debug("checkpoint: could not capture %s: %s", label, cap_exc)
                return None

        def _numpy_rng() -> Any:
            import numpy as _np

            return _np.random.get_state()

        try:
            import random as _random
            from datetime import datetime, timezone
            from pathlib import Path

            import torch

            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = out_dir / CARL_CHECKPOINT_FILE

            step: int | None = None
            metrics: list[Any] = []
            state = getattr(trainer, "state", None)
            if state is not None:
                step = getattr(state, "global_step", None)
                log_history = getattr(state, "log_history", None)
                if isinstance(log_history, list):
                    metrics = list(log_history[-50:])

            payload = {
                "step": step,
                "optimizer_state_dict": _safe_capture(
                    "optimizer", lambda: trainer.optimizer.state_dict()
                ),
                "scheduler_state_dict": _safe_capture(
                    "scheduler", lambda: trainer.lr_scheduler.state_dict()
                ),
                "rng_state": {
                    "torch": _safe_capture("torch_rng", lambda: torch.get_rng_state()),
                    "python": _safe_capture("python_rng", lambda: _random.getstate()),
                    "numpy": _safe_capture("numpy_rng", _numpy_rng),
                },
                "metrics": metrics,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }

            torch.save(payload, str(ckpt_path))
            logger.warning(
                "Crash-checkpoint saved: %s (step=%s exception=%s)",
                ckpt_path,
                step,
                type(exc).__name__,
            )
        except Exception as save_exc:
            logger.error(
                "Failed to save crash-checkpoint to %s: %s",
                output_dir,
                save_exc,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # SFT training
    # ------------------------------------------------------------------

    async def _run_sft(self) -> None:
        """Run SFT training with TRL SFTTrainer."""
        try:
            from datasets import load_dataset
            from peft import LoraConfig as PeftLoraConfig
            from trl import SFTConfig, SFTTrainer
        except ImportError as exc:
            raise ImportError(TRAINING_EXTRA_HINT) from exc

        cfg = self.config
        hf_token = self._get_hf_token()

        dataset = load_dataset(cfg.dataset_repo, split=cfg.dataset_split, token=hf_token)

        eval_dataset = None
        if cfg.eval_dataset_repo:
            eval_dataset = load_dataset(cfg.eval_dataset_repo, split=cfg.eval_split, token=hf_token)

        peft_config = PeftLoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            target_modules=cfg.lora.target_modules,
            lora_dropout=cfg.lora.dropout,
            bias=cfg.lora.bias,
            task_type="CAUSAL_LM",
        )

        output_dir = f"carl-sft-{self.run.id}"
        self._announce_existing_checkpoint(output_dir)

        training_args = SFTConfig(
            output_dir=output_dir,
            push_to_hub=cfg.push_to_hub,
            hub_model_id=cfg.output_repo,
            hub_strategy=cfg.hub_strategy,
            hub_token=hf_token,
            hub_private_repo=cfg.hub_private,
            num_train_epochs=cfg.num_train_epochs,
            max_steps=cfg.max_steps,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            max_length=cfg.max_length,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
            lr_scheduler_type=cfg.lr_scheduler_type,
            bf16=cfg.bf16,
            seed=cfg.seed,
            gradient_checkpointing=True,
            optim="adamw_8bit",
            logging_steps=1,
            logging_first_step=True,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=4,
            report_to="none",
            run_name=cfg.run_name,
        )

        trainer = SFTTrainer(
            model=self._model,
            processing_class=self._tokenizer,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            peft_config=peft_config,
        )

        resume = self._resolve_resume_arg(output_dir)
        try:
            if resume is None:
                trainer.train()
            else:
                trainer.train(resume_from_checkpoint=resume)
        except (Exception, KeyboardInterrupt) as exc:
            self._save_carl_checkpoint(trainer, output_dir, exc)
            self._record(
                ActionType.CHECKPOINT,
                "sft.crash_checkpoint",
                input={"output_dir": output_dir},
                output={"exception_type": type(exc).__name__},
                success=False,
            )
            raise

        self._record(
            ActionType.CHECKPOINT,
            "sft.save",
            input={"output_dir": output_dir},
            output={"run_id": self.run.id},
        )

        if cfg.push_to_hub:
            self.run.phase = RunPhase.PUSHING
            trainer.push_to_hub()

    # ------------------------------------------------------------------
    # GRPO training
    # ------------------------------------------------------------------

    async def _run_grpo(self) -> None:
        """Run GRPO training with CARL rewards + cascade + callbacks."""
        # Apply Unsloth optimizations for GRPO if available
        try:
            from unsloth import PatchFastRL
            from unsloth import FastLanguageModel as _FLM

            PatchFastRL(algorithm="grpo", FastLanguageModel=_FLM)
            logger.info("PatchFastRL applied for GRPO (Unsloth vLLM-backed generation)")
        except ImportError:
            pass  # Unsloth not installed; standard TRL path

        try:
            from datasets import load_dataset
            from peft import LoraConfig as PeftLoraConfig
            from trl import GRPOConfig, GRPOTrainer
        except ImportError as exc:
            raise ImportError(TRAINING_EXTRA_HINT) from exc

        cfg = self.config
        hf_token = self._get_hf_token()

        dataset = load_dataset(cfg.dataset_repo, split=cfg.dataset_split, token=hf_token)

        eval_dataset = None
        if cfg.eval_dataset_repo:
            eval_dataset = load_dataset(cfg.eval_dataset_repo, split=cfg.eval_split, token=hf_token)

        # Build reward chain with cascade wrapping
        self._reward_fns = self._build_rewards(self._model, self._tokenizer)

        # Build callbacks (must be after _build_rewards so _cascade_manager exists)
        callbacks = self._build_callbacks()

        peft_config = PeftLoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            target_modules=cfg.lora.target_modules,
            lora_dropout=cfg.lora.dropout,
            bias=cfg.lora.bias,
            task_type="CAUSAL_LM",
        )

        output_dir = f"carl-grpo-{self.run.id}"
        self._announce_existing_checkpoint(output_dir)

        training_args = GRPOConfig(
            output_dir=output_dir,
            push_to_hub=cfg.push_to_hub,
            hub_model_id=cfg.output_repo,
            hub_strategy=cfg.hub_strategy,
            hub_token=hf_token,
            hub_private_repo=cfg.hub_private,
            num_train_epochs=cfg.num_train_epochs,
            max_steps=cfg.max_steps,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
            lr_scheduler_type=cfg.lr_scheduler_type,
            bf16=cfg.bf16,
            seed=cfg.seed,
            num_generations=cfg.num_generations,
            max_completion_length=cfg.max_completion_length,
            beta=cfg.beta,
            gradient_checkpointing=True,
            optim="adamw_8bit",
            logging_steps=1,
            logging_first_step=True,
            save_strategy="steps",
            save_steps=25,
            save_total_limit=8,
            report_to="none",
            run_name=cfg.run_name,
        )

        trainer = GRPOTrainer(
            model=self._model,
            processing_class=self._tokenizer,
            reward_funcs=self._reward_fns,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            peft_config=peft_config,
            callbacks=callbacks,
        )

        resume = self._resolve_resume_arg(output_dir)
        try:
            if resume is None:
                trainer.train()
            else:
                trainer.train(resume_from_checkpoint=resume)
        except (Exception, KeyboardInterrupt) as exc:
            self._save_carl_checkpoint(trainer, output_dir, exc)
            self._record(
                ActionType.CHECKPOINT,
                "grpo.crash_checkpoint",
                input={"output_dir": output_dir},
                output={"exception_type": type(exc).__name__},
                success=False,
            )
            raise

        self._record(
            ActionType.CHECKPOINT,
            "grpo.save",
            input={"output_dir": output_dir},
            output={"run_id": self.run.id},
        )

        if cfg.push_to_hub:
            self.run.phase = RunPhase.PUSHING
            trainer.push_to_hub()

    # ------------------------------------------------------------------
    # Reward construction
    # ------------------------------------------------------------------

    def _build_rewards(self, model: Any, tokenizer: Any) -> list[Any]:
        """Build 6 reward functions: 5 task + 1 CARL, all cascade-wrapped.

        Two-stage cascade (matches CascadeRewardManager API):
          Stage A: task rewards only (before carl_start)
          Stage B: task + CARL (after carl_start, with warmup)

        Reward chain:
          R1: tool_call_format_reward   (w=2.0, stages A/B)
          R2: tool_selection_reward     (w=1.5, stages A/B)
          R3: chain_completion_reward   (w=2.0, stages A/B)
          R4: neuralese_v2_reward       (w=0.5, stages A/B)
          R5: conciseness_reward        (w=0.5, stages A/B)
          R6: carl_composite_reward     (w=1.5, stage B only)
        """
        from carl_studio.training.cascade import CascadeRewardManager
        from carl_studio.training.rewards.composite import make_carl_reward

        # Resolve CARL activation step from cascade config or default
        carl_start = 50
        if self.config.cascade_stages and len(self.config.cascade_stages) >= 2:
            carl_start = sum(s.steps for s in self.config.cascade_stages[:1])

        cascade = CascadeRewardManager(
            carl_start=carl_start,
            warmup_steps=10,
        )
        self._cascade_manager = cascade

        # Load task reward functions (with fallback to no-ops)
        task_rewards = self._load_task_rewards()

        # CARL reward (needs model + tokenizer for forward pass)
        carl_fn = make_carl_reward(
            model=model,
            tokenizer=tokenizer,
            vocab_size=getattr(tokenizer, "vocab_size", 128000),
            max_length=self.config.max_length,
        )

        # (reward_fn, active_in_stages, weight)
        reward_specs: list[tuple[Any, set[str], float]] = [
            (task_rewards["tool_call_format"], {"A", "B"}, 2.0),
            (task_rewards["tool_selection"], {"A", "B"}, 1.5),
            (task_rewards["chain_completion"], {"A", "B"}, 2.0),
            (task_rewards["neuralese_v2"], {"A", "B"}, 0.5),
            (task_rewards["conciseness"], {"A", "B"}, 0.5),
            (carl_fn, {"B"}, 1.5),
        ]

        wrapped: list[Any] = []
        for fn, stages, weight in reward_specs:
            cascade_fn = cascade.wrap_reward(fn, active_in_stages=stages)
            weighted_fn = _apply_weight(cascade_fn, weight)
            wrapped.append(weighted_fn)

        return wrapped

    def _load_task_rewards(self) -> dict[str, Any]:
        """Load task reward functions from carl_studio.training.rewards.

        Falls back to no-op rewards if task-specific modules are not available.
        """
        rewards: dict[str, Any] = {}

        try:
            from carl_studio.training.rewards import (
                tool_call_format_reward,
                tool_selection_reward,
                chain_completion_reward,
                neuralese_v2_reward,
                conciseness_reward,
            )

            rewards["tool_call_format"] = tool_call_format_reward
            rewards["tool_selection"] = tool_selection_reward
            rewards["chain_completion"] = chain_completion_reward
            rewards["neuralese_v2"] = neuralese_v2_reward
            rewards["conciseness"] = conciseness_reward
        except ImportError:
            logger.warning(
                "Task reward functions not found in carl_studio.training.rewards; "
                "using no-op placeholders.  Only CARL reward will be active."
            )
            for name in (
                "tool_call_format",
                "tool_selection",
                "chain_completion",
                "neuralese_v2",
                "conciseness",
            ):
                rewards[name] = _noop_reward
        return rewards

    # ------------------------------------------------------------------
    # Callback construction
    # ------------------------------------------------------------------

    def _build_callbacks(self) -> list[Any]:
        """Build training callbacks: CascadeCallback + CoherenceMonitorCallback."""
        from carl_studio.training.callbacks import (
            CoherenceMonitorCallback,
            InteractionChainCallback,
        )
        from carl_studio.training.cascade import CascadeCallback

        callbacks: list[Any] = []

        # Cascade stage tracking
        if self._cascade_manager is not None:
            callbacks.append(CascadeCallback(self._cascade_manager))

        # Coherence monitoring -- find the CARL reward fn (carries _last_metrics)
        carl_fn = self._find_carl_reward_fn()
        if carl_fn is not None:
            callbacks.append(
                CoherenceMonitorCallback(
                    carl_fn,
                    chain=self.chain,
                    session_id=self.run.id,
                )
            )
        else:
            logger.warning("Could not locate CARL reward function for coherence monitoring")

        # Emit TRAINING_STEP + CHECKPOINT into InteractionChain when attached.
        if self.chain is not None:
            callbacks.append(
                InteractionChainCallback(
                    chain=self.chain,
                    run_id=self.run.id,
                    step_interval=self._step_interval,
                )
            )

        return callbacks

    def _find_carl_reward_fn(self) -> Optional[Any]:
        """Find the reward function that carries _last_metrics (the CARL reward).

        _apply_weight propagates _last_metrics through weight wrappers,
        so we scan in reverse (CARL is typically last).
        """
        for fn in reversed(self._reward_fns):
            if hasattr(fn, "_last_metrics"):
                return fn
        return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_timeout(timeout_str: str) -> int:
        """Parse timeout string to seconds.

        Accepts:
          "3h"    -> 10800
          "90m"   -> 5400
          "3600"  -> 3600   (bare seconds)
          "2h30m" -> 9000   (compound)
        """
        if not timeout_str or not timeout_str.strip():
            return 10800  # default 3h

        remaining = timeout_str.strip()
        total = 0

        # Hours component
        h_match = re.search(r"(\d+(?:\.\d+)?)\s*h", remaining, re.IGNORECASE)
        if h_match:
            total += int(float(h_match.group(1)) * 3600)
            remaining = remaining[: h_match.start()] + remaining[h_match.end() :]

        # Minutes component
        m_match = re.search(r"(\d+(?:\.\d+)?)\s*m", remaining, re.IGNORECASE)
        if m_match:
            total += int(float(m_match.group(1)) * 60)
            remaining = remaining[: m_match.start()] + remaining[m_match.end() :]

        # Seconds component (explicit 's' suffix or bare digits)
        s_match = re.search(r"(\d+)\s*s?", remaining.strip())
        if s_match and s_match.group(1):
            val = int(s_match.group(1))
            if total == 0:
                # Bare number with no h/m prefix -- treat as seconds
                total = val
            else:
                total += val

        if total <= 0:
            raise ValueError(
                f"Cannot parse timeout: {timeout_str!r}.  Use '3h', '90m', '2h30m', or '14400'."
            )
        return total

    @staticmethod
    def _get_hf_token() -> str | None:
        """Resolve HuggingFace token — prefer hub credential store, fall back to env."""
        try:
            from huggingface_hub import get_token

            token = get_token()
            if token:
                return token
        except ImportError:
            pass
        import os

        return os.environ.get("HF_TOKEN")


# ---------------------------------------------------------------------------
# Module-level helpers (no heavy imports)
# ---------------------------------------------------------------------------


def _apply_weight(reward_fn: Any, weight: float) -> Any:
    """Wrap a reward function to multiply outputs by a static weight.

    Propagates CARL metadata attributes (_last_metrics, _metrics_lock, _step)
    so that CoherenceMonitorCallback can find them through the wrapper.
    """
    if weight == 1.0:
        return reward_fn

    def weighted(completions: list, **kwargs: Any) -> list[float]:
        raw = reward_fn(completions, **kwargs)
        return [r * weight for r in raw]

    weighted.__name__ = getattr(reward_fn, "__name__", "weighted_reward")

    # Propagate CARL metadata (including _last_traces for CoherenceTraceCallback)
    for attr in ("_last_metrics", "_last_traces", "_last_components", "_metrics_lock", "_step"):
        if hasattr(reward_fn, attr):
            setattr(weighted, attr, getattr(reward_fn, attr))

    return weighted


def _noop_reward(completions: list, **kwargs: Any) -> list[float]:
    """No-op reward function returning zeros."""
    return [0.0] * len(completions)


_noop_reward.__name__ = "noop_reward"
