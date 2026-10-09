"""CoherenceMonitorCallback -- logs crystal dynamics metrics per training step.

Ported from CrystalMonitorCallback with fixes:
  L2    -- thread-safe read of _last_metrics via threading.Lock
  A4    -- stale metrics detection via step comparison
  WS-T4 -- empty-batch guard on every len(X) division
  WS-T6 -- `with lock:` context manager instead of manual acquire/release;
           callback-level try/except isolation so a failing callback cannot
           crash the training loop.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

try:
    from transformers import TrainerCallback
except Exception:

    class TrainerCallback:  # type: ignore[no-redef]
        """Fallback base when transformers is unavailable."""

        pass


from carl_core.constants import KAPPA, SIGMA
from carl_core.interaction import ActionType, InteractionChain

from carl_studio.training.rewards.multiscale import clamp_counts, reset_clamp_counts


logger = logging.getLogger(__name__)


class ParameterUpdateCallback(TrainerCallback):
    """Retain initial and final trainable LoRA tensor digests."""

    def __init__(self, run: Any) -> None:
        self.run = run
        self._before: dict[str, dict[str, Any]] | None = None
        self._start_step = 0

    @staticmethod
    def _snapshot(model: Any) -> dict[str, dict[str, Any]]:
        import torch
        from carl_core.hashing import content_hash_bytes

        values: dict[str, dict[str, Any]] = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if not any(part.startswith("lora_") for part in name.split(".")) and (
                ".modules_to_save." not in name
            ):
                continue
            if not bool(torch.isfinite(parameter).all()):
                raise ValueError("Nonfinite trainable parameter")
            if parameter.numel() == 0:
                raise ValueError("Empty trainable parameter")
            raw = (
                parameter.detach()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )
            values[name] = {
                "dtype": str(parameter.dtype),
                "shape": list(parameter.shape),
                "elements": int(parameter.numel()),
                "sha256": content_hash_bytes(raw),
            }
        if not values:
            raise ValueError("Training has no trainable parameters")
        return values

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.is_world_process_zero:
            self._before = self._snapshot(kwargs["model"])
            self._start_step = int(state.global_step)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if not state.is_world_process_zero:
            return
        if self._before is None:
            raise ValueError("Initial parameter-update witness unavailable")
        after = self._snapshot(kwargs["model"])
        if set(after) != set(self._before) or any(
            (after[name]["dtype"], after[name]["shape"]) != (before["dtype"], before["shape"])
            for name, before in self._before.items()
        ):
            raise ValueError("Trainable parameter identities changed during training")
        changed = sorted(
            name for name in after if after[name]["sha256"] != self._before[name]["sha256"]
        )
        modules = sorted({name.rpartition(".")[0] or name for name in changed})
        import os
        import tempfile
        from pathlib import Path

        from carl_core.hashing import canonical_json, content_hash, content_hash_bytes

        from carl_studio.experiment.types import Artifact

        evidence = {
            "schema": "carl.parameter-updates/v1",
            "comparison": "retained_trainable_lora_parameters",
            "run_id": self.run.id,
            "start_step": self._start_step,
            "end_step": int(state.global_step),
            "before": self._before,
            "after": after,
            "before_sha256": content_hash(self._before),
            "after_sha256": content_hash(after),
            "changed_parameters": changed,
            "changed_modules": modules,
        }
        payload = (canonical_json(evidence) + "\n").encode()
        checksum = content_hash_bytes(payload)
        directory = Path(args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"parameter-updates-{checksum}.json"
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Parameter-update artifact bytes changed")
        descriptor, name = tempfile.mkstemp(prefix=".parameter-updates-", dir=directory)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        self.run.resource_usage.update(
            {
                "trainable_parameter_tensors": float(len(after)),
                "updated_parameter_tensors": float(len(changed)),
                "updated_parameter_modules": float(len(modules)),
            }
        )
        self.run.artifacts.append(
            Artifact(
                name="parameter_updates",
                path=str(path.resolve()),
                artifact_type="report",
                produced_at=self.run.id,
                checksum=checksum,
            )
        )


def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    """Mean that never divides by zero.

    Returns ``default`` for empty or non-iterable inputs, and silently skips
    non-numeric values. Used everywhere a callback aggregates across a batch.
    """
    try:
        vals = [float(v) for v in values]
    except (TypeError, ValueError):
        return default
    if not vals:
        return default
    return sum(vals) / len(vals)


class CoherenceMonitorCallback(TrainerCallback):
    """Logs crystal dynamics metrics every training step.

    Reads _last_metrics from the CARL reward function closure in a
    thread-safe manner (fix L2) and skips stale data (fix A4).

    Logged keys:
      coherence/phi_mean            -- mean order parameter
      coherence/cloud_quality       -- mean cloud quality
      coherence/discontinuity_density -- alias for defect density (not yet computed here)
      coherence/cryst_to_melt_ratio -- crystallizations / max(meltings, 1)
      coherence/coherence_scale_{j} -- per-scale coherence (first 8 scales)
      coherence/reward_clamp_total / _nonfinite / _overflow
                                    -- per-period clamp-telemetry counts
    """

    def __init__(
        self,
        carl_reward_fn: Any,
        *,
        chain: InteractionChain | None = None,
        session_id: str | None = None,
    ) -> None:
        self.carl_fn = carl_reward_fn
        self.chain = chain
        self.session_id = session_id

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            self._on_log_impl(args, state, control, logs, **kwargs)
        except Exception as exc:
            # WS-T6: never propagate — training loop must survive buggy callback.
            logger.warning(
                "callback %s failed: %s", type(self).__name__, exc
            )

    def on_epoch_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        try:
            # Emit clamp telemetry summary and reset the counters at the epoch boundary.
            counts = clamp_counts()
            if counts.get("total", 0):
                logger.info(
                    "coherence reward clamps this epoch: total=%d nonfinite=%d overflow=%d",
                    counts.get("total", 0),
                    counts.get("nonfinite", 0),
                    counts.get("overflow", 0),
                )

            # Record REWARD step with aggregated clamp counts on the attached chain.
            if self.chain is not None:
                epoch = getattr(state, "epoch", None)
                global_step = getattr(state, "global_step", None)
                try:
                    self.chain.record(
                        ActionType.REWARD,
                        "coherence.epoch_end",
                        input={
                            "epoch": epoch,
                            "global_step": global_step,
                        },
                        output={
                            "clamp_total": int(counts.get("total", 0)),
                            "clamp_nonfinite": int(counts.get("nonfinite", 0)),
                            "clamp_overflow": int(counts.get("overflow", 0)),
                        },
                        session_id=self.session_id,
                    )
                except Exception as chain_exc:  # pragma: no cover - defensive
                    logger.debug("REWARD chain record failed: %s", chain_exc)

            reset_clamp_counts()
        except Exception as exc:
            logger.warning(
                "callback %s.on_epoch_end failed: %s", type(self).__name__, exc
            )

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        # Hook retained for symmetry; all per-step telemetry currently emits
        # through on_log. Wrapped in try/except so subclasses or future logic
        # cannot break the training loop.
        try:
            return None
        except Exception as exc:
            logger.warning(
                "callback %s.on_step_end failed: %s", type(self).__name__, exc
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_log_impl(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return

        # Log crystal constants on first step
        if state.global_step <= 1:
            logs["coherence/kappa"] = KAPPA
            logs["coherence/sigma"] = SIGMA
            logs["coherence/kappa_x_sigma"] = KAPPA * SIGMA

        # Fix L2 / WS-T6: thread-safe read using `with lock:`
        lock = getattr(self.carl_fn, "_metrics_lock", None)
        metrics_ref = getattr(self.carl_fn, "_last_metrics", None)
        if metrics_ref is None:
            return

        if lock is not None:
            with lock:
                data = metrics_ref[0]
        else:
            data = metrics_ref[0]

        if data is None:
            return

        stored_step, batch_metrics = data

        # Fix A4: stale metrics detection via step comparison
        if abs(stored_step - state.global_step) > 1:
            return  # stale, skip

        if not batch_metrics:
            return

        # batch_metrics is a list of component dicts. Two supported schemas:
        #   new    -- {"multiscale", "cloud_quality", "discontinuity"} from CARLReward.score()
        #   legacy -- {"phi_mean", "cloud_quality_mean", ...} from _logits_to_crystal_metrics
        sample = batch_metrics[0]
        is_legacy = "phi_mean" in sample

        if is_legacy:
            phi_means = [m.get("phi_mean", 0.0) for m in batch_metrics]
            cq_means = [m.get("cloud_quality_mean", 0.0) for m in batch_metrics]
            dd_means = [m.get("defect_density", 0.0) for m in batch_metrics]
            cryst = sum(m.get("n_crystallizations", 0) for m in batch_metrics)
            melts = sum(m.get("n_meltings", 0) for m in batch_metrics)

            # WS-T4: every aggregator uses _safe_mean so empty batches never
            # ZeroDivisionError.
            logs["coherence/phi_mean"] = _safe_mean(phi_means)
            logs["coherence/cloud_quality"] = _safe_mean(cq_means)
            logs["coherence/discontinuity_density"] = _safe_mean(dd_means)
            logs["coherence/cryst_to_melt_ratio"] = cryst / max(melts, 1)

            # Per-scale coherence
            all_scales: set[int] = set()
            for m in batch_metrics:
                sc = m.get("scale_coherence", {}) or {}
                all_scales.update(sc.keys())
            for j in sorted(all_scales)[:8]:
                vals = [
                    (m.get("scale_coherence") or {}).get(j, 0.0)
                    for m in batch_metrics
                ]
                logs[f"coherence/coherence_scale_{j}"] = _safe_mean(vals)
        else:
            ms_vals = [m.get("multiscale", 0.0) for m in batch_metrics]
            cq_vals = [m.get("cloud_quality", 0.0) for m in batch_metrics]
            disc_vals = [m.get("discontinuity", 0.0) for m in batch_metrics]

            logs["coherence/phi_mean"] = _safe_mean(ms_vals)
            logs["coherence/cloud_quality"] = _safe_mean(cq_vals)
            logs["coherence/discontinuity_density"] = _safe_mean(disc_vals)
            composite_vals = [
                0.5 * m.get("multiscale", 0.0)
                + 0.3 * m.get("cloud_quality", 0.0)
                + 0.2 * m.get("discontinuity", 0.0)
                for m in batch_metrics
            ]
            logs["coherence/cryst_to_melt_ratio"] = _safe_mean(composite_vals)

        # Clamp-telemetry: always emit current running counts for observability.
        counts = clamp_counts()
        logs["coherence/reward_clamp_total"] = counts.get("total", 0)
        logs["coherence/reward_clamp_nonfinite"] = counts.get("nonfinite", 0)
        logs["coherence/reward_clamp_overflow"] = counts.get("overflow", 0)


class InteractionChainCallback(TrainerCallback):
    """Forwards HF Trainer events into an :class:`InteractionChain`.

    Emits:

    - ``TRAINING_STEP`` every ``step_interval`` global steps (or whenever the
      HF Trainer emits a log record, whichever is coarser).
    - ``CHECKPOINT`` on every ``on_save`` event.
    - ``TRAINING_STEP`` at ``on_epoch_end`` with the epoch index.

    Never raises: the training loop must survive a buggy or disconnected
    chain. Per-callback exceptions go to the module logger at WARNING.
    """

    def __init__(
        self,
        chain: InteractionChain,
        *,
        run_id: str | None = None,
        step_interval: int = 100,
    ) -> None:
        if chain is None:
            raise ValueError("InteractionChainCallback requires a non-None chain")
        if step_interval < 1:
            raise ValueError(f"step_interval must be >= 1, got {step_interval}")
        self.chain = chain
        self.run_id = run_id
        self.step_interval = int(step_interval)
        self._last_logged_step: int = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_step(
        self,
        name: str,
        *,
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        action: ActionType = ActionType.TRAINING_STEP,
        success: bool = True,
    ) -> None:
        try:
            self.chain.record(
                action,
                name,
                input=input,
                output=output,
                success=success,
                session_id=self.run_id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("chain record failed: %s", exc)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_train_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        try:
            self._record_step(
                "train.begin",
                input={"run_id": self.run_id},
                output={"max_steps": getattr(state, "max_steps", None)},
            )
        except Exception as exc:
            logger.warning(
                "callback %s.on_train_begin failed: %s", type(self).__name__, exc
            )

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            if logs is None:
                return
            global_step = int(getattr(state, "global_step", 0) or 0)
            if global_step <= 0:
                return
            if global_step - self._last_logged_step < self.step_interval:
                return
            self._last_logged_step = global_step

            # Capture a compact subset of metrics — don't flood the chain.
            metrics = {
                k: float(v)
                for k, v in logs.items()
                if isinstance(v, (int, float)) and not _is_skipped_key(k)
            }
            loss = metrics.get("loss")
            lr = metrics.get("learning_rate")
            self._record_step(
                "train.step",
                input={
                    "global_step": global_step,
                    "epoch": getattr(state, "epoch", None),
                },
                output={
                    "loss": loss,
                    "learning_rate": lr,
                    "metrics_keys": sorted(metrics.keys())[:16],
                },
            )
        except Exception as exc:
            logger.warning(
                "callback %s.on_log failed: %s", type(self).__name__, exc
            )

    def on_save(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        try:
            global_step = int(getattr(state, "global_step", 0) or 0)
            output_dir = getattr(args, "output_dir", None)
            self._record_step(
                "train.checkpoint",
                input={"output_dir": output_dir, "global_step": global_step},
                output={"epoch": getattr(state, "epoch", None)},
                action=ActionType.CHECKPOINT,
            )
        except Exception as exc:
            logger.warning(
                "callback %s.on_save failed: %s", type(self).__name__, exc
            )

    def on_epoch_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        try:
            self._record_step(
                "train.epoch_end",
                input={
                    "epoch": getattr(state, "epoch", None),
                    "global_step": getattr(state, "global_step", None),
                },
            )
        except Exception as exc:
            logger.warning(
                "callback %s.on_epoch_end failed: %s", type(self).__name__, exc
            )

    def on_train_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        try:
            self._record_step(
                "train.end",
                input={"run_id": self.run_id},
                output={
                    "global_step": getattr(state, "global_step", None),
                    "epoch": getattr(state, "epoch", None),
                },
            )
        except Exception as exc:
            logger.warning(
                "callback %s.on_train_end failed: %s", type(self).__name__, exc
            )


_SKIPPED_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "total_flos",
        "step",
        "epoch",
    }
)


def _is_skipped_key(key: str) -> bool:
    """Filter out boilerplate keys we do not want in trace output."""
    return key in _SKIPPED_METRIC_KEYS or key.startswith("eval_runtime")
