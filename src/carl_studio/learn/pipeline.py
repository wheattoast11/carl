"""LearnPipeline -- full ingest -> generate -> validate -> train lifecycle.

Orchestrates SourceIngester, QAGenerator, QualityGate, and optionally
triggers SFT training via CARLTrainer.

Heavy imports (torch, transformers, trl) are lazy -- only pulled
when config.model is specified and training is requested.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from carl_studio.learn.ingest import SourceIngester, SourceType
from carl_studio.learn.qa_gen import QAGenerator
from carl_studio.learn.quality import QualityGate

logger = logging.getLogger(__name__)


class LearnConfig(BaseModel):
    """Configuration for the learn pipeline."""

    source: str = Field(description="Path, URL, HF dataset ID, or raw text")
    source_type: Optional[SourceType] = Field(
        default=None, description="Explicit source type. Auto-detected if None."
    )
    model: str = Field(
        default="", description="Model to train. Empty string = skip training."
    )
    output_repo: Optional[str] = Field(
        default=None,
        description="HuggingFace repo to push trained model. Required if model is set.",
    )
    depth: str = Field(
        default="shallow",
        description="'shallow' (5 pairs/chunk) or 'deep' (15 pairs/chunk)",
    )
    quality_threshold: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Minimum pass rate for quality gate"
    )
    lora_rank: int = Field(default=32, ge=1, le=512, description="LoRA rank for training")
    learning_rate: float = Field(default=2e-5, gt=0.0, le=1.0)
    epochs: int = Field(default=3, ge=1, le=100)

    def _pairs_per_chunk(self) -> int:
        """Resolve depth to pairs_per_chunk."""
        return 15 if self.depth == "deep" else 5


class LearnPipeline:
    """Full pipeline: ingest -> generate -> validate -> (train).

    Usage::

        config = LearnConfig(source="./docs/", depth="deep")
        result = LearnPipeline(config).run()
        print(result["quality"]["pass_rate"])
    """

    def __init__(self, config: LearnConfig) -> None:
        self.config = config
        self._ingester = SourceIngester()
        self._generator = QAGenerator()
        self._gate = QualityGate(threshold=config.quality_threshold)

    def run(self, context_prefix: str = "") -> dict[str, Any]:
        """Execute the full pipeline.

        Args:
            context_prefix: Optional domain context (e.g. from WorkFrame.attention_query())
                that biases QA extraction toward the frame's vocabulary.

        Returns:
            dict with keys:
              - chunks: int (number of source chunks)
              - pairs_total: int (QA pairs generated)
              - quality: dict (QualityResult fields)
              - training: dict | None (training result if model specified)
              - dataset_path: str | None (path to saved dataset if not training)
        """
        # 1. Ingest source
        logger.info("Ingesting source: %s", self.config.source)
        chunks = self._ingester.ingest(
            self.config.source, source_type=self.config.source_type
        )
        logger.info("Ingested %d chunks", len(chunks))

        if not chunks:
            return {
                "chunks": 0,
                "pairs_total": 0,
                "quality": {"passed": True, "pass_rate": 1.0, "total": 0, "passed_count": 0, "failed": []},
                "training": None,
                "dataset_path": None,
            }

        # 2. Generate QA pairs
        pairs_per_chunk = self.config._pairs_per_chunk()
        logger.info("Generating QA pairs (%d per chunk)", pairs_per_chunk)
        pairs = self._generator.generate(
            chunks, pairs_per_chunk=pairs_per_chunk, context_prefix=context_prefix,
        )
        logger.info("Generated %d QA pairs", len(pairs))

        # 3. Quality gate
        logger.info("Running quality gate (threshold=%.2f)", self.config.quality_threshold)
        quality_result = self._gate.validate(pairs, chunks)
        logger.info(
            "Quality gate: %s (%.1f%% pass rate, %d/%d passed)",
            "PASS" if quality_result.passed else "FAIL",
            quality_result.pass_rate * 100,
            quality_result.passed_count,
            quality_result.total,
        )

        result: dict[str, Any] = {
            "chunks": len(chunks),
            "pairs_total": len(pairs),
            "quality": quality_result.model_dump(),
            "training": None,
            "dataset_path": None,
        }

        # 4. If gate fails, report and stop
        if not quality_result.passed:
            logger.warning(
                "Quality gate FAILED: %d pairs failed checks",
                len(quality_result.failed),
            )
            for f in quality_result.failed[:10]:
                logger.warning(
                    "  Pair %d: %s", f["pair_index"], f["reason"]
                )
            return result

        # 5. Filter to only passing pairs
        failed_indices = {f["pair_index"] for f in quality_result.failed}
        good_pairs = [p for i, p in enumerate(pairs) if i not in failed_indices]

        # 6. If model specified, prepare training config and train
        if self.config.model:
            training_result = self._train(good_pairs)
            result["training"] = training_result
        else:
            # Save dataset for later use
            dataset_path = self._save_dataset(good_pairs)
            result["dataset_path"] = dataset_path

        return result

    def _train(self, pairs: list) -> dict[str, Any]:
        """Prepare SFT dataset from QA pairs and train with CARLTrainer.

        Lazy-imports all heavy dependencies.
        """
        if not self.config.output_repo:
            raise ValueError(
                "output_repo is required when model is specified for training"
            )

        try:
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError(
                "Install 'datasets' for training: pip install datasets"
            ) from exc

        from carl_studio.types.config import (
            LoraConfig,
            TrainingConfig,
            TrainingMethod,
        )

        logger.info("Preparing SFT dataset from %d QA pairs", len(pairs))

        # Convert QA pairs to chat format
        conversations = []
        for pair in pairs:
            conversations.append({
                "messages": [
                    {"role": "user", "content": pair.question},
                    {"role": "assistant", "content": pair.answer},
                ],
            })

        dataset = Dataset.from_list(conversations)

        # Build training config
        train_config = TrainingConfig(
            run_name=f"carl-learn-{self.config.model.split('/')[-1]}",
            base_model=self.config.model,
            output_repo=self.config.output_repo,
            method=TrainingMethod.SFT,
            dataset_repo="",  # We'll pass dataset directly
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            lora=LoraConfig(r=self.config.lora_rank),
            push_to_hub=self.config.output_repo is not None,
        )

        logger.info("Starting SFT training: model=%s", self.config.model)

        # Inline training to avoid async complexity and pass dataset directly
        import asyncio

        from carl_studio.training.trainer import CARLTrainer

        trainer = CARLTrainer(train_config)
        run = asyncio.get_event_loop().run_until_complete(trainer.train())

        return {
            "run_id": run.id,
            "phase": run.phase.value,
            "hub_job_id": run.hub_job_id,
            "error": run.error_message,
        }

    @staticmethod
    def _save_dataset(pairs: list) -> str:
        """Save QA pairs as JSONL for later training.

        Returns path to saved file.
        """
        import json
        import tempfile
        from pathlib import Path

        output_dir = Path(tempfile.gettempdir()) / "carl_learn"
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / "qa_pairs.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                record = {
                    "messages": [
                        {"role": "user", "content": pair.question},
                        {"role": "assistant", "content": pair.answer},
                    ],
                    "tier": pair.tier.value,
                    "source_chunk_id": pair.source_chunk_id,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Saved %d QA pairs to %s", len(pairs), output_path)
        return str(output_path)
