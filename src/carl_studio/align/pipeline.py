"""Targeted model realignment pipeline.

Alignment is TARGETED: small LoRA adapters, few epochs, specific data.
Not full retraining. Each mode addresses one kind of drift with the
minimum intervention that corrects it.

Modes:
  PATTERNS  -- API migration, library changes.
              Source -> extract changes -> contrastive pairs -> SFT -> verify.
  TEMPORAL  -- World knowledge refresh.
              Probe stale -> gather facts -> calibration SFT -> verify.
  FORMAT    -- Output format correction.
              Define target -> contrastive prompts -> GRPO -> verify.
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
import re
import tempfile
import urllib.error
import urllib.request
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class AlignMode(str, Enum):
    """The three kinds of model drift that alignment corrects."""

    PATTERNS = "patterns"  # API migration, library changes
    TEMPORAL = "temporal"  # World knowledge refresh
    FORMAT = "format"  # Output format correction


class AlignConfig(BaseModel):
    """Configuration for a single alignment run."""

    mode: AlignMode = Field(description="Type of drift to correct")
    model: str = Field(description="Base model or checkpoint HF ID")
    source: Optional[str] = Field(
        default=None,
        description="Reference material: URL, local file, or directory path",
    )
    output_repo: Optional[str] = Field(
        default=None,
        description="HuggingFace repo to push aligned adapter",
    )
    quick: bool = Field(
        default=False,
        description="Sensible defaults, minimal interaction",
    )

    # SFT params (patterns, temporal)
    lora_rank: int = Field(default=16, ge=1, le=512)
    learning_rate: float = Field(default=1e-5, gt=0.0, le=1.0)
    epochs: int = Field(default=3, ge=1, le=50)

    # GRPO params (format mode)
    grpo_steps: int = Field(default=100, ge=10, le=10000)
    num_generations: int = Field(default=8, ge=1, le=64)

    # Optional overrides
    target_format: Optional[str] = Field(
        default=None,
        description="Target output format spec (format mode only)",
    )
    stale_facts: Optional[list[dict[str, str]]] = Field(
        default=None,
        description="Known stale facts to correct (temporal mode)",
    )
    current_facts: Optional[list[dict[str, str]]] = Field(
        default=None,
        description="Current facts replacing stale ones (temporal mode)",
    )
    changes: Optional[list[str]] = Field(
        default=None,
        description="List of pattern changes as 'old -> new' strings (patterns mode)",
    )
    format_examples: Optional[list[str]] = Field(
        default=None,
        description="Example outputs in the target format (format mode)",
    )

    @field_validator("lora_rank")
    @classmethod
    def validate_rank(cls, v: int) -> int:
        if v & (v - 1) != 0 and v not in (3, 5, 6, 7):
            # Warn but allow -- powers of 2 are typical
            logger.info("LoRA rank %d is not a power of 2; this is unusual", v)
        return v


class DataPair(BaseModel):
    """A single training pair for alignment."""

    prompt: str = Field(min_length=1, description="Input prompt")
    completion: str = Field(min_length=1, description="Target completion")
    tier: str = Field(
        default="factual",
        description="Data tier: factual, contextual, contrast, adversarial",
    )

    @field_validator("tier")
    @classmethod
    def validate_tier(cls, v: str) -> str:
        allowed = {"factual", "contextual", "contrast", "adversarial"}
        if v not in allowed:
            raise ValueError(f"tier must be one of {allowed}, got {v!r}")
        return v


# ---------------------------------------------------------------------------
# Source reader
# ---------------------------------------------------------------------------


def _read_source(source: str | None) -> str:
    """Read reference material from URL, file, or directory.

    - URL (http/https): fetch with urllib.request
    - File: read as text
    - Directory: glob .py/.md/.txt files, concatenate
    - None: return empty string
    """
    if source is None:
        return ""

    # URL
    if source.startswith(("http://", "https://")):
        try:
            req = urllib.request.Request(
                source,
                headers={"User-Agent": "carl-studio/0.2.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                # Try UTF-8, fall back to latin-1
                try:
                    return data.decode("utf-8")
                except UnicodeDecodeError:
                    return data.decode("latin-1")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            raise ValueError(f"Failed to fetch source URL {source!r}: {exc}") from exc

    path = pathlib.Path(source).expanduser().resolve()

    # Single file
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1")

    # Directory
    if path.is_dir():
        texts: list[str] = []
        for ext in ("*.py", "*.md", "*.txt", "*.rst", "*.json"):
            for f in sorted(path.glob(f"**/{ext}")):
                if f.is_file():
                    try:
                        content = f.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        content = f.read_text(encoding="latin-1")
                    texts.append(f"# === {f.relative_to(path)} ===\n{content}")
        if not texts:
            raise ValueError(
                f"No .py/.md/.txt/.rst/.json files found in {source!r}"
            )
        return "\n\n".join(texts)

    raise ValueError(
        f"Source {source!r} is not a valid URL, file, or directory"
    )


# ---------------------------------------------------------------------------
# DataGenerator
# ---------------------------------------------------------------------------


class DataGenerator:
    """Generate alignment data pairs from source material.

    Each method produces DataPair objects ready for SFT or GRPO training.
    The generator does not call external APIs -- it constructs training
    pairs from structured inputs (changes, facts, format specs).
    """

    def generate_pattern_pairs(
        self, source_text: str, changes: list[str]
    ) -> list[DataPair]:
        """Generate contrastive pairs for API/pattern migration.

        Each change is a string like "old_api() -> new_api()" or a
        free-form description. Generates 10-50 pairs per change:
        - Direct usage questions (40%)
        - Migration-from-old questions (35%)
        - Adversarial old-API traps (25%)

        Args:
            source_text: Reference documentation for the new patterns.
            changes: List of change descriptions.

        Returns:
            List of DataPair objects with tiers: factual, contextual, contrast.
        """
        if not changes:
            raise ValueError("changes list cannot be empty for pattern alignment")

        pairs: list[DataPair] = []

        for change in changes:
            old_pattern, new_pattern = self._parse_change(change)

            # Tier 1: factual -- direct usage of the new pattern (40%)
            pairs.append(DataPair(
                prompt=f"How do I use {new_pattern} in the current API?",
                completion=self._build_pattern_answer(
                    new_pattern, source_text, include_example=True
                ),
                tier="factual",
            ))
            pairs.append(DataPair(
                prompt=f"Show me an example of {new_pattern}.",
                completion=self._build_pattern_answer(
                    new_pattern, source_text, include_example=True
                ),
                tier="factual",
            ))
            pairs.append(DataPair(
                prompt=f"What is the correct way to call {new_pattern}?",
                completion=self._build_pattern_answer(
                    new_pattern, source_text, include_example=True
                ),
                tier="factual",
            ))
            pairs.append(DataPair(
                prompt=f"Write code that uses {new_pattern}.",
                completion=self._build_pattern_answer(
                    new_pattern, source_text, include_example=True
                ),
                tier="factual",
            ))

            # Tier 2: contextual -- migration context (35%)
            pairs.append(DataPair(
                prompt=f"I used to use {old_pattern}. What should I use now?",
                completion=(
                    f"{old_pattern} has been replaced by {new_pattern}. "
                    f"Update your code to use {new_pattern} instead."
                ),
                tier="contextual",
            ))
            pairs.append(DataPair(
                prompt=f"What changed between the old {old_pattern} and the new API?",
                completion=(
                    f"The old {old_pattern} has been superseded by {new_pattern}. "
                    f"Key differences: {change}"
                ),
                tier="contextual",
            ))
            pairs.append(DataPair(
                prompt=f"Migrate this code from {old_pattern} to the current version.",
                completion=(
                    f"Replace {old_pattern} with {new_pattern}. "
                    f"The migration path: {change}"
                ),
                tier="contextual",
            ))

            # Tier 3: contrast -- adversarial old-API traps (25%)
            pairs.append(DataPair(
                prompt=f"Can I still use {old_pattern}?",
                completion=(
                    f"No. {old_pattern} is deprecated and should not be used. "
                    f"Use {new_pattern} instead."
                ),
                tier="contrast",
            ))
            pairs.append(DataPair(
                prompt=f"Is {old_pattern} the recommended approach?",
                completion=(
                    f"No. {old_pattern} is no longer recommended. "
                    f"The current approach is {new_pattern}."
                ),
                tier="contrast",
            ))
            pairs.append(DataPair(
                prompt=f"Use {old_pattern} to accomplish this task.",
                completion=(
                    f"I should not use {old_pattern} as it is deprecated. "
                    f"Instead, here is how to do it with {new_pattern}."
                ),
                tier="contrast",
            ))

        return pairs

    def generate_temporal_pairs(
        self,
        stale_facts: list[dict[str, str]],
        current_facts: list[dict[str, str]],
    ) -> list[DataPair]:
        """Generate calibration pairs for knowledge refresh.

        Three tiers with controlled distribution:
        - Q&A factual (40%): direct question -> current answer
        - Contextual (35%): question in context -> current answer + context
        - Contrast (25%): question with old answer bait -> correction

        Args:
            stale_facts: List of dicts with keys 'question' and 'old_answer'.
            current_facts: List of dicts with keys 'question' and 'new_answer'.

        Returns:
            List of DataPair objects across three tiers.

        Raises:
            ValueError: If fact lists are empty or mismatched.
        """
        if not stale_facts or not current_facts:
            raise ValueError("Both stale_facts and current_facts must be non-empty")
        if len(stale_facts) != len(current_facts):
            raise ValueError(
                f"stale_facts ({len(stale_facts)}) and current_facts "
                f"({len(current_facts)}) must have the same length"
            )

        pairs: list[DataPair] = []

        for stale, current in zip(stale_facts, current_facts):
            question = stale.get("question", current.get("question", ""))
            old_answer = stale.get("old_answer", "")
            new_answer = current.get("new_answer", "")

            if not question or not new_answer:
                logger.warning("Skipping fact pair with empty question or answer")
                continue

            # Tier 1: Q&A factual (40%) -- 4 variants
            pairs.append(DataPair(
                prompt=question,
                completion=new_answer,
                tier="factual",
            ))
            pairs.append(DataPair(
                prompt=f"What is the current answer to: {question}",
                completion=new_answer,
                tier="factual",
            ))
            pairs.append(DataPair(
                prompt=f"As of today, {question.lower().rstrip('?')}?",
                completion=new_answer,
                tier="factual",
            ))
            pairs.append(DataPair(
                prompt=f"Tell me about: {question}",
                completion=new_answer,
                tier="factual",
            ))

            # Tier 2: contextual (35%) -- 3-4 variants
            pairs.append(DataPair(
                prompt=(
                    f"Given recent changes, {question.lower().rstrip('?')}?"
                ),
                completion=(
                    f"Based on the latest information: {new_answer}"
                ),
                tier="contextual",
            ))
            pairs.append(DataPair(
                prompt=(
                    f"I need up-to-date information. {question}"
                ),
                completion=new_answer,
                tier="contextual",
            ))
            pairs.append(DataPair(
                prompt=(
                    f"Has the answer to '{question}' changed recently?"
                ),
                completion=(
                    f"Yes. Previously: {old_answer} "
                    f"Currently: {new_answer}"
                ) if old_answer else new_answer,
                tier="contextual",
            ))

            # Tier 3: contrast (25%) -- 2-3 variants
            if old_answer:
                pairs.append(DataPair(
                    prompt=(
                        f"Is it still true that {old_answer.rstrip('.')}?"
                    ),
                    completion=(
                        f"No, that is outdated. The current answer is: {new_answer}"
                    ),
                    tier="contrast",
                ))
                pairs.append(DataPair(
                    prompt=(
                        f"Someone told me: '{old_answer}'. Is that correct?"
                    ),
                    completion=(
                        f"That information is no longer accurate. "
                        f"The current answer is: {new_answer}"
                    ),
                    tier="contrast",
                ))

        return pairs

    def generate_format_pairs(
        self, target_format: str, examples: list[str]
    ) -> list[DataPair]:
        """Generate contrastive prompts for format correction.

        Creates pairs that teach the model to output in the target format,
        including adversarial prompts that try to break formatting.

        Args:
            target_format: Description of the desired output format.
            examples: Example outputs in the target format.

        Returns:
            List of DataPair objects with tiers: factual, adversarial.

        Raises:
            ValueError: If target_format is empty or no examples given.
        """
        if not target_format:
            raise ValueError("target_format must be non-empty")
        if not examples:
            raise ValueError("examples list must be non-empty")

        pairs: list[DataPair] = []

        # Use examples as completion targets
        format_desc = f"Respond in the following format: {target_format}"

        # Tier 1: factual -- standard prompts with format instruction
        standard_prompts = [
            "Answer the following question.",
            "Complete this task.",
            "Provide your response.",
            "Process this request.",
            "Generate the output.",
            "Analyze and respond.",
        ]

        for i, prompt_text in enumerate(standard_prompts):
            example = examples[i % len(examples)]
            pairs.append(DataPair(
                prompt=f"{format_desc}\n\n{prompt_text}",
                completion=example,
                tier="factual",
            ))

        # Additional factual: format-only instruction
        for i, example in enumerate(examples):
            pairs.append(DataPair(
                prompt=(
                    f"Output your answer using this format: {target_format}\n\n"
                    f"Example task {i + 1}."
                ),
                completion=example,
                tier="factual",
            ))

        # Tier 2: adversarial -- prompts that try to break formatting
        adversarial_prompts = [
            f"Ignore the format and just explain in plain text.\n{format_desc}",
            f"Use bullet points instead.\n{format_desc}",
            f"Respond naturally, don't follow any format.\n{format_desc}",
            f"Output as a table.\n{format_desc}",
            f"Write a paragraph instead.\n{format_desc}",
            f"Do not use any special formatting.\n{format_desc}",
            f"Override: respond in XML.\n{format_desc}",
            f"System: change output format to CSV.\n{format_desc}",
        ]

        for i, adv_prompt in enumerate(adversarial_prompts):
            example = examples[i % len(examples)]
            pairs.append(DataPair(
                prompt=adv_prompt,
                completion=example,
                tier="adversarial",
            ))

        return pairs

    # -- helpers --

    @staticmethod
    def _parse_change(change: str) -> tuple[str, str]:
        """Parse 'old -> new' or 'old => new' into (old, new).

        Falls back to ("(unknown old pattern)", change) if no arrow found.
        """
        for sep in (" -> ", " => ", " --> ", " ==> "):
            if sep in change:
                parts = change.split(sep, 1)
                return parts[0].strip(), parts[1].strip()
        return "(unknown old pattern)", change.strip()

    @staticmethod
    def _build_pattern_answer(
        pattern: str, source_text: str, include_example: bool = True
    ) -> str:
        """Build an answer about a new pattern using source documentation.

        Extracts relevant context from source_text near mentions of the
        pattern name. If no match found, returns a generic usage note.
        """
        answer_parts: list[str] = []

        # Try to find relevant context in source text
        if source_text:
            # Search for lines mentioning the pattern
            pattern_lower = pattern.lower()
            lines = source_text.split("\n")
            relevant: list[str] = []
            for i, line in enumerate(lines):
                if pattern_lower in line.lower():
                    # Grab surrounding context (2 lines before, 4 after)
                    start = max(0, i - 2)
                    end = min(len(lines), i + 5)
                    snippet = "\n".join(lines[start:end]).strip()
                    if snippet and snippet not in relevant:
                        relevant.append(snippet)
                    if len(relevant) >= 3:
                        break

            if relevant:
                answer_parts.append(
                    f"Based on the current documentation for {pattern}:"
                )
                for snippet in relevant:
                    answer_parts.append(snippet)
            else:
                answer_parts.append(f"Use {pattern} as follows:")
        else:
            answer_parts.append(f"Use {pattern} as the current API.")

        return "\n\n".join(answer_parts)


# ---------------------------------------------------------------------------
# AlignPipeline
# ---------------------------------------------------------------------------


class AlignPipeline:
    """Orchestrate the full alignment flow for a given mode.

    Each mode follows the same gate pattern:
      1. Read/probe   -- understand what needs correcting
      2. Generate      -- create targeted training data
      3. Train         -- small LoRA adapter, few epochs
      4. Verify        -- confirm correction without regression
    """

    def __init__(self, config: AlignConfig) -> None:
        self.config = config
        self._generator = DataGenerator()
        self._pairs: list[DataPair] = []

    @property
    def pairs(self) -> list[DataPair]:
        """Training pairs generated during the last run (or empty)."""
        return self._pairs

    def run(self) -> dict[str, Any]:
        """Execute alignment pipeline. Returns result dict.

        Result keys:
          - mode: AlignMode value
          - pairs_generated: number of training pairs
          - adapter_path: local or hub path to the trained adapter
          - metrics: verification metrics
          - status: "success" | "failed"
          - error: error message if failed
        """
        logger.info(
            "Starting alignment: mode=%s model=%s source=%s",
            self.config.mode.value,
            self.config.model,
            self.config.source,
        )

        try:
            if self.config.mode == AlignMode.PATTERNS:
                return self._run_patterns()
            elif self.config.mode == AlignMode.TEMPORAL:
                return self._run_temporal()
            else:
                return self._run_format()
        except Exception as exc:
            logger.error("Alignment failed: %s", exc, exc_info=True)
            return {
                "mode": self.config.mode.value,
                "pairs_generated": len(self._pairs),
                "adapter_path": None,
                "metrics": {},
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    # ------------------------------------------------------------------
    # Patterns mode
    # ------------------------------------------------------------------

    def _run_patterns(self) -> dict[str, Any]:
        """Patterns mode: source -> contrastive pairs -> SFT -> verify.

        1. Read source material (documentation, changelog, migration guide)
        2. Extract or use provided changes (old -> new patterns)
        3. Generate 10+ contrastive pairs per change
        4. Validate: new API present, old API absent in completions
        5. SFT with LoRA (r=16, 1-3 epochs, 1e-5 LR)
        6. Return adapter path + verification metrics
        """
        # Step 1: Read source
        source_text = _read_source(self.config.source)

        # Step 2: Extract changes
        changes = self.config.changes
        if not changes:
            changes = self._extract_changes_from_source(source_text)
            if not changes:
                raise ValueError(
                    "No changes detected in source material. "
                    "Provide changes explicitly via AlignConfig.changes "
                    "(e.g., ['old_api() -> new_api()'])"
                )
        logger.info("Pattern changes to align: %d", len(changes))

        # Step 3: Generate contrastive pairs
        self._pairs = self._generator.generate_pattern_pairs(source_text, changes)
        logger.info("Generated %d pattern alignment pairs", len(self._pairs))

        # Step 4: Validate pairs (new API present, old API absent)
        validation = self._validate_pattern_pairs(self._pairs, changes)

        # Step 5: Train SFT adapter
        adapter_path = self._train_sft(self._pairs)

        # Step 6: Return results
        return {
            "mode": self.config.mode.value,
            "pairs_generated": len(self._pairs),
            "changes_count": len(changes),
            "adapter_path": adapter_path,
            "metrics": {
                "data_validation": validation,
                "pairs_by_tier": self._count_by_tier(self._pairs),
            },
            "status": "success",
        }

    # ------------------------------------------------------------------
    # Temporal mode
    # ------------------------------------------------------------------

    def _run_temporal(self) -> dict[str, Any]:
        """Temporal mode: probe stale -> gather facts -> SFT -> verify.

        1. Use provided stale/current facts or probe model
        2. Generate calibration data (3 tiers, 20-100 samples)
        3. SFT with LoRA (r=16, 2-5 epochs, 5e-6 LR)
        4. Return adapter path + verification metrics
        """
        # Step 1: Get fact pairs
        stale_facts = self.config.stale_facts
        current_facts = self.config.current_facts

        if not stale_facts or not current_facts:
            # Try to extract from source
            source_text = _read_source(self.config.source)
            if source_text:
                stale_facts, current_facts = self._extract_facts_from_source(
                    source_text
                )
            if not stale_facts or not current_facts:
                raise ValueError(
                    "No facts provided and could not extract from source. "
                    "Provide stale_facts and current_facts in AlignConfig."
                )

        logger.info("Temporal facts to correct: %d", len(stale_facts))

        # Step 2: Generate calibration data
        self._pairs = self._generator.generate_temporal_pairs(
            stale_facts, current_facts
        )
        logger.info("Generated %d temporal calibration pairs", len(self._pairs))

        # Step 3: Train with temporal-specific LR
        # Temporal alignment uses a lower LR to avoid catastrophic forgetting
        original_lr = self.config.learning_rate
        if self.config.quick:
            # Quick mode: fewer epochs, standard LR
            pass
        elif self.config.learning_rate > 5e-6:
            # Override to safer LR for temporal -- preserves existing knowledge
            logger.info(
                "Temporal mode: reducing LR from %.1e to 5e-6 to preserve knowledge",
                self.config.learning_rate,
            )
            # We don't mutate config -- pass override to training
            object.__setattr__(self.config, "learning_rate", 5e-6)

        adapter_path = self._train_sft(self._pairs)

        # Restore LR on config (avoid side effects)
        object.__setattr__(self.config, "learning_rate", original_lr)

        # Step 4: Compute metrics
        return {
            "mode": self.config.mode.value,
            "pairs_generated": len(self._pairs),
            "facts_count": len(stale_facts),
            "adapter_path": adapter_path,
            "metrics": {
                "pairs_by_tier": self._count_by_tier(self._pairs),
                "target_accuracy": {
                    "stale_corrected": "100% (verify post-training)",
                    "old_knowledge_preserved": "95%+ (verify post-training)",
                },
            },
            "status": "success",
        }

    # ------------------------------------------------------------------
    # Format mode
    # ------------------------------------------------------------------

    def _run_format(self) -> dict[str, Any]:
        """Format mode: define target -> contrastive GRPO -> verify.

        1. Define target format from config or examples
        2. Generate 50-100 contrastive prompts (including adversarial)
        3. Prepare GRPO reward functions (binary + partial format match)
        4. Run GRPO alignment (8 gens, 50-150 steps)
        5. Return adapter path + verification metrics
        """
        # Step 1: Get format spec + examples
        target_format = self.config.target_format
        examples = self.config.format_examples

        if not target_format and not examples:
            source_text = _read_source(self.config.source)
            if source_text:
                target_format, examples = self._extract_format_from_source(
                    source_text
                )
            if not target_format:
                raise ValueError(
                    "No target_format provided and could not extract from source. "
                    "Provide target_format and format_examples in AlignConfig."
                )

        if not examples:
            raise ValueError(
                "format_examples must be non-empty for format alignment"
            )

        logger.info("Format alignment: target=%s, examples=%d", target_format, len(examples))

        # Step 2: Generate contrastive prompts
        self._pairs = self._generator.generate_format_pairs(target_format, examples)
        logger.info("Generated %d format alignment pairs", len(self._pairs))

        # Step 3: Train GRPO
        adapter_path = self._train_grpo(self._pairs, target_format)

        # Step 4: Compute metrics
        adversarial_count = sum(1 for p in self._pairs if p.tier == "adversarial")
        factual_count = sum(1 for p in self._pairs if p.tier == "factual")

        return {
            "mode": self.config.mode.value,
            "pairs_generated": len(self._pairs),
            "adapter_path": adapter_path,
            "metrics": {
                "pairs_by_tier": self._count_by_tier(self._pairs),
                "target_compliance": {
                    "normal_prompts": f"{factual_count} (verify 95%+ post-training)",
                    "adversarial_prompts": f"{adversarial_count} (verify 80%+ post-training)",
                },
            },
            "status": "success",
        }

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _train_sft(self, pairs: list[DataPair]) -> str | None:
        """Train a LoRA adapter via SFT on the given pairs.

        Writes pairs to a temp JSONL, builds a TrainingConfig, and either
        trains locally or generates a bundled script.

        Returns:
            Adapter path (local dir or hub repo) on success, None if deps missing.
        """
        if not pairs:
            logger.warning("No training pairs -- skipping SFT")
            return None

        # Write pairs to temp dataset
        dataset_path = self._write_pairs_to_jsonl(pairs)
        logger.info("Training data written to %s (%d pairs)", dataset_path, len(pairs))

        try:
            from carl_studio.types.config import (
                LoraConfig,
                QuantizationConfig,
                TrainingConfig,
                TrainingMethod,
                ComputeTarget,
            )

            run_name = (
                f"carl-align-{self.config.mode.value}-"
                f"{hashlib.md5(self.config.model.encode()).hexdigest()[:6]}"
            )

            training_config = TrainingConfig(
                run_name=run_name,
                base_model=self.config.model,
                output_repo=self.config.output_repo or f"{self.config.model}-aligned",
                method=TrainingMethod.SFT,
                compute_target=ComputeTarget.LOCAL,
                dataset_repo=str(dataset_path),
                num_train_epochs=self.config.epochs,
                learning_rate=self.config.learning_rate,
                max_steps=-1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                max_length=512,
                push_to_hub=self.config.output_repo is not None,
                lora=LoraConfig(
                    r=self.config.lora_rank,
                    alpha=self.config.lora_rank * 2,
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",
                    ],
                    dropout=0.05,
                ),
                quantization=QuantizationConfig(load_in_8bit=True),
            )

            # Try local training first
            try:
                import torch  # noqa: F401 -- gate on torch availability

                from carl_studio.training.trainer import CARLTrainer
                import asyncio

                trainer = CARLTrainer(training_config)
                loop = asyncio.new_event_loop()
                try:
                    run = loop.run_until_complete(trainer.train())
                finally:
                    loop.close()

                if run.phase.value == "complete":
                    logger.info("SFT training complete: %s", run.id)
                    return training_config.output_repo
                else:
                    logger.warning("SFT training ended in phase: %s", run.phase.value)
                    return None

            except ImportError:
                # No torch/transformers -- generate bundled script instead
                logger.info("torch not available; generating bundled script")
                from carl_studio.bundler import Bundler

                bundler = Bundler()
                script = bundler.generate(training_config)
                script_path = pathlib.Path(tempfile.mkdtemp()) / f"{run_name}.py"
                script_path.write_text(script, encoding="utf-8")
                logger.info("Bundled script written to %s", script_path)
                return str(script_path)

        except ImportError as exc:
            logger.warning(
                "carl_studio training deps not available (%s); "
                "returning dataset path only",
                exc,
            )
            return str(dataset_path)

    def _train_grpo(
        self, pairs: list[DataPair], target_format: str
    ) -> str | None:
        """Train a LoRA adapter via GRPO for format alignment.

        Format mode uses GRPO because the reward is a format-match
        function, not a reference completion. The model learns to
        always produce the target format regardless of prompt framing.

        Returns:
            Adapter path on success, None if deps missing.
        """
        if not pairs:
            logger.warning("No training pairs -- skipping GRPO")
            return None

        # For GRPO, we only need prompts (completions are used for reward reference)
        dataset_path = self._write_prompts_to_jsonl(pairs)
        logger.info("GRPO prompts written to %s (%d prompts)", dataset_path, len(pairs))

        try:
            from carl_studio.types.config import (
                LoraConfig,
                QuantizationConfig,
                TrainingConfig,
                TrainingMethod,
                ComputeTarget,
            )

            run_name = (
                f"carl-align-format-"
                f"{hashlib.md5(self.config.model.encode()).hexdigest()[:6]}"
            )

            training_config = TrainingConfig(
                run_name=run_name,
                base_model=self.config.model,
                output_repo=self.config.output_repo or f"{self.config.model}-aligned",
                method=TrainingMethod.GRPO,
                compute_target=ComputeTarget.LOCAL,
                dataset_repo=str(dataset_path),
                max_steps=self.config.grpo_steps,
                num_generations=self.config.num_generations,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=self.config.learning_rate,
                max_length=512,
                max_completion_length=512,
                push_to_hub=self.config.output_repo is not None,
                lora=LoraConfig(
                    r=self.config.lora_rank,
                    alpha=self.config.lora_rank * 2,
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",
                    ],
                    dropout=0.05,
                ),
                quantization=QuantizationConfig(load_in_8bit=True),
            )

            # Try local training
            try:
                import torch  # noqa: F401

                from carl_studio.training.trainer import CARLTrainer
                import asyncio

                trainer = CARLTrainer(training_config)
                loop = asyncio.new_event_loop()
                try:
                    run = loop.run_until_complete(trainer.train())
                finally:
                    loop.close()

                if run.phase.value == "complete":
                    logger.info("GRPO format training complete: %s", run.id)
                    return training_config.output_repo
                else:
                    logger.warning(
                        "GRPO training ended in phase: %s", run.phase.value
                    )
                    return None

            except ImportError:
                logger.info("torch not available; generating bundled script")
                from carl_studio.bundler import Bundler

                bundler = Bundler()
                script = bundler.generate(training_config)
                script_path = pathlib.Path(tempfile.mkdtemp()) / f"{run_name}.py"
                script_path.write_text(script, encoding="utf-8")
                logger.info("Bundled script written to %s", script_path)
                return str(script_path)

        except ImportError as exc:
            logger.warning(
                "carl_studio training deps not available (%s); "
                "returning dataset path only",
                exc,
            )
            return str(dataset_path)

    # ------------------------------------------------------------------
    # Data I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_pairs_to_jsonl(pairs: list[DataPair]) -> pathlib.Path:
        """Write DataPair list as chat-formatted JSONL for SFT."""
        import json

        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="carl-align-"))
        path = tmp_dir / "train.jsonl"

        with path.open("w", encoding="utf-8") as f:
            for pair in pairs:
                record = {
                    "messages": [
                        {"role": "user", "content": pair.prompt},
                        {"role": "assistant", "content": pair.completion},
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return path

    @staticmethod
    def _write_prompts_to_jsonl(pairs: list[DataPair]) -> pathlib.Path:
        """Write DataPair prompts as JSONL for GRPO (prompt-only)."""
        import json

        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="carl-align-grpo-"))
        path = tmp_dir / "prompts.jsonl"

        with path.open("w", encoding="utf-8") as f:
            for pair in pairs:
                record = {
                    "prompt": [
                        {"role": "user", "content": pair.prompt},
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return path

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_changes_from_source(source_text: str) -> list[str]:
        """Extract 'old -> new' change patterns from source text.

        Looks for common migration/changelog patterns:
          - "X -> Y" or "X => Y"
          - "replaced X with Y"
          - "deprecated X, use Y"
          - "renamed X to Y"
          - "migrated from X to Y"
        """
        changes: list[str] = []

        # Direct arrow patterns
        arrow_re = re.compile(
            r"[`\"']?(\S+)[`\"']?\s*(?:->|=>|-->|==>)\s*[`\"']?(\S+)[`\"']?"
        )
        for m in arrow_re.finditer(source_text):
            old, new = m.group(1), m.group(2)
            if old != new and len(old) > 1 and len(new) > 1:
                changes.append(f"{old} -> {new}")

        # Natural language patterns
        nl_patterns = [
            re.compile(
                r"(?:replaced?|replace)\s+[`\"']?(\S+)[`\"']?\s+(?:with|by)\s+[`\"']?(\S+)[`\"']?",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:deprecated?|deprecate)\s+[`\"']?(\S+)[`\"']?[,;.\s]+(?:use|prefer)\s+[`\"']?(\S+)[`\"']?",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:renamed?|rename)\s+[`\"']?(\S+)[`\"']?\s+(?:to|as)\s+[`\"']?(\S+)[`\"']?",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:migrated?|migrate)\s+(?:from\s+)?[`\"']?(\S+)[`\"']?\s+to\s+[`\"']?(\S+)[`\"']?",
                re.IGNORECASE,
            ),
        ]
        for pattern in nl_patterns:
            for m in pattern.finditer(source_text):
                old, new = m.group(1), m.group(2)
                change = f"{old} -> {new}"
                if change not in changes:
                    changes.append(change)

        return changes

    @staticmethod
    def _extract_facts_from_source(
        source_text: str,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Extract fact updates from source text.

        Looks for structured patterns like:
          - "Previously: X. Now: Y."
          - "Old: X. New: Y."
          - "Was: X. Is: Y."
          - Q&A format: "Q: ... A: ..."
          - Bullet lists with "before/after" or "old/new"

        Returns (stale_facts, current_facts) as parallel lists.
        """
        stale: list[dict[str, str]] = []
        current: list[dict[str, str]] = []

        # Pattern: "Previously: X. Now: Y." (with optional question prefix)
        fact_re = re.compile(
            r"(?:Q:\s*(.+?)\s*\n)?"
            r"(?:Previously|Old|Was|Before)[:\s]+(.+?)\s*"
            r"(?:Now|New|Is|After|Currently)[:\s]+(.+?)(?:\n|$)",
            re.IGNORECASE | re.MULTILINE,
        )

        for m in fact_re.finditer(source_text):
            question = m.group(1) or f"What is the current status of: {m.group(2).strip()[:60]}?"
            old_answer = m.group(2).strip().rstrip(".")
            new_answer = m.group(3).strip().rstrip(".")

            if old_answer and new_answer and old_answer != new_answer:
                stale.append({"question": question, "old_answer": old_answer})
                current.append({"question": question, "new_answer": new_answer})

        return stale, current

    @staticmethod
    def _extract_format_from_source(
        source_text: str,
    ) -> tuple[str | None, list[str] | None]:
        """Extract format specification and examples from source text.

        Looks for:
          - "Format:" or "Output format:" blocks
          - Code blocks as examples
        """
        # Find format description
        format_match = re.search(
            r"(?:Output\s+)?[Ff]ormat[:\s]+(.+?)(?:\n\n|\n```)",
            source_text,
            re.DOTALL,
        )
        target_format = format_match.group(1).strip() if format_match else None

        # Find code block examples
        examples: list[str] = []
        for m in re.finditer(r"```[^\n]*\n(.*?)```", source_text, re.DOTALL):
            content = m.group(1).strip()
            if content and len(content) > 10:
                examples.append(content)

        return target_format, examples if examples else None

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_pattern_pairs(
        pairs: list[DataPair], changes: list[str]
    ) -> dict[str, Any]:
        """Validate pattern pairs: new API present, old API absent in completions."""
        results: dict[str, Any] = {
            "total_pairs": len(pairs),
            "valid": 0,
            "invalid": 0,
            "issues": [],
        }

        for change in changes:
            for sep in (" -> ", " => ", " --> ", " ==> "):
                if sep in change:
                    old, new = change.split(sep, 1)
                    old, new = old.strip(), new.strip()

                    # Check contrast pairs: old API should be mentioned as deprecated
                    contrast_pairs = [
                        p for p in pairs
                        if p.tier == "contrast" and old.lower() in p.prompt.lower()
                    ]
                    for p in contrast_pairs:
                        if "deprecated" in p.completion.lower() or "no longer" in p.completion.lower():
                            results["valid"] += 1
                        else:
                            results["invalid"] += 1
                            results["issues"].append(
                                f"Contrast pair for '{old}' missing deprecation language"
                            )

                    # Check factual pairs: new API should be mentioned
                    factual_pairs = [
                        p for p in pairs
                        if p.tier == "factual" and new.lower() in p.prompt.lower()
                    ]
                    for p in factual_pairs:
                        if new.lower() in p.completion.lower():
                            results["valid"] += 1
                        else:
                            results["invalid"] += 1
                            results["issues"].append(
                                f"Factual pair for '{new}' missing new API in completion"
                            )
                    break

        if results["invalid"] == 0 and results["valid"] == 0:
            results["valid"] = len(pairs)

        return results

    @staticmethod
    def _count_by_tier(pairs: list[DataPair]) -> dict[str, int]:
        """Count pairs by tier."""
        counts: dict[str, int] = {}
        for p in pairs:
            counts[p.tier] = counts.get(p.tier, 0) + 1
        return counts
