"""Test alignment module: config, data generation, pipeline structure."""

import pytest
from carl_studio.align import AlignConfig, AlignMode, AlignPipeline, DataGenerator, DataPair


class TestAlignMode:
    def test_enum_values(self):
        assert AlignMode.PATTERNS.value == "patterns"
        assert AlignMode.TEMPORAL.value == "temporal"
        assert AlignMode.FORMAT.value == "format"

    def test_string_construction(self):
        assert AlignMode("patterns") is AlignMode.PATTERNS
        assert AlignMode("temporal") is AlignMode.TEMPORAL
        assert AlignMode("format") is AlignMode.FORMAT


class TestAlignConfig:
    def test_minimal_config(self):
        cfg = AlignConfig(mode=AlignMode.PATTERNS, model="test/model")
        assert cfg.mode == AlignMode.PATTERNS
        assert cfg.model == "test/model"
        assert cfg.lora_rank == 16
        assert cfg.learning_rate == 1e-5
        assert cfg.epochs == 3
        assert cfg.quick is False

    def test_format_mode_defaults(self):
        cfg = AlignConfig(mode=AlignMode.FORMAT, model="test/model")
        assert cfg.grpo_steps == 100
        assert cfg.num_generations == 8

    def test_source_optional(self):
        cfg = AlignConfig(mode=AlignMode.TEMPORAL, model="test/model")
        assert cfg.source is None

    def test_custom_params(self):
        cfg = AlignConfig(
            mode=AlignMode.PATTERNS,
            model="test/model",
            lora_rank=32,
            learning_rate=2e-5,
            epochs=5,
            output_repo="user/aligned-model",
        )
        assert cfg.lora_rank == 32
        assert cfg.learning_rate == 2e-5
        assert cfg.epochs == 5
        assert cfg.output_repo == "user/aligned-model"

    def test_invalid_lora_rank(self):
        with pytest.raises(ValueError):
            AlignConfig(mode=AlignMode.PATTERNS, model="test/model", lora_rank=0)

    def test_invalid_lr(self):
        with pytest.raises(ValueError):
            AlignConfig(mode=AlignMode.PATTERNS, model="test/model", learning_rate=-1)


class TestDataPair:
    def test_basic_creation(self):
        pair = DataPair(prompt="How?", completion="Like this.")
        assert pair.prompt == "How?"
        assert pair.completion == "Like this."
        assert pair.tier == "factual"

    def test_custom_tier(self):
        pair = DataPair(prompt="Q", completion="A", tier="adversarial")
        assert pair.tier == "adversarial"

    def test_invalid_tier(self):
        with pytest.raises(ValueError):
            DataPair(prompt="Q", completion="A", tier="invalid")

    def test_empty_prompt(self):
        with pytest.raises(ValueError):
            DataPair(prompt="", completion="A")

    def test_empty_completion(self):
        with pytest.raises(ValueError):
            DataPair(prompt="Q", completion="")

    def test_all_tiers(self):
        for tier in ("factual", "contextual", "contrast", "adversarial"):
            pair = DataPair(prompt="Q", completion="A", tier=tier)
            assert pair.tier == tier


class TestDataGenerator:
    def setup_method(self):
        self.gen = DataGenerator()

    # -- Pattern pairs --

    def test_pattern_pairs_basic(self):
        pairs = self.gen.generate_pattern_pairs(
            "Use new_func() for all operations.",
            ["old_func() -> new_func()"],
        )
        assert len(pairs) >= 10
        tiers = {p.tier for p in pairs}
        assert "factual" in tiers
        assert "contextual" in tiers
        assert "contrast" in tiers

    def test_pattern_pairs_multiple_changes(self):
        pairs = self.gen.generate_pattern_pairs(
            "Migration guide.",
            ["api_v1 -> api_v2", "connect() -> create_client()"],
        )
        assert len(pairs) >= 20  # 10+ per change

    def test_pattern_pairs_empty_changes(self):
        with pytest.raises(ValueError):
            self.gen.generate_pattern_pairs("source", [])

    def test_pattern_pairs_no_arrow(self):
        pairs = self.gen.generate_pattern_pairs("doc", ["use new_method instead"])
        assert len(pairs) >= 10
        # Should still work, with "(unknown old pattern)" as the old

    # -- Temporal pairs --

    def test_temporal_pairs_basic(self):
        stale = [{"question": "What year?", "old_answer": "2024"}]
        current = [{"question": "What year?", "new_answer": "2026"}]
        pairs = self.gen.generate_temporal_pairs(stale, current)
        assert len(pairs) >= 9  # 4 factual + 3 contextual + 2 contrast per fact
        tiers = {p.tier for p in pairs}
        assert "factual" in tiers
        assert "contextual" in tiers
        assert "contrast" in tiers

    def test_temporal_pairs_mismatched(self):
        with pytest.raises(ValueError):
            self.gen.generate_temporal_pairs(
                [{"question": "Q", "old_answer": "A"}],
                [],
            )

    def test_temporal_pairs_length_mismatch(self):
        with pytest.raises(ValueError):
            self.gen.generate_temporal_pairs(
                [{"question": "Q1", "old_answer": "A1"}],
                [{"question": "Q1", "new_answer": "B1"}, {"question": "Q2", "new_answer": "B2"}],
            )

    # -- Format pairs --

    def test_format_pairs_basic(self):
        pairs = self.gen.generate_format_pairs(
            "JSON with keys: action, target",
            ['{"action": "click", "target": "button"}'],
        )
        assert len(pairs) >= 14  # 6+ standard + 1+ example + 8 adversarial
        tiers = {p.tier for p in pairs}
        assert "factual" in tiers
        assert "adversarial" in tiers

    def test_format_pairs_empty_format(self):
        with pytest.raises(ValueError):
            self.gen.generate_format_pairs("", ["example"])

    def test_format_pairs_no_examples(self):
        with pytest.raises(ValueError):
            self.gen.generate_format_pairs("JSON format", [])


class TestAlignPipeline:
    def test_pipeline_init(self):
        cfg = AlignConfig(mode=AlignMode.PATTERNS, model="test/model")
        pipeline = AlignPipeline(cfg)
        assert pipeline.config is cfg
        assert pipeline.pairs == []

    def test_patterns_mode_requires_changes_or_source(self):
        cfg = AlignConfig(mode=AlignMode.PATTERNS, model="test/model")
        pipeline = AlignPipeline(cfg)
        result = pipeline.run()
        assert result["status"] == "failed"
        assert "No changes" in result["error"] or "source" in result["error"].lower()

    def test_temporal_mode_requires_facts_or_source(self):
        cfg = AlignConfig(mode=AlignMode.TEMPORAL, model="test/model")
        pipeline = AlignPipeline(cfg)
        result = pipeline.run()
        assert result["status"] == "failed"
        assert "facts" in result["error"].lower() or "source" in result["error"].lower()

    def test_format_mode_requires_format_or_source(self):
        cfg = AlignConfig(mode=AlignMode.FORMAT, model="test/model")
        pipeline = AlignPipeline(cfg)
        result = pipeline.run()
        assert result["status"] == "failed"

    def test_result_dict_structure(self):
        cfg = AlignConfig(mode=AlignMode.TEMPORAL, model="test/model")
        pipeline = AlignPipeline(cfg)
        result = pipeline.run()
        # Even on failure, result has required keys
        assert "mode" in result
        assert "pairs_generated" in result
        assert "status" in result
        assert result["mode"] == "temporal"
