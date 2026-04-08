"""Tests for carl_studio.learn module."""

import ast
import tempfile
import textwrap
from pathlib import Path

import pytest

from carl_studio.learn.ingest import SourceChunk, SourceIngester, SourceType
from carl_studio.learn.qa_gen import QAGenerator, QAPair, QATier
from carl_studio.learn.quality import QualityGate, QualityResult
from carl_studio.learn.pipeline import LearnConfig, LearnPipeline


# ======================================================================
# SourceIngester
# ======================================================================


class TestSourceIngester:
    def test_ingest_text_directly(self):
        ingester = SourceIngester()
        chunks = ingester.ingest("This is raw text for testing.\n\nSecond paragraph.", SourceType.TEXT)
        assert len(chunks) >= 1
        assert chunks[0].source == "<text>"
        assert chunks[0].chunk_id == 0

    def test_ingest_file(self, tmp_path: Path):
        f = tmp_path / "sample.md"
        f.write_text("# Heading\n\nSome content about testing.\n\n# Another\n\nMore content here.")
        ingester = SourceIngester()
        chunks = ingester.ingest(str(f), SourceType.FILE)
        assert len(chunks) >= 1
        assert str(f) in chunks[0].source

    def test_ingest_directory(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("def foo():\n    '''Foo does things.'''\n    return 1\n")
        (tmp_path / "b.md").write_text("# Title\n\nContent here.\n")
        (tmp_path / "c.jpg").write_bytes(b"\xff\xd8")  # non-ingestable
        ingester = SourceIngester()
        chunks = ingester.ingest(str(tmp_path), SourceType.DIRECTORY)
        assert len(chunks) >= 1
        sources = {c.source for c in chunks}
        assert any("a.py" in s for s in sources)
        assert any("b.md" in s for s in sources)
        assert not any("c.jpg" in s for s in sources)

    def test_ingest_file_not_found(self):
        ingester = SourceIngester()
        with pytest.raises(FileNotFoundError):
            ingester.ingest("/nonexistent/path.txt", SourceType.FILE)

    def test_ingest_empty_directory(self, tmp_path: Path):
        ingester = SourceIngester()
        with pytest.raises(ValueError, match="No ingestable files"):
            ingester.ingest(str(tmp_path), SourceType.DIRECTORY)

    def test_detect_type_file(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("content")
        ingester = SourceIngester()
        assert ingester._detect_type(str(f)) == SourceType.FILE

    def test_detect_type_directory(self, tmp_path: Path):
        ingester = SourceIngester()
        assert ingester._detect_type(str(tmp_path)) == SourceType.DIRECTORY

    def test_detect_type_url(self):
        ingester = SourceIngester()
        assert ingester._detect_type("https://example.com/doc") == SourceType.URL

    def test_detect_type_text(self):
        ingester = SourceIngester()
        long_text = "word " * 50
        assert ingester._detect_type(long_text) == SourceType.TEXT

    def test_chunk_ids_sequential(self, tmp_path: Path):
        f = tmp_path / "big.txt"
        # Write enough content to produce multiple chunks
        f.write_text(("Paragraph about topic.\n\n") * 200)
        ingester = SourceIngester()
        chunks = ingester.ingest(str(f), SourceType.FILE)
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(ids)))

    def test_python_splitting(self):
        code = textwrap.dedent("""\
            import os

            class Foo:
                '''A class.'''
                pass

            def bar():
                '''A function.'''
                return 1

            def baz():
                '''Another function.'''
                return 2
        """)
        ingester = SourceIngester()
        segments = ingester._split_python(code)
        assert len(segments) >= 2  # At least class + functions

    def test_markdown_splitting(self):
        md = "# Intro\n\nSome text.\n\n## Details\n\nMore text.\n\n### Sub\n\nDeep text."
        ingester = SourceIngester()
        segments = ingester._split_markdown(md)
        assert len(segments) >= 2


# ======================================================================
# QAGenerator
# ======================================================================


class TestQAGenerator:
    @pytest.fixture
    def sample_chunks(self) -> list[SourceChunk]:
        return [
            SourceChunk(
                text=textwrap.dedent("""\
                    CoherenceProbe is a monitoring tool that measures alignment during training.
                    It computes the Phi order parameter, which ranges from 0 (uniform) to 1 (delta).

                    The PhaseTransitionGate detects when crystallization occurs.
                    It uses a sliding window to check if Phi exceeds the threshold consistently.

                    ```python
                    def compute_phi(logits):
                        '''Compute the order parameter from logits.'''
                        probs = softmax(logits)
                        entropy = -sum(p * log(p) for p in probs)
                        return 1 - entropy / log(len(probs))
                    ```

                    CARL stands for Coherence-Aware Reinforcement Learning.
                    It uses the conservation law T* = kappa * d to govern training signals.
                """),
                source="docs/carl.md",
                chunk_id=0,
            ),
        ]

    def test_generate_produces_pairs(self, sample_chunks: list[SourceChunk]):
        gen = QAGenerator()
        pairs = gen.generate(sample_chunks, pairs_per_chunk=10)
        assert len(pairs) > 0
        assert all(isinstance(p, QAPair) for p in pairs)

    def test_generate_has_all_tiers(self, sample_chunks: list[SourceChunk]):
        gen = QAGenerator()
        pairs = gen.generate(sample_chunks, pairs_per_chunk=15)
        tiers = {p.tier for p in pairs}
        # Should have at least factual and one other tier
        assert QATier.FACTUAL in tiers
        assert len(tiers) >= 2

    def test_generate_source_chunk_ids_match(self, sample_chunks: list[SourceChunk]):
        gen = QAGenerator()
        pairs = gen.generate(sample_chunks, pairs_per_chunk=5)
        valid_ids = {c.chunk_id for c in sample_chunks}
        for p in pairs:
            assert p.source_chunk_id in valid_ids

    def test_generate_empty_chunks(self):
        gen = QAGenerator()
        pairs = gen.generate([], pairs_per_chunk=10)
        assert pairs == []

    def test_generate_invalid_pairs_per_chunk(self, sample_chunks: list[SourceChunk]):
        gen = QAGenerator()
        with pytest.raises(ValueError):
            gen.generate(sample_chunks, pairs_per_chunk=0)

    def test_extract_definitions(self):
        text = "CoherenceProbe is a tool that monitors training alignment."
        gen = QAGenerator()
        defs = gen._extract_definitions(text)
        assert len(defs) >= 1
        assert any("CoherenceProbe" in d[0] for d in defs)

    def test_extract_entities(self):
        text = "The PhaseTransitionGate uses `compute_phi` to detect changes."
        gen = QAGenerator()
        entities = gen._extract_entities(text)
        assert "PhaseTransitionGate" in entities
        assert "compute_phi" in entities

    def test_extract_code_blocks(self):
        text = "Example:\n```python\ndef foo():\n    return 42\n```\nEnd."
        gen = QAGenerator()
        blocks = gen._extract_code_blocks(text)
        assert len(blocks) == 1
        assert "def foo" in blocks[0]


# ======================================================================
# QualityGate
# ======================================================================


class TestQualityGate:
    @pytest.fixture
    def grounded_pair_and_chunk(self) -> tuple[QAPair, SourceChunk]:
        chunk = SourceChunk(
            text="CoherenceProbe monitors alignment by computing the Phi order parameter during training runs.",
            source="test.md",
            chunk_id=0,
        )
        pair = QAPair(
            question="What does CoherenceProbe do?",
            answer="CoherenceProbe monitors alignment by computing the Phi order parameter.",
            tier=QATier.FACTUAL,
            source_chunk_id=0,
        )
        return pair, chunk

    def test_grounded_pair_passes(self, grounded_pair_and_chunk):
        pair, chunk = grounded_pair_and_chunk
        gate = QualityGate(threshold=0.9)
        result = gate.validate([pair], [chunk])
        assert result.passed
        assert result.pass_rate == 1.0

    def test_ungrounded_pair_fails(self):
        chunk = SourceChunk(text="The cat sat on the mat.", source="test.md", chunk_id=0)
        pair = QAPair(
            question="What is quantum entanglement?",
            answer="Quantum entanglement is a phenomenon where particles become correlated across spacetime.",
            tier=QATier.FACTUAL,
            source_chunk_id=0,
        )
        gate = QualityGate(threshold=0.9)
        result = gate.validate([pair], [chunk])
        assert len(result.failed) > 0

    def test_code_validity_good_python(self):
        pair = QAPair(
            question="Show an example.",
            answer='Here is code:\n```python\ndef hello():\n    return "world"\n```',
            tier=QATier.APPLICATION,
            source_chunk_id=0,
        )
        gate = QualityGate()
        reason = gate._check_code_validity(pair)
        assert reason is None

    def test_code_validity_bad_python(self):
        pair = QAPair(
            question="Show an example.",
            answer='Code:\n```python\ndef hello(\n    return\n```',
            tier=QATier.APPLICATION,
            source_chunk_id=0,
        )
        gate = QualityGate()
        reason = gate._check_code_validity(pair)
        assert reason is not None
        assert "syntax error" in reason.lower()

    def test_code_validity_balanced_braces(self):
        pair = QAPair(
            question="Show JSON.",
            answer='```json\n{"key": "value", "nested": {"a": 1}}\n```',
            tier=QATier.APPLICATION,
            source_chunk_id=0,
        )
        gate = QualityGate()
        reason = gate._check_code_validity(pair)
        assert reason is None

    def test_code_validity_unbalanced_braces(self):
        pair = QAPair(
            question="Show JSON.",
            answer='```json\n{"key": "value", "nested": {"a": 1}\n```',
            tier=QATier.APPLICATION,
            source_chunk_id=0,
        )
        gate = QualityGate()
        reason = gate._check_code_validity(pair)
        assert reason is not None
        assert "unbalanced" in reason.lower()

    def test_completeness_empty_answer(self):
        pair = QAPair(question="What?", answer="", tier=QATier.FACTUAL, source_chunk_id=0)
        gate = QualityGate()
        reason = gate._check_completeness(pair)
        assert reason is not None

    def test_completeness_placeholder(self):
        pair = QAPair(
            question="What?",
            answer="This is a TODO placeholder for later.",
            tier=QATier.FACTUAL,
            source_chunk_id=0,
        )
        gate = QualityGate()
        reason = gate._check_completeness(pair)
        assert reason is not None
        assert "placeholder" in reason.lower() or "TODO" in reason

    def test_completeness_trailing_ellipsis(self):
        pair = QAPair(
            question="What?",
            answer="The system works by processing inputs and...",
            tier=QATier.FACTUAL,
            source_chunk_id=0,
        )
        gate = QualityGate()
        reason = gate._check_completeness(pair)
        assert reason is not None

    def test_completeness_good_answer(self):
        pair = QAPair(
            question="What?",
            answer="CoherenceProbe monitors training alignment through continuous measurement.",
            tier=QATier.FACTUAL,
            source_chunk_id=0,
        )
        gate = QualityGate()
        reason = gate._check_completeness(pair)
        assert reason is None

    def test_consistency_no_contradiction(self):
        pairs = [
            QAPair(question="What is X?", answer="X is a tool for monitoring.", tier=QATier.FACTUAL, source_chunk_id=0),
            QAPair(question="What is Y?", answer="Y is a gate for detection.", tier=QATier.FACTUAL, source_chunk_id=0),
        ]
        gate = QualityGate()
        conflicts = gate._check_consistency(pairs)
        assert conflicts == []

    def test_validate_empty_pairs(self):
        gate = QualityGate()
        result = gate.validate([], [])
        assert result.passed
        assert result.total == 0

    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            QualityGate(threshold=1.5)
        with pytest.raises(ValueError):
            QualityGate(threshold=-0.1)

    def test_braces_balanced(self):
        gate = QualityGate()
        assert gate._braces_balanced('{ "a": [1, 2] }')
        assert not gate._braces_balanced('{ "a": [1, 2 }')
        assert gate._braces_balanced('function() { return (x + y); }')
        assert gate._braces_balanced('')  # empty is balanced

    def test_braces_ignores_strings(self):
        gate = QualityGate()
        # Braces inside strings should not count
        assert gate._braces_balanced('{"key": "value with { in it"}')


# ======================================================================
# LearnConfig
# ======================================================================


class TestLearnConfig:
    def test_defaults(self):
        config = LearnConfig(source="./docs")
        assert config.depth == "shallow"
        assert config.quality_threshold == 0.9
        assert config.model == ""
        assert config._pairs_per_chunk() == 5

    def test_deep_depth(self):
        config = LearnConfig(source="./docs", depth="deep")
        assert config._pairs_per_chunk() == 15

    def test_threshold_bounds(self):
        with pytest.raises(Exception):
            LearnConfig(source="x", quality_threshold=2.0)


# ======================================================================
# LearnPipeline (integration)
# ======================================================================


class TestLearnPipeline:
    def test_full_pipeline_from_file(self, tmp_path: Path):
        f = tmp_path / "knowledge.md"
        f.write_text(textwrap.dedent("""\
            # CARL Framework

            CoherenceProbe is a monitoring tool that measures alignment during training.
            It computes the Phi order parameter, which ranges from 0 to 1.

            The PhaseTransitionGate detects when crystallization occurs during training.
            It uses a sliding window approach to check if Phi exceeds the threshold.

            CARL stands for Coherence-Aware Reinforcement Learning.
            The conservation law T* = kappa * d governs the training signal.
            This relationship between complexity and training time is fundamental.

            ```python
            def compute_phi(logits):
                '''Compute the Phi order parameter from model logits.'''
                probs = softmax(logits)
                entropy = -sum(p * log(p) for p in probs)
                return 1 - entropy / log(len(probs))
            ```

            The system processes inputs through multiple stages of evaluation.
            Each stage validates the output against coherence criteria before proceeding.
            This gated approach ensures quality at every step of the pipeline.
        """))

        config = LearnConfig(source=str(f), depth="shallow")
        pipeline = LearnPipeline(config)
        result = pipeline.run()

        assert result["chunks"] >= 1
        assert result["pairs_total"] >= 1
        assert "quality" in result
        assert result["quality"]["total"] >= 1
        # Dataset should be saved since no model specified
        if result["quality"]["passed"] and result["pairs_total"] > 0:
            assert result["dataset_path"] is not None or result["training"] is not None

    def test_pipeline_empty_source(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        config = LearnConfig(source=str(f))
        pipeline = LearnPipeline(config)
        result = pipeline.run()
        assert result["chunks"] == 0
        assert result["pairs_total"] == 0
