"""Template-based QA pair generation from source chunks.

No LLM required.  Uses structural parsing + regex extraction to produce
three tiers of questions:
  - FACTUAL (40%):     Direct extraction from source
  - CONCEPTUAL (35%):  Requires understanding relationships
  - APPLICATION (25%): Apply knowledge to new context
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from carl_studio.learn.ingest import SourceChunk


class QATier(str, Enum):
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    APPLICATION = "application"


class QAPair(BaseModel):
    """A single question-answer pair derived from source material."""

    question: str
    answer: str
    tier: QATier
    source_chunk_id: int


# Tier distribution targets
_TIER_RATIOS = {QATier.FACTUAL: 0.40, QATier.CONCEPTUAL: 0.35, QATier.APPLICATION: 0.25}


class QAGenerator:
    """Generates QA pairs from source chunks using templates, not LLM calls."""

    def generate(
        self,
        chunks: list[SourceChunk],
        pairs_per_chunk: int = 15,
        context_prefix: str = "",
    ) -> list[QAPair]:
        """Generate QA pairs across all 3 tiers for each chunk.

        Args:
            chunks: Source chunks to generate from.
            pairs_per_chunk: Target number of pairs per chunk. Actual count
                may be lower if the chunk lacks extractable content.
            context_prefix: Optional domain context (e.g. from WorkFrame.attention_query())
                prepended to each chunk's text for entity/definition extraction,
                biasing extraction toward the frame's vocabulary.

        Returns:
            Flat list of QAPair across all chunks and tiers.
        """
        if not chunks:
            return []
        if pairs_per_chunk < 1:
            raise ValueError(f"pairs_per_chunk must be >= 1, got {pairs_per_chunk}")

        all_pairs: list[QAPair] = []

        for chunk in chunks:
            # Compute per-tier targets
            n_factual = max(1, round(pairs_per_chunk * _TIER_RATIOS[QATier.FACTUAL]))
            n_conceptual = max(1, round(pairs_per_chunk * _TIER_RATIOS[QATier.CONCEPTUAL]))
            n_application = max(1, round(pairs_per_chunk * _TIER_RATIOS[QATier.APPLICATION]))

            # Extract entities and definitions from the chunk
            # When a frame context_prefix is active, prepend it to bias extraction
            # toward the frame's domain vocabulary and entities.
            enriched = f"{context_prefix}\n\n{chunk.text}" if context_prefix else chunk.text
            definitions = self._extract_definitions(enriched)
            entities = self._extract_entities(enriched)
            code_blocks = self._extract_code_blocks(chunk.text)

            pairs: list[QAPair] = []

            # --- Factual tier ---
            pairs.extend(
                self._generate_factual(chunk, definitions, entities, code_blocks, n_factual)
            )

            # --- Conceptual tier ---
            pairs.extend(
                self._generate_conceptual(chunk, definitions, entities, n_conceptual)
            )

            # --- Application tier ---
            pairs.extend(
                self._generate_application(chunk, definitions, entities, code_blocks, n_application)
            )

            all_pairs.extend(pairs)

        return all_pairs

    # ------------------------------------------------------------------
    # Tier generators
    # ------------------------------------------------------------------

    def _generate_factual(
        self,
        chunk: SourceChunk,
        definitions: list[tuple[str, str]],
        entities: list[str],
        code_blocks: list[str],
        target: int,
    ) -> list[QAPair]:
        """Factual: 'What is X?' / 'How does Y work?' from source."""
        pairs: list[QAPair] = []

        # From definitions: "What is <term>?"
        for term, definition in definitions[:target]:
            pairs.append(
                QAPair(
                    question=f"What is {term}?",
                    answer=definition,
                    tier=QATier.FACTUAL,
                    source_chunk_id=chunk.chunk_id,
                )
            )
            if len(pairs) >= target:
                break

        # From code blocks: "What does this code do?"
        for block in code_blocks:
            if len(pairs) >= target:
                break
            # Extract function/class name if present
            name_match = re.search(r"(?:def|class|function)\s+(\w+)", block)
            if name_match:
                name = name_match.group(1)
                pairs.append(
                    QAPair(
                        question=f"What does {name} do?",
                        answer=self._summarize_code(block),
                        tier=QATier.FACTUAL,
                        source_chunk_id=chunk.chunk_id,
                    )
                )

        # From entities: "What is <entity>?"
        used_terms = {p.question for p in pairs}
        for entity in entities:
            if len(pairs) >= target:
                break
            q = f"What is {entity}?"
            if q in used_terms:
                continue
            # Find the sentence containing this entity
            sentence = self._find_context_sentence(chunk.text, entity)
            if sentence:
                pairs.append(
                    QAPair(
                        question=q,
                        answer=sentence,
                        tier=QATier.FACTUAL,
                        source_chunk_id=chunk.chunk_id,
                    )
                )
                used_terms.add(q)

        return pairs[:target]

    def _generate_conceptual(
        self,
        chunk: SourceChunk,
        definitions: list[tuple[str, str]],
        entities: list[str],
        target: int,
    ) -> list[QAPair]:
        """Conceptual: 'Why does X do Y?' / 'What is the relationship between X and Y?'"""
        pairs: list[QAPair] = []
        terms = [d[0] for d in definitions] + entities

        # Relationship questions between pairs of entities
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                if len(pairs) >= target:
                    break
                a, b = terms[i], terms[j]
                # Both must appear in the same passage
                context = self._find_joint_context(chunk.text, a, b)
                if context:
                    pairs.append(
                        QAPair(
                            question=f"What is the relationship between {a} and {b}?",
                            answer=context,
                            tier=QATier.CONCEPTUAL,
                            source_chunk_id=chunk.chunk_id,
                        )
                    )
            if len(pairs) >= target:
                break

        # "Why" questions from definitions
        for term, definition in definitions:
            if len(pairs) >= target:
                break
            if any(kw in definition.lower() for kw in ("because", "in order to", "to ", "for ", "so that")):
                pairs.append(
                    QAPair(
                        question=f"Why is {term} used?",
                        answer=definition,
                        tier=QATier.CONCEPTUAL,
                        source_chunk_id=chunk.chunk_id,
                    )
                )

        # "How" questions from longer definitions
        for term, definition in definitions:
            if len(pairs) >= target:
                break
            if len(definition) > 100:
                pairs.append(
                    QAPair(
                        question=f"How does {term} work?",
                        answer=definition,
                        tier=QATier.CONCEPTUAL,
                        source_chunk_id=chunk.chunk_id,
                    )
                )

        return pairs[:target]

    def _generate_application(
        self,
        chunk: SourceChunk,
        definitions: list[tuple[str, str]],
        entities: list[str],
        code_blocks: list[str],
        target: int,
    ) -> list[QAPair]:
        """Application: 'Given scenario Z, how would you use X?'"""
        pairs: list[QAPair] = []

        # Code usage questions
        for block in code_blocks:
            if len(pairs) >= target:
                break
            name_match = re.search(r"(?:def|class|function)\s+(\w+)", block)
            if name_match:
                name = name_match.group(1)
                pairs.append(
                    QAPair(
                        question=f"Write an example showing how to use {name}.",
                        answer=block.strip(),
                        tier=QATier.APPLICATION,
                        source_chunk_id=chunk.chunk_id,
                    )
                )

        # Application from definitions
        for term, definition in definitions:
            if len(pairs) >= target:
                break
            context = self._find_context_sentence(chunk.text, term)
            if context and len(context) > 50:
                pairs.append(
                    QAPair(
                        question=f"In a project that needs {term.lower()}, how would you apply it?",
                        answer=f"{term}: {definition} In practice, {context}",
                        tier=QATier.APPLICATION,
                        source_chunk_id=chunk.chunk_id,
                    )
                )

        # Scenario questions from entities
        for entity in entities:
            if len(pairs) >= target:
                break
            context = self._find_context_sentence(chunk.text, entity)
            if context:
                pairs.append(
                    QAPair(
                        question=f"Given a system that uses {entity}, what should you consider?",
                        answer=context,
                        tier=QATier.APPLICATION,
                        source_chunk_id=chunk.chunk_id,
                    )
                )

        return pairs[:target]

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_definitions(text: str) -> list[tuple[str, str]]:
        """Extract term-definition pairs from text.

        Matches patterns:
          - "X is ..." / "X are ..."
          - "X: ..." (definition-style)
          - "X -- ..." (em-dash definitions)
          - "class X" / "def X" docstrings
        """
        defs: list[tuple[str, str]] = []
        seen_terms: set[str] = set()

        # Pattern: "X is/are ..."
        for m in re.finditer(
            r"(?:^|\.\s+)([A-Z][\w\s]{1,40}?)\s+(?:is|are)\s+(.+?)(?:\.|$)",
            text,
            re.MULTILINE,
        ):
            term = m.group(1).strip()
            defn = m.group(2).strip()
            if term.lower() not in seen_terms and len(defn) > 10:
                defs.append((term, defn))
                seen_terms.add(term.lower())

        # Pattern: "**Term** -- definition" or "Term: definition"
        for m in re.finditer(
            r"(?:\*\*(.+?)\*\*|`(.+?)`)\s*(?:--|:|\u2014)\s*(.+?)(?:\n|$)", text
        ):
            term = (m.group(1) or m.group(2)).strip()
            defn = m.group(3).strip()
            if term.lower() not in seen_terms and len(defn) > 10:
                defs.append((term, defn))
                seen_terms.add(term.lower())

        # Pattern: Python class/function with docstring
        for m in re.finditer(
            r'(?:class|def)\s+(\w+)[^:]*:\s*\n\s*"""(.+?)"""',
            text,
            re.DOTALL,
        ):
            term = m.group(1).strip()
            docstring = m.group(2).strip().split("\n")[0]  # First line only
            if term.lower() not in seen_terms and len(docstring) > 10:
                defs.append((term, docstring))
                seen_terms.add(term.lower())

        return defs

    @staticmethod
    def _extract_entities(text: str) -> list[str]:
        """Extract likely entity names (capitalized multi-word, CamelCase, backtick-quoted)."""
        entities: list[str] = []
        seen: set[str] = set()

        # CamelCase identifiers
        for m in re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", text):
            e = m.group(1)
            if e.lower() not in seen:
                entities.append(e)
                seen.add(e.lower())

        # Backtick-quoted terms
        for m in re.finditer(r"`(\w[\w.]+)`", text):
            e = m.group(1)
            if e.lower() not in seen and len(e) > 2:
                entities.append(e)
                seen.add(e.lower())

        # Capitalized phrases (2-4 words, not at sentence start)
        for m in re.finditer(r"(?<=\s)([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})\b", text):
            e = m.group(1)
            if e.lower() not in seen:
                entities.append(e)
                seen.add(e.lower())

        return entities

    @staticmethod
    def _extract_code_blocks(text: str) -> list[str]:
        """Extract fenced code blocks and indented code."""
        blocks: list[str] = []

        # Fenced: ```...```
        for m in re.finditer(r"```\w*\n(.+?)```", text, re.DOTALL):
            block = m.group(1).strip()
            if len(block) > 20:
                blocks.append(block)

        # Indented code (4+ spaces or tab, multi-line)
        if not blocks:
            for m in re.finditer(r"(?:^(?:    |\t).+\n?){3,}", text, re.MULTILINE):
                block = m.group(0).strip()
                if len(block) > 20:
                    blocks.append(block)

        return blocks

    @staticmethod
    def _find_context_sentence(text: str, term: str) -> Optional[str]:
        """Find the sentence containing a term."""
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            if term.lower() in sent.lower() and len(sent.strip()) > 20:
                return sent.strip()
        return None

    @staticmethod
    def _find_joint_context(text: str, term_a: str, term_b: str) -> Optional[str]:
        """Find a passage containing both terms."""
        # Try sentence-level first
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            lower = sent.lower()
            if term_a.lower() in lower and term_b.lower() in lower and len(sent) > 30:
                return sent.strip()

        # Fall back to paragraph-level
        paragraphs = re.split(r"\n\s*\n", text)
        for para in paragraphs:
            lower = para.lower()
            if term_a.lower() in lower and term_b.lower() in lower:
                # Trim to ~500 chars
                if len(para) > 500:
                    return para[:500].rsplit(" ", 1)[0] + "..."
                return para.strip()

        return None

    @staticmethod
    def _summarize_code(code: str) -> str:
        """Extract a brief description of what a code block does.

        Uses the docstring if present, otherwise the first comment,
        otherwise the first few lines.
        """
        # Docstring
        m = re.search(r'"""(.+?)"""', code, re.DOTALL)
        if m:
            return m.group(1).strip().split("\n")[0]

        # Single-line docstring
        m = re.search(r"'''(.+?)'''", code, re.DOTALL)
        if m:
            return m.group(1).strip().split("\n")[0]

        # Comment
        m = re.search(r"#\s*(.+)", code)
        if m:
            return m.group(1).strip()

        # Fallback: first 2 meaningful lines
        lines = [l.strip() for l in code.split("\n") if l.strip() and not l.strip().startswith("#")]
        return " ".join(lines[:2])[:200]
