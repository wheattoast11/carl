"""Deterministic quality gate for QA pairs.

No LLM calls.  Four checks, all rule-based:
  1. Source Grounding: key terms in answer traceable to source chunks
  2. Internal Consistency: no contradictions between pairs
  3. Code Validity: code blocks must parse (ast.parse for Python, brace-match for others)
  4. Answer Completeness: no trailing fragments, no placeholders
"""

from __future__ import annotations

import ast
import re
from typing import Optional

from pydantic import BaseModel, Field

from carl_studio.learn.ingest import SourceChunk
from carl_studio.learn.qa_gen import QAPair


class QualityResult(BaseModel):
    """Outcome of the quality gate."""

    passed: bool
    pass_rate: float = Field(ge=0.0, le=1.0)
    total: int
    passed_count: int
    failed: list[dict] = Field(
        default_factory=list,
        description="List of {pair_index, reason} for each failed pair",
    )


# Stopwords to exclude from grounding checks
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "about", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then", "once",
    "that", "this", "these", "those", "it", "its", "and", "but", "or",
    "nor", "not", "so", "yet", "both", "each", "all", "any", "few",
    "more", "most", "other", "some", "such", "no", "only", "own",
    "same", "than", "too", "very", "just", "also", "how", "what",
    "which", "who", "whom", "why", "when", "where", "here", "there",
    "up", "down", "if", "while", "because", "until", "although",
})

# Placeholder patterns
_PLACEHOLDER_PATTERNS = [
    r"\bTODO\b",
    r"\bFIXME\b",
    r"\bXXX\b",
    r"\bHACK\b",
    r"\.\.\.\s*$",
    r"the above",
    r"as mentioned",
    r"as described",
    r"see below",
    r"placeholder",
    r"\[insert\b",
    r"\[your\b",
    r"\[fill\b",
]
_PLACEHOLDER_RE = re.compile("|".join(_PLACEHOLDER_PATTERNS), re.IGNORECASE)

# Incomplete sentence endings (no terminal punctuation, ends with conjunction, etc.)
_INCOMPLETE_ENDINGS = re.compile(
    r"(?:,\s*$|;\s*$|\band\s*$|\bor\s*$|\bbut\s*$|\bthe\s*$|\bwith\s*$|\bin\s*$|\bfor\s*$)"
)


class QualityGate:
    """Deterministic quality gate for generated QA pairs.

    All checks are rule-based.  No network calls, no LLM inference.
    """

    def __init__(self, threshold: float = 0.9) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold

    def validate(
        self, pairs: list[QAPair], chunks: list[SourceChunk]
    ) -> QualityResult:
        """Run all 4 checks.  Returns QualityResult with pass/fail per pair.

        Args:
            pairs: QA pairs to validate.
            chunks: Source chunks used to generate the pairs.

        Returns:
            QualityResult with pass_rate compared to self.threshold.
        """
        if not pairs:
            return QualityResult(
                passed=True, pass_rate=1.0, total=0, passed_count=0, failed=[]
            )

        # Build chunk lookup
        chunk_map = {c.chunk_id: c for c in chunks}

        # Find internally inconsistent pairs (affects multiple pairs)
        inconsistent_indices = set(self._check_consistency(pairs))

        failed: list[dict] = []
        for i, pair in enumerate(pairs):
            reasons: list[str] = []

            # 1. Source grounding
            source_chunk = chunk_map.get(pair.source_chunk_id)
            if source_chunk is not None:
                grounding_reason = self._check_grounding(pair, [source_chunk])
                if grounding_reason:
                    reasons.append(grounding_reason)

            # 2. Internal consistency
            if i in inconsistent_indices:
                reasons.append("Contradicts another QA pair in the set")

            # 3. Code validity
            code_reason = self._check_code_validity(pair)
            if code_reason:
                reasons.append(code_reason)

            # 4. Completeness
            completeness_reason = self._check_completeness(pair)
            if completeness_reason:
                reasons.append(completeness_reason)

            if reasons:
                failed.append({"pair_index": i, "reason": "; ".join(reasons)})

        passed_count = len(pairs) - len(failed)
        pass_rate = passed_count / len(pairs) if pairs else 1.0

        return QualityResult(
            passed=pass_rate >= self.threshold,
            pass_rate=round(pass_rate, 4),
            total=len(pairs),
            passed_count=passed_count,
            failed=failed,
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_grounding(
        self, pair: QAPair, chunks: list[SourceChunk]
    ) -> Optional[str]:
        """Check that key terms in the answer appear in source chunks.

        Extracts content words (non-stopword, 3+ chars) from the answer
        and verifies at least 60% appear in the source text.

        Returns reason string if failed, None if passed.
        """
        answer_words = self._extract_key_terms(pair.answer)
        if not answer_words:
            return None  # Very short answer, skip grounding check

        source_text = " ".join(c.text for c in chunks).lower()

        grounded = sum(1 for w in answer_words if w in source_text)
        ratio = grounded / len(answer_words) if answer_words else 1.0

        if ratio < 0.6:
            ungrounded = [w for w in answer_words if w not in source_text][:5]
            return (
                f"Low source grounding ({ratio:.0%}): "
                f"terms not found in source: {ungrounded}"
            )
        return None

    def _check_consistency(self, pairs: list[QAPair]) -> list[int]:
        """Find pairs that contradict each other.

        Simple heuristic: if two pairs answer the same question
        (normalized) with different answers, flag both.

        Returns indices of conflicting pairs.
        """
        conflicting: set[int] = set()
        q_map: dict[str, list[int]] = {}

        for i, pair in enumerate(pairs):
            normalized_q = self._normalize_question(pair.question)
            q_map.setdefault(normalized_q, []).append(i)

        for indices in q_map.values():
            if len(indices) < 2:
                continue
            # Compare answers pairwise
            for a_idx in range(len(indices)):
                for b_idx in range(a_idx + 1, len(indices)):
                    i, j = indices[a_idx], indices[b_idx]
                    if self._answers_contradict(pairs[i].answer, pairs[j].answer):
                        conflicting.add(i)
                        conflicting.add(j)

        return sorted(conflicting)

    def _check_code_validity(self, pair: QAPair) -> Optional[str]:
        """If the answer contains code blocks, check they parse.

        - Python code: ast.parse()
        - Other code: basic brace/bracket matching

        Returns reason string if failed, None if passed.
        """
        code_blocks = self._extract_code_from_answer(pair.answer)
        if not code_blocks:
            return None

        for i, (lang, code) in enumerate(code_blocks):
            if lang in ("python", "py", ""):
                # Try Python parse
                try:
                    ast.parse(code)
                except SyntaxError as exc:
                    return f"Code block {i} has Python syntax error: {exc.msg} (line {exc.lineno})"
            else:
                # Basic brace matching for other languages
                if not self._braces_balanced(code):
                    return f"Code block {i} has unbalanced braces/brackets"

        return None

    def _check_completeness(self, pair: QAPair) -> Optional[str]:
        """Check the answer is complete — no fragments, no placeholders.

        Returns reason string if failed, None if passed.
        """
        answer = pair.answer.strip()

        if not answer:
            return "Answer is empty"

        if len(answer) < 10:
            return f"Answer too short ({len(answer)} chars)"

        # Check for placeholder patterns
        m = _PLACEHOLDER_RE.search(answer)
        if m:
            return f"Answer contains placeholder: {m.group()!r}"

        # Check for incomplete trailing
        if _INCOMPLETE_ENDINGS.search(answer):
            return "Answer ends with incomplete fragment"

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_key_terms(text: str) -> list[str]:
        """Extract non-stopword terms (3+ chars, lowercased) from text."""
        # Strip code blocks before extracting terms
        cleaned = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        words = re.findall(r"\b[a-zA-Z_]\w{2,}\b", cleaned.lower())
        return [w for w in words if w not in _STOPWORDS]

    @staticmethod
    def _normalize_question(q: str) -> str:
        """Normalize a question for comparison."""
        q = q.lower().strip().rstrip("?").strip()
        q = re.sub(r"\s+", " ", q)
        return q

    @staticmethod
    def _answers_contradict(a: str, b: str) -> bool:
        """Simple contradiction check between two answers.

        Two answers contradict if they share key terms but one
        contains a negation the other doesn't.  This is conservative --
        it only catches obvious contradictions.
        """
        a_lower, b_lower = a.lower(), b.lower()
        negations = {"not", "no", "never", "none", "cannot", "can't", "won't", "don't", "doesn't"}

        a_neg = any(neg in a_lower.split() for neg in negations)
        b_neg = any(neg in b_lower.split() for neg in negations)

        # One negated, one not -> potential contradiction
        if a_neg != b_neg:
            # Check they share meaningful content
            a_terms = set(re.findall(r"\b\w{4,}\b", a_lower)) - _STOPWORDS
            b_terms = set(re.findall(r"\b\w{4,}\b", b_lower)) - _STOPWORDS
            overlap = a_terms & b_terms
            if len(overlap) >= 2:
                return True

        return False

    @staticmethod
    def _extract_code_from_answer(answer: str) -> list[tuple[str, str]]:
        """Extract (language, code) tuples from fenced code blocks in the answer."""
        blocks: list[tuple[str, str]] = []
        for m in re.finditer(r"```(\w*)\n(.+?)```", answer, re.DOTALL):
            lang = m.group(1).lower()
            code = m.group(2).strip()
            if code:
                blocks.append((lang, code))
        return blocks

    @staticmethod
    def _braces_balanced(code: str) -> bool:
        """Check that braces, brackets, and parens are balanced."""
        stack: list[str] = []
        pairs = {")": "(", "]": "[", "}": "{"}
        openers = set(pairs.values())

        # Ignore characters inside string literals
        in_string: Optional[str] = None
        prev_char = ""

        for ch in code:
            if in_string:
                if ch == in_string and prev_char != "\\":
                    in_string = None
                prev_char = ch
                continue

            if ch in ('"', "'", "`"):
                in_string = ch
                prev_char = ch
                continue

            if ch in openers:
                stack.append(ch)
            elif ch in pairs:
                if not stack or stack[-1] != pairs[ch]:
                    return False
                stack.pop()

            prev_char = ch

        return len(stack) == 0
