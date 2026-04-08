"""Source ingestion and chunking for carl learn.

Supports: URL, file, directory, HuggingFace dataset, raw text.
Chunking strategy adapts to content type:
  - Python/code: module-level boundaries (class/function defs)
  - Markdown: heading boundaries
  - Plain text: paragraph boundaries
Target chunk size: ~500-2000 tokens (estimated as chars / 4).
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    URL = "url"
    FILE = "file"
    DIRECTORY = "directory"
    HUB_DATASET = "hub_dataset"
    TEXT = "text"


class SourceChunk(BaseModel):
    """A chunk of source material."""

    text: str
    source: str = Field(description="Origin path, URL, or identifier")
    chunk_id: int


# Token estimate: ~4 chars per token
_MIN_CHUNK_CHARS = 200
_TARGET_CHUNK_CHARS = 3000
_MAX_CHUNK_CHARS = 8000

# File extensions to ingest from directories
_INGESTABLE_EXTS = {".py", ".md", ".txt", ".json", ".rst", ".yaml", ".yml", ".toml"}


class SourceIngester:
    """Reads and chunks source material."""

    def ingest(
        self, source: str, source_type: Optional[SourceType] = None
    ) -> list[SourceChunk]:
        """Auto-detect source type if not provided, then ingest and chunk.

        Args:
            source: Path, URL, HF dataset ID, or raw text.
            source_type: Explicit type override. Auto-detected if None.

        Returns:
            List of SourceChunk with chunk_id starting at 0.

        Raises:
            ValueError: If source cannot be read or type is unrecognized.
            FileNotFoundError: If file/directory does not exist.
        """
        if source_type is None:
            source_type = self._detect_type(source)

        if source_type == SourceType.TEXT:
            return self._chunk_text(source, origin="<text>")
        elif source_type == SourceType.FILE:
            return self._ingest_file(source)
        elif source_type == SourceType.DIRECTORY:
            return self._ingest_directory(source)
        elif source_type == SourceType.URL:
            return self._ingest_url(source)
        elif source_type == SourceType.HUB_DATASET:
            return self._ingest_hub_dataset(source)
        else:
            raise ValueError(f"Unknown source type: {source_type!r}")

    # ------------------------------------------------------------------
    # Type detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_type(source: str) -> SourceType:
        """Heuristic source type detection."""
        if source.startswith(("http://", "https://")):
            return SourceType.URL

        path = Path(source)
        if path.is_dir():
            return SourceType.DIRECTORY
        if path.is_file():
            return SourceType.FILE

        # HF dataset pattern: org/dataset or org/dataset-name
        if re.match(r"^[\w.-]+/[\w.-]+$", source) and not path.exists():
            return SourceType.HUB_DATASET

        # Fallback: if it contains newlines or is long, treat as text
        if "\n" in source or len(source) > 200:
            return SourceType.TEXT

        raise ValueError(
            f"Cannot detect source type for {source!r}. "
            "Provide source_type explicitly."
        )

    # ------------------------------------------------------------------
    # Ingestion methods
    # ------------------------------------------------------------------

    def _ingest_file(self, path: str) -> list[SourceChunk]:
        """Read a single file and chunk it."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        text = p.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return []
        return self._chunk_text(text, origin=str(p))

    def _ingest_directory(self, path: str) -> list[SourceChunk]:
        """Glob ingestable files from a directory, chunk each."""
        p = Path(path)
        if not p.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")

        chunks: list[SourceChunk] = []
        files = sorted(
            f for f in p.rglob("*") if f.is_file() and f.suffix in _INGESTABLE_EXTS
        )
        if not files:
            raise ValueError(
                f"No ingestable files found in {path}. "
                f"Supported extensions: {_INGESTABLE_EXTS}"
            )

        chunk_id = 0
        for f in files:
            text = f.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
            file_chunks = self._chunk_text(text, origin=str(f), start_id=chunk_id)
            chunks.extend(file_chunks)
            chunk_id += len(file_chunks)
        return chunks

    def _ingest_url(self, url: str) -> list[SourceChunk]:
        """Fetch URL content and chunk it."""
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "carl-studio/0.2.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise ValueError(f"Failed to fetch URL {url}: {exc}") from exc

        if not text.strip():
            return []

        # Strip HTML tags if content looks like HTML
        if "<html" in text.lower()[:500]:
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

        return self._chunk_text(text, origin=url)

    def _ingest_hub_dataset(self, dataset_id: str) -> list[SourceChunk]:
        """Load a HuggingFace dataset and extract text fields."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Install 'datasets' to ingest HuggingFace datasets: "
                "pip install datasets"
            ) from exc

        ds = load_dataset(dataset_id, split="train")
        text_fields = [
            col
            for col in ds.column_names
            if col in ("text", "content", "question", "answer", "instruction", "output")
        ]
        if not text_fields:
            raise ValueError(
                f"No text fields found in {dataset_id}. "
                f"Columns: {ds.column_names}"
            )

        chunks: list[SourceChunk] = []
        chunk_id = 0
        for i, row in enumerate(ds):
            parts = [str(row[f]) for f in text_fields if row.get(f)]
            combined = "\n\n".join(parts)
            if len(combined) < _MIN_CHUNK_CHARS:
                # Small rows: accumulate until we hit target
                if chunks and len(chunks[-1].text) < _TARGET_CHUNK_CHARS:
                    chunks[-1] = SourceChunk(
                        text=chunks[-1].text + "\n\n" + combined,
                        source=f"{dataset_id}[{chunks[-1].source.split('[')[1]}",
                        chunk_id=chunks[-1].chunk_id,
                    )
                    continue
            chunk = SourceChunk(
                text=combined, source=f"{dataset_id}[{i}]", chunk_id=chunk_id
            )
            chunks.append(chunk)
            chunk_id += 1
        return chunks

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_text(
        self, text: str, origin: str, start_id: int = 0
    ) -> list[SourceChunk]:
        """Split text into chunks using content-aware boundaries.

        Strategy selection:
          - .py files: split on top-level class/function definitions
          - .md/.rst files: split on headings
          - Everything else: split on double-newline (paragraph) boundaries
        """
        if origin.endswith(".py"):
            segments = self._split_python(text)
        elif origin.endswith((".md", ".rst")):
            segments = self._split_markdown(text)
        else:
            segments = self._split_paragraphs(text)

        # Merge small segments, split oversized ones
        merged = self._normalize_segments(segments)

        return [
            SourceChunk(text=seg.strip(), source=origin, chunk_id=start_id + i)
            for i, seg in enumerate(merged)
            if seg.strip()
        ]

    @staticmethod
    def _split_python(text: str) -> list[str]:
        """Split Python source on top-level class/function boundaries."""
        pattern = re.compile(r"^(?=(?:class |def |async def ))", re.MULTILINE)
        parts = pattern.split(text)
        return [p for p in parts if p.strip()]

    @staticmethod
    def _split_markdown(text: str) -> list[str]:
        """Split markdown on heading boundaries (# through ####)."""
        pattern = re.compile(r"^(?=#{1,4}\s)", re.MULTILINE)
        parts = pattern.split(text)
        return [p for p in parts if p.strip()]

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Split on double-newline boundaries."""
        parts = re.split(r"\n\s*\n", text)
        return [p for p in parts if p.strip()]

    @staticmethod
    def _normalize_segments(segments: list[str]) -> list[str]:
        """Merge undersized segments, split oversized ones."""
        result: list[str] = []
        buffer = ""

        for seg in segments:
            candidate = (buffer + "\n\n" + seg).strip() if buffer else seg.strip()

            if len(candidate) > _MAX_CHUNK_CHARS:
                # Flush buffer first
                if buffer.strip():
                    result.append(buffer.strip())
                    buffer = ""
                # Force-split oversized segment at sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", seg)
                sub_buf = ""
                for sent in sentences:
                    if len(sub_buf) + len(sent) > _TARGET_CHUNK_CHARS and sub_buf:
                        result.append(sub_buf.strip())
                        sub_buf = sent
                    else:
                        sub_buf = (sub_buf + " " + sent).strip() if sub_buf else sent
                if sub_buf.strip():
                    buffer = sub_buf.strip()
            elif len(candidate) >= _TARGET_CHUNK_CHARS:
                result.append(candidate)
                buffer = ""
            else:
                buffer = candidate

        if buffer.strip():
            result.append(buffer.strip())

        return result
