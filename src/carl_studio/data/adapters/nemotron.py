"""Adapter for NVIDIA Nemotron-Cascade-RL-SWE dataset."""

from __future__ import annotations

from typing import Any, Iterator

from carl_studio.data.adapters.base import DataAdapter
from carl_studio.data.types import (
    Domain,
    EnvironmentSpec,
    FileRef,
    Modality,
    UnifiedSample,
    Verification,
)


class NemotronAdapter(DataAdapter):
    """Adapt Nemotron cascade SWE format.

    Schema: instance_id, prompt, golden_patch, relevant_file_contents, source.
    """

    def adapt(self, raw: list[dict[str, Any]]) -> Iterator[UnifiedSample]:
        for row in raw:
            instance_id = row.get("instance_id", "")
            if not instance_id:
                continue

            prompt_text = row.get("prompt", "")
            golden_patch = row.get("golden_patch")

            # Build context files from relevant_file_contents
            context_files = []
            for fc in row.get("relevant_file_contents", []):
                if isinstance(fc, dict) and "file_path" in fc:
                    context_files.append(FileRef(
                        path=fc["file_path"],
                        content=fc.get("content", ""),
                    ))

            # Build environment spec
            env = EnvironmentSpec(
                initial_files=context_files[:10],  # Cap for sandbox
            )

            yield UnifiedSample(
                id=instance_id,
                prompt=[
                    {"role": "system", "content": "You are a software engineer. Fix the bug described in the issue."},
                    {"role": "user", "content": prompt_text},
                ],
                problem_statement=prompt_text,
                domain=Domain.SWE,
                modality=Modality.CODE,
                golden_solution=golden_patch,
                verification=Verification(golden_patch=golden_patch),
                environment=env,
                context_files=context_files,
                source=self.source.name,
                metadata={
                    "original_source": row.get("source", ""),
                    "original_prompt": row.get("original_prompt", ""),
                },
            )
