"""Standalone encoder imports remain independent of studio and heavy workers."""

from __future__ import annotations

import subprocess
import sys


def test_standalone_lightweight_imports() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            "import sys; from carl_encoders.types import SemanticInput, GEMMA_REVISION; from carl_encoders.release import EncoderRelease; from carl_encoders.worker import metadata; binding=metadata(); assert binding['dependencies']['carl.encoder.worker_sha256']; assert 'carl_studio' not in sys.modules; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
