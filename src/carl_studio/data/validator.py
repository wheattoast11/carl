"""Three-way sample validation and SCORE:X/Y parsing."""
from __future__ import annotations
import os
import re
import shutil
import subprocess
import tempfile
from pydantic import BaseModel

__all__ = ["parse_score", "validate_sample", "ValidationResult"]

_SCORE_RE = re.compile(r"SCORE:(\d+)/(\d+)")


def parse_score(text: str) -> float | None:
    if not text:
        return None
    matches = _SCORE_RE.findall(text)
    if not matches:
        return None
    num, den = int(matches[-1][0]), int(matches[-1][1])
    return num / den if den > 0 else None


class ValidationResult(BaseModel):
    red_score: float
    green_score: float
    antigame_score: float
    valid: bool
    errors: list[str]


def _run_in_sandbox(files: dict[str, str], test_file: str = "test_solution.py") -> float:
    workdir = tempfile.mkdtemp(prefix="carl_validate_")
    try:
        for fname, content in files.items():
            fpath = os.path.join(workdir, fname)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "w") as f:
                f.write(content)
        result = subprocess.run(
            ["python", test_file],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=workdir,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        score = parse_score(result.stdout + result.stderr)
        return score if score is not None else 0.0
    except (subprocess.TimeoutExpired, Exception):
        return 0.0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _apply_fix(
    files: dict[str, str],
    fix: str,
    target: str | None,
) -> dict[str, str]:
    patched = dict(files)
    if target is None:
        for fname in files:
            if fname.endswith(".py") and "test_" not in fname:
                target = fname
                break
    if target:
        patched[target] = fix
    return patched


def validate_sample(sample: dict) -> ValidationResult:
    files: dict[str, str] = sample.get("initial_files", {})
    fix_target: str | None = sample.get("fix_target")
    intended_fix: str = sample.get("intended_fix", "")
    trivial_bypass: str = sample.get("trivial_bypass", "")
    errors: list[str] = []

    if "test_solution.py" not in files:
        return ValidationResult(
            red_score=0,
            green_score=0,
            antigame_score=0,
            valid=False,
            errors=["No test_solution.py"],
        )

    red_score = _run_in_sandbox(files)

    if intended_fix:
        green_score = _run_in_sandbox(_apply_fix(files, intended_fix, fix_target))
    else:
        green_score = 0.0
        errors.append("No intended_fix")

    if trivial_bypass:
        antigame_score = _run_in_sandbox(_apply_fix(files, trivial_bypass, fix_target))
    else:
        antigame_score = 0.0
        errors.append("No trivial_bypass")

    valid = (
        red_score < 0.5
        and green_score >= 0.9
        and antigame_score < 0.5
        and not errors
    )
    return ValidationResult(
        red_score=red_score,
        green_score=green_score,
        antigame_score=antigame_score,
        valid=valid,
        errors=errors,
    )
