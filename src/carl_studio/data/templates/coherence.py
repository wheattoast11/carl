"""Coherence problem templates."""
from __future__ import annotations
from carl_studio.data.templates import ProblemTemplate, TestTier

TEMPLATES = [
    ProblemTemplate(
        id="cohere-standardize",
        name="Standardize pattern across modules",
        dims=["coherence"],
        description=(
            "5-6 files each handle errors/logging/config differently. "
            "Must read all, understand each approach, then standardize to a common pattern."
        ),
        file_skeleton={
            "errors.py": "Standard error class (AppError) — the target pattern",
            "api.py": "Uses raw exceptions",
            "db.py": "Uses error codes (ints)",
            "auth.py": "Uses string error messages",
            "cache.py": "Uses boolean success flags",
            "worker.py": "Silently swallows errors",
            "test_solution.py": "T1=all import, T2=each raises AppError, T3=error messages preserved, T4=no silent swallowing",
        },
        bug_pattern=(
            "Each module handles errors differently. Task is to make all use AppError. "
            "Requires understanding what each module currently does to preserve semantics."
        ),
        tiers=[
            TestTier("exists", "All modules import", "import api, db, auth, cache, worker"),
            TestTier("correct", "Each raises AppError on failure", "try: mod.fail_op(); except AppError: pass"),
            TestTier("robust", "Error messages are meaningful", "str(error) contains context"),
            TestTier("complete", "No silent error swallowing", "worker.process(bad) raises, not returns None"),
        ],
        variation_axes=[
            "pattern to standardize (errors, logging, config access, validation)",
            "module count (4, 5, 6)",
        ],
    ),
]
