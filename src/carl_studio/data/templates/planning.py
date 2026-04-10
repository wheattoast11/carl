"""Planning problem templates."""
from __future__ import annotations
from carl_studio.data.templates import ProblemTemplate, TestTier

TEMPLATES = [
    ProblemTemplate(
        id="plan-dependency-order",
        name="Modification order matters",
        dims=["planning"],
        description=(
            "Multiple files need changes, but modifying them in wrong order "
            "breaks the project. Must read all files first to understand dependencies."
        ),
        file_skeleton={
            "base.py": "Base class with interface contract",
            "impl_a.py": "Implementation A, extends base",
            "impl_b.py": "Implementation B, extends base, depends on A's output format",
            "runner.py": "Orchestrates A then B",
            "test_solution.py": "T1=imports, T2=A works, T3=B works, T4=pipeline works",
        },
        bug_pattern=(
            "Task is to add a new parameter. Must update base interface first, "
            "then A, then B (because B depends on A's output). Wrong order -> AttributeError."
        ),
        tiers=[
            TestTier("exists", "All modules import", "import runner"),
            TestTier("correct", "Impl A handles new param", "a.process(data, new_param=X) works"),
            TestTier("robust", "Impl B handles A's new output", "b.process(a_output) works"),
            TestTier("complete", "Full pipeline end-to-end", "runner.run(data, new_param=X) == expected"),
        ],
        variation_axes=[
            "change type (add parameter, change return type, add validation)",
            "dependency depth (2-file chain, 3-file chain, diamond)",
        ],
    ),
    ProblemTemplate(
        id="plan-monkey-patch-trap",
        name="Monkey-patching trap",
        dims=["planning", "diagnosis"],
        description=(
            "One file monkey-patches another at import time. Naive modification "
            "to the patched file gets overwritten. Must read the patcher first."
        ),
        file_skeleton={
            "core.py": "Core functions that get monkey-patched",
            "patches.py": "Monkey-patches core at import time (import side effect)",
            "main.py": "Imports patches (triggering the patch), then uses core",
            "test_solution.py": "T1=imports, T2=basic works, T3=patched behavior correct",
        },
        bug_pattern=(
            "core.py has a bug, but patches.py overwrites the buggy function at import. "
            "Fixing core.py alone doesn't work because patches.py re-overwrites it. "
            "Must fix in patches.py or remove the monkey-patch."
        ),
        tiers=[
            TestTier("exists", "Main imports without error", "import main"),
            TestTier("correct", "Core function returns right value", "core.func(input) == expected"),
            TestTier("robust", "Fix persists after patches import", "import patches; core.func(input) == expected"),
        ],
        variation_axes=[
            "patch mechanism (direct assignment, decorator, metaclass)",
            "domain (logging, validation, serialization)",
        ],
    ),
]
