"""Diagnosis problem templates."""
from __future__ import annotations
from carl_studio.data.templates import ProblemTemplate, TestTier

TEMPLATES = [
    ProblemTemplate(
        id="diag-wrong-file",
        name="Bug in wrong file",
        dims=["diagnosis"],
        description=(
            "Error message points to file A, but root cause is in file B. "
            "Requires reading both files and tracing the data flow."
        ),
        file_skeleton={
            "main.py": "Imports from utils.py, crashes with a clear error message",
            "utils.py": "Contains the actual bug (wrong default, off-by-one, type coercion)",
            "test_solution.py": "Graded test: T1=imports, T2=basic case, T3=edge case",
        },
        bug_pattern=(
            "Error manifests in main.py but root cause is in utils.py. "
            "A naive fix to main.py would pass T1 but fail T2/T3."
        ),
        tiers=[
            TestTier("exists", "Code imports without error", "from main import process"),
            TestTier("correct", "Basic case produces right output", "process(normal_input) == expected"),
            TestTier("robust", "Edge case that naive fix misses", "process(edge_input) == edge_expected"),
        ],
        variation_axes=[
            "bug type (off-by-one, None handling, type coercion, wrong default)",
            "domain (data processing, config parsing, API handling)",
        ],
    ),
    ProblemTemplate(
        id="diag-config-dependent",
        name="Config-dependent bug",
        dims=["diagnosis", "reasoning"],
        description=(
            "Code works in one config but fails in another. Must read config, "
            "understand flag behavior, then fix the conditional logic."
        ),
        file_skeleton={
            "main.py": "Uses config values, works when DEBUG=True fails when DEBUG=False",
            "config.py": "Settings dict with subtle behavior difference per mode",
            "test_solution.py": "Graded: T1=imports, T2=works in debug, T3=works in production, T4=both modes",
        },
        bug_pattern=(
            "Config has mode-dependent behavior. Code assumes DEBUG-mode defaults. "
            "In production mode, a key is missing/different."
        ),
        tiers=[
            TestTier("exists", "Module imports", "import main"),
            TestTier("correct", "Works in debug mode", "main.run(debug=True) succeeds"),
            TestTier("robust", "Works in production mode", "main.run(debug=False) succeeds"),
            TestTier("complete", "Both modes produce correct output", "outputs match expected"),
        ],
        variation_axes=[
            "config type (env vars, dict, dataclass)",
            "failure mode (missing key, wrong type, empty string vs None)",
        ],
    ),
    ProblemTemplate(
        id="diag-import-chain",
        name="Import chain bug",
        dims=["diagnosis", "planning"],
        description=(
            "Circular or broken import chain across 3+ files. Error traceback "
            "is misleading. Must trace imports to find the cycle or missing init."
        ),
        file_skeleton={
            "app.py": "Entry point, imports from services/",
            "services/__init__.py": "Re-exports from submodules",
            "services/auth.py": "Auth service, imports from models",
            "models.py": "Data models, may import from services (circular)",
            "test_solution.py": "T1=imports app, T2=auth works, T3=no circular import warning",
        },
        bug_pattern=(
            "Import cycle between services and models. Fixing requires restructuring "
            "one import to be lazy or moving shared types to a separate file."
        ),
        tiers=[
            TestTier("exists", "App imports without ImportError", "import app"),
            TestTier("correct", "Auth service works", "app.authenticate(user) returns token"),
            TestTier("robust", "No deprecation/circular import warnings", "no warnings in stderr"),
        ],
        variation_axes=[
            "cycle location (models<>services, utils<>core, handlers<>middleware)",
            "fix strategy (lazy import, extract shared types, dependency inversion)",
        ],
    ),
]
