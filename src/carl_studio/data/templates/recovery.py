"""Recovery and reasoning problem templates."""
from __future__ import annotations
from carl_studio.data.templates import ProblemTemplate, TestTier

RECOVERY_TEMPLATES = [
    ProblemTemplate(
        id="recover-timeout-trap",
        name="Timeout trap — naive approach too slow",
        dims=["recovery", "reasoning"],
        description=(
            "Task has an obvious Python solution that exceeds the 10s sandbox timeout. "
            "Must recognize the timeout, then use a more efficient approach."
        ),
        file_skeleton={
            "data.txt": "Large dataset (generated, ~50K lines)",
            "test_solution.py": "T1=output file exists, T2=correct result, T3=completes under 5s",
        },
        bug_pattern=(
            "Naive Python (read all lines, sort in memory) works for small data but "
            "times out on 50K lines. Shell commands (sort, awk, grep) are 10-100x faster."
        ),
        tiers=[
            TestTier("exists", "Output file was created", "os.path.exists('result.txt')"),
            TestTier("correct", "Output matches expected", "result == expected_first_10_lines"),
            TestTier("efficient", "Completed within 5 seconds", "elapsed < 5.0"),
        ],
        variation_axes=[
            "operation (sort, filter, aggregate, transform)",
            "data type (CSV, TSV, JSON lines, plain text)",
        ],
    ),
    ProblemTemplate(
        id="recover-edge-case-trap",
        name="Edge case trap — first attempt misses boundary",
        dims=["recovery", "diagnosis"],
        description=(
            "Function works for normal inputs but fails on edge cases. "
            "First execution succeeds, but test harness catches the edge case."
        ),
        file_skeleton={
            "solution.py": "Stub or partial implementation",
            "test_solution.py": "T1=exists, T2=normal case, T3=empty input, T4=boundary value",
        },
        bug_pattern=(
            "Normal cases work, but empty list, zero, negative, None, or unicode "
            "input causes crash or wrong output. Model must iterate after seeing T3/T4 fail."
        ),
        tiers=[
            TestTier("exists", "Function exists and is callable", "callable(solution.func)"),
            TestTier("correct", "Normal inputs work", "func([1,2,3]) == expected"),
            TestTier("robust_empty", "Empty input handled", "func([]) == empty_expected"),
            TestTier("robust_boundary", "Boundary values handled", "func([MAX_INT]) == boundary_expected"),
        ],
        variation_axes=[
            "function type (sort, search, parse, validate, transform)",
            "edge case (empty, None, negative, unicode, overflow)",
        ],
    ),
]

REASONING_TEMPLATES = [
    ProblemTemplate(
        id="reason-profile-optimize",
        name="Profile then optimize",
        dims=["reasoning", "recovery"],
        description=(
            "Function is correct but slow. Must profile to find bottleneck, "
            "then optimize the specific hot path. Blind optimization fails."
        ),
        file_skeleton={
            "solution.py": "Working but slow implementation with non-obvious bottleneck",
            "benchmark.py": "Timing harness that measures performance",
            "test_solution.py": "T1=correct output, T2=matches reference, T3=under time threshold",
        },
        bug_pattern=(
            "Function has O(n^2) inner loop that looks innocent. The obvious optimization "
            "(caching) doesn't help because the bottleneck is elsewhere (e.g., string "
            "concatenation in a loop, repeated list scanning). Must profile first."
        ),
        tiers=[
            TestTier("correct", "Output matches reference", "func(input) == reference_output"),
            TestTier("complete", "All test cases pass", "all(func(tc) == expected for tc in cases)"),
            TestTier("efficient", "10x faster than original", "new_time < original_time / 10"),
        ],
        variation_axes=[
            "bottleneck type (string concat, nested loop, repeated lookup, unnecessary copy)",
            "data structure (list, dict, string, nested)",
        ],
    ),
    ProblemTemplate(
        id="reason-analyze-decide-act",
        name="Analyze data then decide approach",
        dims=["reasoning", "planning"],
        description=(
            "Given a data file, must analyze its structure/content before deciding "
            "how to process it. Approach depends on what's in the data."
        ),
        file_skeleton={
            "data.json": "Structured data with non-obvious properties",
            "test_solution.py": "T1=output exists, T2=correct summary, T3=handles all record types",
        },
        bug_pattern=(
            "Data has mixed record types (some have field X, some have field Y). "
            "A uniform processing approach misses one type. Must read and analyze "
            "the data to discover the heterogeneity before writing the processor."
        ),
        tiers=[
            TestTier("exists", "Output file created", "os.path.exists('summary.json')"),
            TestTier("correct", "Summary counts match", "summary['total'] == expected_total"),
            TestTier("robust", "All record types handled", "all types present in summary['by_type']"),
        ],
        variation_axes=[
            "data heterogeneity (mixed types, optional fields, nested vs flat)",
            "output format (summary stats, filtered subset, transformed records)",
        ],
    ),
]

TEMPLATES = RECOVERY_TEMPLATES + REASONING_TEMPLATES
