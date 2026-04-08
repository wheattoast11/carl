"""Tests for CARL environment protocol, registry, validation, and builtins."""

import os
import sys

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from carl_studio.environments.protocol import BaseEnvironment, EnvironmentLane, EnvironmentSpec
from carl_studio.environments.registry import register_environment, get_environment, list_environments, clear_registry
from carl_studio.environments.validation import validate_environment

# =========================================================================
# Test helpers
# =========================================================================

_pass = 0
_fail = 0


def assert_eq(a, b, msg=""):
    global _pass, _fail
    if a == b:
        _pass += 1
    else:
        _fail += 1
        print(f"  FAIL: {msg} — expected {b!r}, got {a!r}")


def assert_true(val, msg=""):
    global _pass, _fail
    if val:
        _pass += 1
    else:
        _fail += 1
        print(f"  FAIL: {msg} — expected truthy, got {val!r}")


def assert_false(val, msg=""):
    global _pass, _fail
    if not val:
        _pass += 1
    else:
        _fail += 1
        print(f"  FAIL: {msg} — expected falsy, got {val!r}")


def assert_in(item, container, msg=""):
    global _pass, _fail
    if item in container:
        _pass += 1
    else:
        _fail += 1
        print(f"  FAIL: {msg} — {item!r} not in {container!r}")


def assert_raises(exc_type, fn, msg=""):
    global _pass, _fail
    try:
        fn()
        _fail += 1
        print(f"  FAIL: {msg} — no exception raised")
    except exc_type:
        _pass += 1
    except Exception as e:
        _fail += 1
        print(f"  FAIL: {msg} — expected {exc_type.__name__}, got {type(e).__name__}: {e}")


# =========================================================================
# EnvironmentLane
# =========================================================================

print("Testing EnvironmentLane...")
assert_eq(EnvironmentLane.CODE, "code", "CODE value")
assert_eq(EnvironmentLane.QUERY, "query", "QUERY value")
assert_eq(EnvironmentLane.RETRIEVAL, "retrieval", "RETRIEVAL value")
assert_eq(EnvironmentLane.ROUTING, "routing", "ROUTING value")
assert_eq(EnvironmentLane.INFRA, "infra", "INFRA value")
assert_eq(EnvironmentLane.VISUAL, "visual", "VISUAL value")
assert_eq(len(EnvironmentLane), 6, "6 lanes (MECE)")

# =========================================================================
# EnvironmentSpec
# =========================================================================

print("Testing EnvironmentSpec...")
spec = EnvironmentSpec(
    lane=EnvironmentLane.CODE,
    name="test-env",
    tools=("read", "write"),
    max_turns=5,
)
assert_eq(spec.lane, EnvironmentLane.CODE, "spec lane")
assert_eq(spec.name, "test-env", "spec name")
assert_eq(spec.tools, ("read", "write"), "spec tools")
assert_eq(spec.max_turns, 5, "spec max_turns")
assert_eq(spec.reward_type, "binary", "spec default reward_type")
assert_eq(spec.multimodal, False, "spec default multimodal")

# Frozen
try:
    spec.name = "changed"
    _fail += 1
    print("  FAIL: spec should be frozen")
except AttributeError:
    _pass += 1

# =========================================================================
# Registry
# =========================================================================

print("Testing Registry...")

clear_registry()
assert_eq(list_environments(), [], "empty after clear")

# Register a test env
class _TestEnv(BaseEnvironment):
    spec = EnvironmentSpec(lane=EnvironmentLane.CODE, name="test-reg", tools=("do_thing",))
    def reset(self, **kwargs):
        super().reset(**kwargs)
        return None
    def do_thing(self, x: str) -> str:
        """Do a thing.

        Args:
            x: The thing.
        """
        return x

register_environment(_TestEnv)
assert_eq(len(list_environments()), 1, "one registered")
assert_eq(get_environment("test-reg"), _TestEnv, "lookup works")

# Duplicate raises
assert_raises(ValueError, lambda: register_environment(_TestEnv), "duplicate raises ValueError")

# Unknown raises
assert_raises(KeyError, lambda: get_environment("nonexistent"), "unknown raises KeyError")

# Filter by lane
assert_eq(len(list_environments(lane=EnvironmentLane.CODE)), 1, "filter CODE")
assert_eq(len(list_environments(lane=EnvironmentLane.QUERY)), 0, "filter QUERY empty")

clear_registry()

# =========================================================================
# Validation
# =========================================================================

print("Testing Validation...")

# Valid env
errors = validate_environment(_TestEnv)
assert_eq(errors, [], "valid env has no errors")

# Missing spec
class _NoSpec(BaseEnvironment):
    def reset(self, **kwargs): return None
    def tool(self, x: str) -> str:
        """T. Args: x: thing."""
        return x

errors = validate_environment(_NoSpec)
assert_true(any("spec" in e.lower() for e in errors), "missing spec detected")

# Init with required args
class _BadInit(BaseEnvironment):
    spec = EnvironmentSpec(lane=EnvironmentLane.CODE, name="bad-init", tools=("t",))
    def __init__(self, required_arg):
        super().__init__()
    def reset(self, **kwargs): return None
    def t(self, x: str) -> str:
        """T. Args: x: thing."""
        return x

errors = validate_environment(_BadInit)
assert_true(any("init" in e.lower() or "__init__" in e for e in errors), "required init arg detected")

# No tools
class _NoTools(BaseEnvironment):
    spec = EnvironmentSpec(lane=EnvironmentLane.CODE, name="no-tools", tools=())
    def reset(self, **kwargs): return None

errors = validate_environment(_NoTools)
assert_true(any("tool" in e.lower() for e in errors), "no tools detected")

# Tool missing docstring
class _NoDoc(BaseEnvironment):
    spec = EnvironmentSpec(lane=EnvironmentLane.CODE, name="no-doc", tools=("bad_tool",))
    def reset(self, **kwargs): return None
    def bad_tool(self, x: str) -> str:
        return x

errors = validate_environment(_NoDoc)
assert_true(any("docstring" in e.lower() for e in errors), "missing docstring detected")

# Tool missing type hint
class _NoHint(BaseEnvironment):
    spec = EnvironmentSpec(lane=EnvironmentLane.CODE, name="no-hint", tools=("untyped",))
    def reset(self, **kwargs): return None
    def untyped(self, x) -> str:
        """Do thing. Args: x: thing."""
        return str(x)

errors = validate_environment(_NoHint)
assert_true(any("type hint" in e.lower() for e in errors), "missing type hint detected")

# Spec/method mismatch
class _Mismatch(BaseEnvironment):
    spec = EnvironmentSpec(lane=EnvironmentLane.CODE, name="mismatch", tools=("declared_but_missing",))
    def reset(self, **kwargs): return None
    def actual_method(self, x: str) -> str:
        """Do thing. Args: x: thing."""
        return x

errors = validate_environment(_Mismatch)
assert_true(any("not found" in e.lower() or "not declared" in e.lower() for e in errors), "mismatch detected")

# =========================================================================
# CodingSandboxEnv
# =========================================================================

print("Testing CodingSandboxEnv...")
from carl_studio.environments.builtins.code_sandbox import CodingSandboxEnv
errors = validate_environment(CodingSandboxEnv)
assert_eq(errors, [], f"CodingSandboxEnv valid: {errors}")

# Spec
assert_eq(CodingSandboxEnv.spec.lane, EnvironmentLane.CODE, "code lane")
assert_eq(CodingSandboxEnv.spec.name, "python-sandbox", "name")
assert_eq(len(CodingSandboxEnv.spec.tools), 4, "4 tools")

# Reset + execute
env = CodingSandboxEnv()
obs = env.reset(task_description="Test task")
assert_eq(obs, "Test task", "reset returns task")
assert_eq(env.reward, 0.0, "initial reward 0")
assert_eq(env.turn_count, 0, "initial turn count 0")

# Write + execute
env.write_file("test.py", "print('hello')")
assert_eq(env.turn_count, 1, "turn count after write")

result = env.execute_code("print('hello')")
assert_in("hello", result, "execution output")
assert_eq(env.reward, 1.0, "reward after success")
assert_eq(env.turn_count, 2, "turn count after execute")
assert_true(env._execution_succeeded, "execution succeeded flag")

# Failed execution
result = env.execute_code("raise ValueError('oops')")
assert_in("Error", result, "error in output")
assert_eq(env.reward, 0.0, "reward after failure")

# Read file
env.write_file("data.txt", "content here")
result = env.read_file("data.txt")
assert_eq(result, "content here", "read file")

# Read nonexistent
result = env.read_file("nope.txt")
assert_in("Error", result, "read nonexistent")

# Path escape
result = env.read_file("../../etc/passwd")
assert_in("Error", result, "path escape blocked")

# History
assert_true(len(env.history) > 0, "history recorded")

# Cleanup
del env

# =========================================================================
# SQLSandboxEnv
# =========================================================================

print("Testing SQLSandboxEnv...")
from carl_studio.environments.builtins.sql_sandbox import SQLSandboxEnv

# Already registered via @register_environment decorator on import
errors = validate_environment(SQLSandboxEnv)
assert_eq(errors, [], f"SQLSandboxEnv valid: {errors}")

assert_eq(SQLSandboxEnv.spec.lane, EnvironmentLane.QUERY, "query lane")
assert_eq(SQLSandboxEnv.spec.name, "sqlite-sandbox", "name")

# Reset with schema
env = SQLSandboxEnv()
obs = env.reset(
    task_description="Count users",
    schema_ddl="CREATE TABLE users (id INTEGER, name TEXT); INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie');",
    expected_result="3",
)
assert_eq(obs, "Count users", "sql reset obs")

# Describe schema
schema = env.describe_schema()
assert_in("users", schema, "schema has users table")
assert_in("3 rows", schema, "schema shows row count")

# Run query
result = env.run_query("SELECT COUNT(*) as cnt FROM users")
assert_in("3", result, "count query result")

# Reject non-SELECT
result = env.run_query("DROP TABLE users")
assert_in("Error", result, "non-SELECT blocked")

# Submit correct answer
env.submit_answer("3")
assert_eq(env.reward, 1.0, "correct answer reward")
assert_true(env.done, "done after submit")

# Submit wrong answer
env2 = SQLSandboxEnv()
env2.reset(task_description="Count", schema_ddl="CREATE TABLE t (x INT);", expected_result="0")
env2.submit_answer("42")
assert_eq(env2.reward, 0.0, "wrong answer reward")

# Partial match
env3 = SQLSandboxEnv()
env3.reset(task_description="Find name", schema_ddl="CREATE TABLE t (x INT);", expected_result="Alice and Bob")
env3.submit_answer("The answer is Alice")
assert_eq(env3.reward, 0.5, "partial match reward")

# History
assert_true(env.turn_count > 0, "sql turn count")

# =========================================================================
# Cross-lane registry
# =========================================================================

print("Testing cross-lane registry...")
all_envs = list_environments()
assert_eq(len(all_envs), 2, "2 total environments")

code = list_environments(lane=EnvironmentLane.CODE)
query = list_environments(lane=EnvironmentLane.QUERY)
assert_eq(len(code), 1, "1 code env")
assert_eq(len(query), 1, "1 query env")

clear_registry()

# =========================================================================
# Summary
# =========================================================================

print(f"\n{'='*40}")
print(f"  {_pass} passed, {_fail} failed")
print(f"{'='*40}")
if _fail > 0:
    sys.exit(1)
