# A2A (Agent-to-Agent)

Agent-to-agent protocol for dispatching tasks between CARL instances. Loosely based on the Google A2A spec, adapted for CARL training workflows.

## Key Abstractions

- `CARLAgentCard` -- Capability manifest describing what a CARL agent can do (name, version, tier, capabilities, skills, endpoint). Built dynamically from current settings and installed skills.
- `A2ATask` -- Pydantic model for a unit of work. Status lifecycle: PENDING -> RUNNING -> DONE/FAILED/CANCELLED. Immutable transitions via `mark_running()`, `mark_done()`, `mark_failed()`.
- `A2AMessage` -- Typed message attached to a task (progress updates, artifacts).
- `LocalBus` -- SQLite-backed message bus with WAL mode. Offline-first, no external dependencies. Stores tasks and messages with priority-based dispatch.
- `SupabaseBus` -- Cloud transport stub for PAID tier (not yet implemented).
- `spec.py` -- JSON-RPC serialization helpers: `agent_card_to_spec()`, `task_to_jsonrpc_result()`, `wrap_jsonrpc_response()`.

## Architecture

The A2A layer sits between the CLI/MCP surface and the skill runner. External agents discover capabilities via `CARLAgentCard`, dispatch `A2ATask` objects through a bus, and the skill runner executes them. Local transport uses SQLite; cloud transport will use Supabase Realtime.

## Usage

```python
from carl_studio.a2a import CARLAgentCard, LocalBus, A2ATask

card = CARLAgentCard.current()
bus = LocalBus()
task = bus.submit("observer", inputs={"model": "my-model"})
```
