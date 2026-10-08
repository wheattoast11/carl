---
name: carl
description: Use CARL to observe coherence, validate training configuration, train and evaluate models, run CARL skills, or delegate bounded work to Codex, Claude Code, and OpenCode.
---

# CARL

Use the installed CARL MCP tools when available. Use the CARL CLI otherwise.
CARL means Coherence-Aware Reinforcement Learning.

1. Inspect the current project and the requested outcome.
2. Use `validate_config` before starting training.
3. Use observe and eval to measure the result the user requested.
4. Use `list_skills` and `run_skill` for existing CARL workflows.
5. Use `list_agent_harnesses` and `delegate_agent` when another native agent should do part of the work.

Local plugin access is FREE. Scheduled autonomy, paid services, publication, and
private-runtime operations retain their own authorization and entitlement checks.
Do not start paid training, transfer data, publish, or broaden a workspace grant
without the user's authorization.

## Delegation

Pass the chosen host, instruction, workspace, and a stable request ID.
Read-only is the default. Request editing only when the user authorized it.
Poll `tasks_get`; answer the exact pending request with `tasks_reply`.
Read its transient details with `read_agent_input` before answering.
Cancel with `tasks_cancel`. Cancellation succeeds after execution stops.
Retrieve the final artifact with `read_agent_result`.

CARL owns execution, scope, limits, and task state. Child agents cannot delegate
again through CARL. Do not change a child policy or bypass permission checks.
Keep instruction and tool content out of operational traces. Return the useful
answer and user-requested artifacts.

## CLI

```bash
carl plugin doctor
carl lab agent harnesses
carl lab agent delegate codex < instruction.txt
carl lab agent status TASK_ID
carl lab agent result TASK_ID
carl lab agent cancel TASK_ID
carl lab skill list
```

Install from a CARL checkout with `./scripts/install-plugin`.
Update with `carl plugin update`; remove only CARL's owned entries with
`carl plugin uninstall`.

See [workflows](../../../skills/carl/references/workflows.md) for training and verification commands.
