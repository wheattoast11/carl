# CARL workflows

Use `carl --help` and the selected subcommand's help for the installed release.

```bash
carl init
carl observe --help
carl train --help
carl eval --help
carl lab skill list
carl lab agent harnesses
```

The lightweight base package can measure coherence without training dependencies.
Training and optional backends require their documented extras and supported
hardware. Check availability before launching a job.

Use the actual objective and held-out eval to assess training. Coherence alone
does not establish task success. Preserve inputs, configuration, result artifacts,
and the named evaluation commands requested by the user.

CARL's delegation task IDs remain stable across its MCP and A2A observation
surfaces. Resume input only through the original live execution owner.
