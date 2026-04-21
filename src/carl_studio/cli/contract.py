"""Contract CLI — ``carl camp contract`` and ``carl contract``.

Service contract management, witnessing, and the constitutional ledger.
Register in wiring.py via camp_app.add_typer(contract_app, name='contract').
The same ``contract_app`` is also mounted at the top level by cli/__init__.py
so ``carl contract constitution [...]`` works without the ``camp`` prefix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

contract_app = typer.Typer(
    name="contract",
    help="Service contract management, witnessing, and constitutional ledger.",
    no_args_is_help=True,
)


@contract_app.command(name="show")
def contract_show(
    contract_id: str = typer.Argument("", help="Contract ID (blank for latest)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show contract details and witness status."""
    from carl_studio.console import get_console
    from carl_studio.contract import ContractWitness

    c = get_console()
    witness = ContractWitness()

    if contract_id:
        contract = witness.get_contract(contract_id)
    else:
        contracts = witness.list_contracts(limit=1)
        contract = contracts[0] if contracts else None

    if contract is None:
        c.info("No contracts found.")
        raise typer.Exit(0)

    if json_output:
        typer.echo(json.dumps(contract.model_dump(), indent=2))
        raise typer.Exit(0)

    c.header("Contract", contract.id[:12])
    c.kv("ID", contract.id, key_width=16)
    c.kv("Status", contract.status, key_width=16)
    c.kv("Parties", ", ".join(contract.parties) or "(none)", key_width=16)
    c.kv("Terms URL", contract.terms_url, key_width=16)
    c.kv("Terms hash", contract.terms_hash[:16] + "..." if contract.terms_hash else "(none)", key_width=16)
    c.kv("Signed at", contract.signed_at or "(not signed)", key_width=16)
    c.kv("Witness", contract.witness_hash[:16] + "..." if contract.witness_hash else "(none)", key_width=16)
    c.kv("Chain", contract.chain or "(local only)", key_width=16)


@contract_app.command(name="sign")
def contract_sign(
    terms_url: str = typer.Argument(..., help="URL of terms to sign"),
    party: list[str] = typer.Option([], "--party", "-p", help="Party identifier (repeatable)"),
) -> None:
    """Sign a service contract by fetching and witnessing terms."""
    from carl_studio.console import get_console
    from carl_studio.contract import ContractError, ContractWitness, ServiceContract

    c = get_console()
    witness = ContractWitness()

    try:
        terms_content = witness.fetch_terms(terms_url)
    except ContractError as exc:
        c.error(str(exc))
        raise typer.Exit(1)

    terms_hash = witness.hash_terms(terms_content)
    contract = ServiceContract(
        parties=party or ["self"],
        terms_hash=terms_hash,
        terms_url=terms_url,
    )

    try:
        envelope = witness.sign(contract)
    except ContractError as exc:
        c.error(str(exc))
        raise typer.Exit(1)

    c.ok(f"Contract witnessed: {contract.id[:12]}...")
    c.kv("Terms hash", terms_hash[:16] + "...", key_width=16)
    c.kv("Witness hash", envelope.witness_hash[:16] + "...", key_width=16)
    c.kv("Witnessed at", envelope.witnessed_at, key_width=16)


@contract_app.command(name="list")
def contract_list(
    limit: int = typer.Option(10, "--limit", help="Number of contracts to show"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List locally stored contracts."""
    from carl_studio.console import get_console
    from carl_studio.contract import ContractWitness

    c = get_console()
    witness = ContractWitness()
    contracts = witness.list_contracts(limit=limit)

    if json_output:
        typer.echo(json.dumps([ct.model_dump() for ct in contracts], indent=2))
        raise typer.Exit(0)

    if not contracts:
        c.info("No contracts stored locally.")
        return

    table = c.make_table("ID", "Status", "Parties", "Terms URL", title="Contracts")
    for ct in contracts:
        table.add_row(
            ct.id[:12] + "...",
            ct.status,
            ", ".join(ct.parties)[:30],
            ct.terms_url[:40],
        )
    c.print(table)


@contract_app.command(name="verify")
def contract_verify(
    contract_id: str = typer.Argument(..., help="Contract ID to verify"),
) -> None:
    """Verify a contract's witness hash."""
    from carl_studio.console import get_console
    from carl_studio.contract import ContractWitness

    c = get_console()
    witness = ContractWitness()
    envelope = witness.get_envelope(contract_id)

    if envelope is None:
        c.error(f"Contract '{contract_id}' not found or has no witness envelope.")
        raise typer.Exit(1)

    valid = witness.verify(envelope)

    if valid:
        c.ok(f"Witness hash verified for {contract_id[:12]}...")
    else:
        c.error(f"Witness hash INVALID for {contract_id[:12]}...")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Constitutional ledger — ``carl contract constitution [genesis|verify|evaluate]``
# ---------------------------------------------------------------------------


@contract_app.command(name="constitution")
def constitution(
    action: str = typer.Argument(
        ..., help="Operation: genesis | verify | evaluate | status"
    ),
    policy_path: Optional[Path] = typer.Option(
        None,
        "--policy-path",
        help="Path to a ConstitutionalPolicy JSON (genesis/evaluate).",
    ),
    ledger_root: Optional[Path] = typer.Option(
        None,
        "--ledger-root",
        help="Directory for the ledger (default: ~/.carl/constitutional).",
    ),
    action_json: Optional[str] = typer.Option(
        None,
        "--action-json",
        help="Inline JSON dict describing the action to evaluate.",
    ),
    threshold: float = typer.Option(
        0.0,
        "--threshold",
        help="Threshold tau when synthesizing a default policy at genesis.",
    ),
) -> None:
    """Constitutional ledger operations.

    Subactions:
      - ``genesis``  — install a constitutional policy + write genesis block.
      - ``verify``   — walk the chain, validate hashes + ed25519 signatures.
      - ``evaluate`` — run the policy against a single action without appending.
      - ``status``   — print height, head hash, and policy id.
    """
    from carl_studio.console import get_console
    from carl_studio.fsm_ledger import (
        build_default_policy,
        default_ledger_root,
    )

    c = get_console()
    root = Path(ledger_root) if ledger_root else default_ledger_root()

    try:
        from carl_core.constitutional import (
            ConstitutionalLedger,
            ConstitutionalPolicy,
            encode_action_features,
        )
    except ImportError as exc:
        c.error(str(exc))
        raise typer.Exit(2)

    act = (action or "").strip().lower()
    if act not in {"genesis", "verify", "evaluate", "status"}:
        c.error(f"unknown action '{action}' — expected genesis|verify|evaluate|status")
        raise typer.Exit(2)

    ledger = ConstitutionalLedger(root=root)

    if act == "genesis":
        if policy_path is not None:
            try:
                policy = ConstitutionalPolicy.load(Path(policy_path))
            except Exception as exc:
                c.error(f"failed to load policy: {exc}")
                raise typer.Exit(2)
        else:
            policy = build_default_policy(threshold=threshold)
        try:
            block = ledger.genesis(policy)
        except ImportError as exc:
            c.error(str(exc))
            raise typer.Exit(2)
        except Exception as exc:
            c.error(f"genesis failed: {exc}")
            raise typer.Exit(1)
        c.ok(f"Genesis block written at {root}")
        c.kv("policy_id", policy.policy_id, key_width=16)
        c.kv("threshold", f"{policy.threshold:.6f}", key_width=16)
        c.kv("block_hash", block.block_hash(), key_width=16)
        return

    if act == "verify":
        try:
            ok, bad = ledger.verify_chain()
        except ImportError as exc:
            c.error(str(exc))
            raise typer.Exit(2)
        if ok:
            c.ok(f"chain valid (height={ledger.height()})")
            return
        c.error(f"chain INVALID — {len(bad)} bad blocks: {bad[:10]}{'...' if len(bad) > 10 else ''}")
        raise typer.Exit(1)

    if act == "status":
        head = ledger.head()
        if head is None:
            c.info("no genesis yet — run: carl contract constitution genesis")
            return
        try:
            pid = ledger.policy().policy_id
        except Exception:
            pid = "(policy missing)"
        c.header("Constitutional ledger", str(root))
        c.kv("height", str(ledger.height()), key_width=16)
        c.kv("policy_id", pid, key_width=16)
        c.kv("head_block", str(head.block_id), key_width=16)
        c.kv("head_hash", head.block_hash(), key_width=16)
        return

    # evaluate
    if not action_json:
        c.error("--action-json is required for 'evaluate'")
        raise typer.Exit(2)
    try:
        act_dict = json.loads(action_json)
    except json.JSONDecodeError as exc:
        c.error(f"bad --action-json: {exc}")
        raise typer.Exit(2)
    if policy_path is not None:
        try:
            policy = ConstitutionalPolicy.load(Path(policy_path))
        except Exception as exc:
            c.error(f"failed to load policy: {exc}")
            raise typer.Exit(2)
    else:
        try:
            policy = ledger.policy()
        except Exception as exc:
            c.error(str(exc))
            raise typer.Exit(2)
    features = encode_action_features(act_dict)
    allowed, score = policy.evaluate(features)
    verdict = "ALLOW" if allowed else "DENY"
    c.kv("verdict", verdict, key_width=16)
    c.kv("score", f"{score:.6f}", key_width=16)
    c.kv("threshold", f"{policy.threshold:.6f}", key_width=16)
    c.kv("policy_id", policy.policy_id[:16] + "...", key_width=16)
    if not allowed:
        raise typer.Exit(1)
