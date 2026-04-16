"""Contract CLI — ``carl camp contract``.

Service contract management and witnessing.
Register in wiring.py via camp_app.add_typer(contract_app, name='contract').
"""

from __future__ import annotations

import json

import typer

contract_app = typer.Typer(
    name="contract",
    help="Service contract management and witnessing.",
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
