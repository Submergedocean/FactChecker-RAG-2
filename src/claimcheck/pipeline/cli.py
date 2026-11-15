"""`claimcheck pipeline` commands.

claimcheck pipeline verify "<claim>" [--json] [--no-cache] [--evidence]
"""

from __future__ import annotations

import textwrap
import time
from typing import Annotated

import typer

app = typer.Typer(
    help="Claim extraction and verification pipeline.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Claim extraction and verification pipeline."""


def _wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(
        " ".join(text.split()), width=96, initial_indent=indent, subsequent_indent=indent
    )


@app.command()
def verify(
    text: Annotated[str, typer.Argument(help="The claim (10-500 characters).")],
    as_json: Annotated[bool, typer.Option("--json", help="Print the API response JSON.")] = False,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Run afresh: skip the stored result and the LLM cache."),
    ] = False,
    evidence: Annotated[
        bool, typer.Option("--evidence", help="Also list the retrieved passages.")
    ] = False,
    variant: Annotated[
        str, typer.Option("--variant", help="table_aware or plain_text.")
    ] = "table_aware",
) -> None:
    """Verify TEXT and print the verdict, reasoning and citations of each claim."""
    from claimcheck.llm.client import BudgetExhausted, LLMUnavailable
    from claimcheck.pipeline.service import ClaimCheckService
    from claimcheck.textnorm import InputGuardError

    t_load = time.perf_counter()
    service = ClaimCheckService.from_settings(variant=variant)
    load_s = time.perf_counter() - t_load
    try:
        response = service.verify(text, use_cache=not no_cache)
    except InputGuardError as exc:
        typer.echo(f"invalid input: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except (BudgetExhausted, LLMUnavailable) as exc:
        typer.echo(f"LLM unavailable: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    if as_json:
        typer.echo(response.model_dump_json(indent=2))
        return

    lat = response.meta.latency_ms
    typer.echo(f"{response.id}  language={response.input.language}  status={response.status}")
    if response.status == "rejected":
        typer.echo(f"rejected ({response.reject_reason}): {response.message}")
    for i, claim in enumerate(response.claims, start=1):
        flags = f"  flags={','.join(claim.flags)}" if claim.flags else ""
        typer.echo(f"\n[{i}] {claim.verdict}{flags}")
        typer.echo(_wrap(f"claim: {claim.claim_en}"))
        typer.echo(_wrap(f"reasoning: {claim.reasoning}"))
        if not claim.citations:
            typer.echo("    citations: none")
        for c in claim.citations:
            where = f"{c.doc_id} p.{c.page}" + (
                f" (printed {c.printed_page})" if c.printed_page else ""
            )
            if c.table_id:
                where += f" {c.table_id}"
            typer.echo(_wrap(f'- {where}: "{c.quote}"', indent="    "))
            typer.echo(f"      {c.source_url}")
        if evidence:
            for p in claim.evidence:
                mark = "*" if p.cited else " "
                typer.echo(
                    f"   {mark}{p.rank}. {p.doc_id} p.{p.page_start}-{p.page_end} {p.kind} "
                    f"ranks={p.ranks} rrf={p.rrf:.4f}"
                )
    typer.echo(
        f"\nmodels: {response.meta.models}  cached={response.meta.cached}\n"
        f"latency ms: extract={lat.extract} retrieve={lat.retrieve} verify={lat.verify} "
        f"total={lat.total}  (startup {load_s:.1f} s)"
    )
