"""`claimcheck eval` commands. Stub: no commands yet."""

import typer

app = typer.Typer(
    help="Evaluation, baselines and ablations.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Evaluation, baselines and ablations."""
