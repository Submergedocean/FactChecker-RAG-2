"""`claimcheck api` commands: `serve` runs the FastAPI app with one Uvicorn worker."""

from __future__ import annotations

import logging

import typer

app = typer.Typer(
    help="FastAPI backend server.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """FastAPI backend server."""


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind address (default: api.host)."),
    port: int | None = typer.Option(None, help="Port (default: api.port)."),
    log_level: str = typer.Option("info", help="Uvicorn and app log level."),
) -> None:
    """Serve the API (and the built SPA, if any) with one worker."""
    import uvicorn

    from claimcheck.api.main import create_app
    from claimcheck.api.security import UnsafeConfigError, check_startup
    from claimcheck.config import get_settings

    settings = get_settings()
    try:
        check_startup(settings)
    except UnsafeConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    logging.basicConfig(
        level=log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    uvicorn.run(
        create_app(settings),
        host=host or settings.api.host,
        port=port or settings.api.port,
        workers=1,
        log_level=log_level,
        proxy_headers=False,  # client identity is resolved in the app
        access_log=False,  # the access log carries client addresses; no client identity is logged
    )
