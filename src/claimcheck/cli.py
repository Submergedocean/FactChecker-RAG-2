"""Root `claimcheck` CLI. Only mounts the sub-apps; commands live in each package's cli.py."""

import typer

from claimcheck.api.cli import app as api_app
from claimcheck.claims.cli import app as claims_app
from claimcheck.eval.cli import app as eval_app
from claimcheck.index.cli import app as index_app
from claimcheck.ingest.cli import app as ingest_app
from claimcheck.pipeline.cli import app as pipeline_app
from claimcheck.retrieval.cli import app as retrieval_app

app = typer.Typer(help="Claim Checker (PES RAG).", no_args_is_help=True)

SUB_APPS: dict[str, typer.Typer] = {
    "ingest": ingest_app,
    "index": index_app,
    "retrieval": retrieval_app,
    "pipeline": pipeline_app,
    "claims": claims_app,
    "eval": eval_app,
    "api": api_app,
}

for _name, _sub in SUB_APPS.items():
    app.add_typer(_sub, name=_name)


if __name__ == "__main__":
    app()
