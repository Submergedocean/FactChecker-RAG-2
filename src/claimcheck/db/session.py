"""SQLite engines and sessions.

Every connection gets `journal_mode=WAL`, `foreign_keys=ON` and `busy_timeout=5000`. The app
database is `paths.app_db`; evaluation uses `paths.eval_cache_db` (same schema, only its
`llm_cache` and `llm_usage` tables are written).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from claimcheck.db.models import Base

BUSY_TIMEOUT_MS = 5000


def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    finally:
        cursor.close()


def make_engine(path: Path | str, *, create: bool = True) -> Engine:
    """Engine for the SQLite file at `path`, PRAGMAs set on every connection.

    With `create` the parent directory and all five tables are created if missing
    (`CREATE TABLE IF NOT EXISTS`; existing tables are left as they are).
    """
    path = Path(path)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{path}",
        # FastAPI runs sync handlers in a thread pool; each session still uses one thread.
        connect_args={"check_same_thread": False, "timeout": BUSY_TIMEOUT_MS / 1000},
    )
    event.listen(engine, "connect", _set_pragmas)
    if create:
        Base.metadata.create_all(engine)
    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """One transaction: commit on success, roll back on error."""
    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
