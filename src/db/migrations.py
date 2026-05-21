from __future__ import annotations

import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from src.db.models import Base
from src.db.session import engine
from src.utils.config import project_root


def run_migrations() -> None:
    if os.getenv("RUN_MIGRATIONS", "true").lower() not in {"1", "true", "yes"}:
        return
    root = project_root()
    alembic_cfg = Config(str(root / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(root / "alembic"))
    Path(root / "artifacts").mkdir(parents=True, exist_ok=True)
    with engine.connect() as connection:
        inspector = inspect(connection)
        tables = set(inspector.get_table_names())
        if "prediction_logs" in tables and "alembic_version" not in tables:
            command.stamp(alembic_cfg, "head")
            return
    command.upgrade(alembic_cfg, "head")


def ensure_schema() -> None:
    try:
        run_migrations()
    except Exception:
        Base.metadata.create_all(bind=engine)
