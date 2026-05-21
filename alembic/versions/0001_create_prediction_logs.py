"""create prediction logs

Revision ID: 0001_create_prediction_logs
Revises:
Create Date: 2026-05-21
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0001_create_prediction_logs"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "prediction_logs" in inspector.get_table_names():
        return
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("model_name", sa.String(length=80), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("prediction", sa.Integer(), nullable=False),
        sa.Column("risk_label", sa.String(length=40), nullable=False),
        sa.Column("request_json", sa.Text(), nullable=False),
        sa.Column("response_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_prediction_logs_created_at", "prediction_logs", ["created_at"])
    op.create_index("ix_prediction_logs_id", "prediction_logs", ["id"])
    op.create_index("ix_prediction_logs_model_name", "prediction_logs", ["model_name"])
    op.create_index("ix_prediction_logs_risk_label", "prediction_logs", ["risk_label"])


def downgrade() -> None:
    op.drop_index("ix_prediction_logs_risk_label", table_name="prediction_logs")
    op.drop_index("ix_prediction_logs_model_name", table_name="prediction_logs")
    op.drop_index("ix_prediction_logs_id", table_name="prediction_logs")
    op.drop_index("ix_prediction_logs_created_at", table_name="prediction_logs")
    op.drop_table("prediction_logs")
