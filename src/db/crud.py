from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import PredictionLog


def create_prediction_log(
    session: Session,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
) -> PredictionLog:
    row = PredictionLog(
        model_name=response_payload["model_name"],
        probability=response_payload["probability"],
        threshold=response_payload["threshold"],
        prediction=response_payload["prediction"],
        risk_label=response_payload["risk_label"],
        request_json=json.dumps(request_payload, ensure_ascii=False),
        response_json=json.dumps(response_payload, ensure_ascii=False),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def latest_predictions(session: Session, limit: int = 50) -> list[PredictionLog]:
    statement = select(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(limit)
    return list(session.scalars(statement))

