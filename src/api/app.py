from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.schemas import HealthResponse, HistoryItem, PredictionRequest, PredictionResponse, ReadyResponse
from src.db.crud import create_prediction_log, latest_predictions
from src.db.models import Base
from src.db.session import engine, get_session
from src.models.predict import ModelRegistry
from src.utils.config import project_root
from src.utils.io import read_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Diabetes Risk Intelligence API",
    version="0.1.0",
    description="Ensemble ML API for diabetes risk prediction.",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:8510").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry: ModelRegistry | None = None


@app.on_event("startup")
def startup() -> None:
    global registry
    Base.metadata.create_all(bind=engine)
    try:
        registry = ModelRegistry()
        logger.info("Model registry loaded with models: %s", registry.available_models())
    except FileNotFoundError:
        registry = None
        logger.warning("Model artifacts are missing. API will report not ready.")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_type": exc.__class__.__name__},
    )


def get_registry() -> ModelRegistry:
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are missing. Run `make train` before starting the API.",
        )
    return registry


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if registry is None:
        return HealthResponse(status="healthy", model_registry="missing")
    return HealthResponse(
        status="healthy",
        database="not_checked",
        model_registry="ok",
        available_models=registry.available_models(),
        default_model=registry.default_model(),
    )


@app.get("/ready", response_model=ReadyResponse)
def ready(model_registry: ModelRegistry = Depends(get_registry)) -> ReadyResponse:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return ReadyResponse(
        status="ready",
        database="ok",
        model_registry="ok",
        available_models=model_registry.available_models(),
        default_model=model_registry.default_model(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    session: Session = Depends(get_session),
    model_registry: ModelRegistry = Depends(get_registry),
) -> PredictionResponse:
    payload = request.features.model_dump()
    try:
        result = model_registry.predict_one(payload, request.model_name, request.threshold_policy)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    create_prediction_log(session, payload, result)
    logger.info(
        "prediction_logged model=%s probability=%.4f risk=%s",
        result["model_name"],
        result["probability"],
        result["risk_label"],
    )
    return PredictionResponse(**result)


@app.get("/history", response_model=list[HistoryItem])
def history(limit: int = 50, session: Session = Depends(get_session)) -> list[HistoryItem]:
    rows = latest_predictions(session, limit=limit)
    return [
        HistoryItem(
            id=row.id,
            created_at=row.created_at,
            **json.loads(row.response_json),
        )
        for row in rows
    ]


@app.get("/model/metrics/{model_name}")
def model_metrics(model_name: str) -> dict:
    path = project_root() / "artifacts" / "reports" / f"metrics_{model_name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Metrics not found for model '{model_name}'")
    return read_json(path)


@app.get("/model/comparison")
def model_comparison() -> list[dict]:
    path = project_root() / "artifacts" / "reports" / "model_comparison.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model comparison is missing.")
    return pd.read_csv(path).to_dict(orient="records")


@app.get("/dataset/preview")
def dataset_preview(limit: int = 100) -> list[dict]:
    path = project_root() / "artifacts" / "dataset_preview.csv"
    if not path.exists():
        path = project_root() / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail="Dataset preview is missing.")
    return pd.read_csv(path).head(limit).to_dict(orient="records")


@app.get("/artifacts/manifest")
def manifest() -> dict:
    return read_json(project_root() / "artifacts" / "manifest.json")
