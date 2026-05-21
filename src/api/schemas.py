from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from pydantic import ConfigDict


class APIModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class PatientFeatures(APIModel):
    HighBP: int = Field(1, ge=0, le=1)
    HighChol: int = Field(1, ge=0, le=1)
    CholCheck: int = Field(1, ge=0, le=1)
    BMI: float = Field(31.0, ge=10, le=80)
    Smoker: int = Field(0, ge=0, le=1)
    Stroke: int = Field(0, ge=0, le=1)
    HeartDiseaseorAttack: int = Field(0, ge=0, le=1)
    PhysActivity: int = Field(1, ge=0, le=1)
    Fruits: int = Field(1, ge=0, le=1)
    Veggies: int = Field(1, ge=0, le=1)
    HvyAlcoholConsump: int = Field(0, ge=0, le=1)
    AnyHealthcare: int = Field(1, ge=0, le=1)
    NoDocbcCost: int = Field(0, ge=0, le=1)
    GenHlth: int = Field(3, ge=1, le=5)
    MentHlth: int = Field(2, ge=0, le=30)
    PhysHlth: int = Field(3, ge=0, le=30)
    DiffWalk: int = Field(0, ge=0, le=1)
    Sex: int = Field(1, ge=0, le=1)
    Age: int = Field(9, ge=1, le=13)
    Education: int = Field(5, ge=1, le=6)
    Income: int = Field(6, ge=1, le=8)


class PredictionRequest(APIModel):
    model_name: str | None = None
    threshold_policy: Literal["balanced_f1", "recall_oriented"] = "balanced_f1"
    features: PatientFeatures


class PredictionResponse(APIModel):
    model_name: str
    threshold_policy: Literal["balanced_f1", "recall_oriented"]
    probability: float
    prediction: int
    threshold: float
    risk_label: Literal["low", "moderate", "high", "critical"]


class HistoryItem(PredictionResponse):
    id: int
    created_at: datetime


class HealthResponse(APIModel):
    status: str
    database: str | None = None
    model_registry: str | None = None
    available_models: list[str] = []
    default_model: str | None = None


class ReadyResponse(APIModel):
    status: str
    database: str
    model_registry: str
    available_models: list[str]
    default_model: str


class ErrorResponse(APIModel):
    detail: str
    error_type: str
