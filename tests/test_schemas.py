from __future__ import annotations

from src.api.schemas import PatientFeatures, PredictionRequest


def test_prediction_request_defaults_are_valid() -> None:
    request = PredictionRequest(features=PatientFeatures())
    assert request.features.BMI == 31.0
    assert request.features.Age == 9

