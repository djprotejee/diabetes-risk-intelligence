from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.features.engineering import prepare_features
from src.utils.config import project_root
from src.utils.io import load_joblib, read_json


class ModelRegistry:
    def __init__(self, artifacts_dir: str | Path = "artifacts") -> None:
        self.root = project_root()
        self.artifacts_dir = self.root / artifacts_dir
        self.manifest = read_json(self.artifacts_dir / "manifest.json")
        self._cache: dict[str, Any] = {}

    def available_models(self) -> list[str]:
        return sorted(self.manifest["models"].keys())

    def default_model(self) -> str:
        default = self.manifest.get("default_model", "stacking")
        if default in self.manifest["models"]:
            return default
        return self.available_models()[0]

    def load(self, model_name: str | None = None) -> Any:
        name = model_name or self.default_model()
        if name not in self.manifest["models"]:
            raise KeyError(f"Unknown model '{name}'. Available: {self.available_models()}")
        if name not in self._cache:
            self._cache[name] = load_joblib(self.root / self.manifest["models"][name]["artifact"])
        return self._cache[name]

    def threshold(self, model_name: str | None = None) -> float:
        name = model_name or self.default_model()
        return float(self.manifest["models"][name]["threshold"])

    def policy_threshold(self, model_name: str, threshold_policy: str = "balanced_f1") -> float:
        metrics_path = self.root / self.manifest["models"][model_name]["metrics"]
        try:
            metrics = read_json(metrics_path)
            return float(metrics["thresholds"][threshold_policy]["threshold"])
        except (FileNotFoundError, KeyError):
            return self.threshold(model_name)

    def predict_one(
        self,
        payload: dict[str, Any],
        model_name: str | None = None,
        threshold_policy: str = "balanced_f1",
    ) -> dict[str, Any]:
        name = model_name or self.default_model()
        model = self.load(name)
        frame = pd.DataFrame([payload])
        features = prepare_features(frame, add_features=True)
        probability = float(model.predict_proba(features)[:, 1][0])
        threshold = self.policy_threshold(name, threshold_policy)
        prediction = int(probability >= threshold)
        return {
            "model_name": name,
            "threshold_policy": threshold_policy,
            "probability": probability,
            "prediction": prediction,
            "threshold": threshold,
            "risk_label": risk_label(probability),
        }


def risk_label(probability: float) -> str:
    if probability < 0.25:
        return "low"
    if probability < 0.50:
        return "moderate"
    if probability < 0.75:
        return "high"
    return "critical"
