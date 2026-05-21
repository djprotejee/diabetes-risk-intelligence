from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load import load_diabetes_data
from src.data.split import stratified_train_valid_test
from src.features.engineering import prepare_features
from src.models.metrics import (
    evaluate_classifier,
    pick_threshold,
    plot_reliability,
    plot_roc_pr,
    threshold_candidates,
)
from src.utils.config import load_yaml, project_root, resolve_path
from src.utils.io import ensure_dir, save_joblib, write_json


def build_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_models(params: dict[str, dict[str, Any]], enabled: dict[str, dict[str, Any]]):
    models: dict[str, Any] = {}
    if enabled["logistic_regression"]["enabled"]:
        models["logistic_regression"] = LogisticRegression(**params["logistic_regression"])
    if enabled["random_forest"]["enabled"]:
        models["random_forest"] = RandomForestClassifier(**params["random_forest"])
    if enabled["extra_trees"]["enabled"]:
        models["extra_trees"] = ExtraTreesClassifier(**params["extra_trees"])
    if enabled["hist_gradient_boosting"]["enabled"]:
        models["hist_gradient_boosting"] = HistGradientBoostingClassifier(
            **params["hist_gradient_boosting"]
        )
    if enabled["lightgbm"]["enabled"]:
        try:
            from lightgbm import LGBMClassifier

            models["lightgbm"] = LGBMClassifier(**params["lightgbm"])
        except ImportError:
            print("[WARN] LightGBM is not installed. Skipping lightgbm.")
    if enabled["xgboost"]["enabled"]:
        try:
            from xgboost import XGBClassifier

            models["xgboost"] = XGBClassifier(**params["xgboost"])
        except ImportError:
            print("[WARN] XGBoost is not installed. Skipping xgboost.")
    return models


def build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])


def feature_names(X: pd.DataFrame) -> list[str]:
    return X.columns.tolist()


def save_feature_importance(model_name: str, pipeline: Pipeline, features: list[str], output_dir) -> None:
    estimator = pipeline.named_steps["model"]
    importances = None
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importances = np.abs(np.asarray(estimator.coef_).ravel())
    if importances is None or len(importances) != len(features):
        return
    df = pd.DataFrame({"feature": features, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    df.to_csv(output_dir / f"feature_importance_{model_name}.csv", index=False)
    top = df.head(20).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"])
    plt.title(f"Top feature importance - {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"feature_importance_{model_name}.png", dpi=140)
    plt.close()


def save_error_slices(
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    output_dir,
) -> None:
    frame = X_test.copy()
    frame["actual"] = y_test.to_numpy()
    frame["probability"] = y_prob
    frame["predicted"] = (y_prob >= threshold).astype(int)
    frame["correct"] = frame["actual"] == frame["predicted"]
    slices = []
    for column in ["Age", "BMI", "HighBP", "HighChol", "Income", "Education", "Sex"]:
        if column not in frame.columns:
            continue
        if column == "BMI":
            groups = pd.cut(frame[column], bins=[0, 25, 30, 35, 100], include_lowest=True)
        else:
            groups = frame[column]
        grouped = frame.groupby(groups, observed=False)
        for value, part in grouped:
            if len(part) < 20:
                continue
            slices.append(
                {
                    "model": model_name,
                    "slice_column": column,
                    "slice_value": str(value),
                    "n": int(len(part)),
                    "positive_rate": float(part["actual"].mean()),
                    "mean_probability": float(part["probability"].mean()),
                    "accuracy": float(part["correct"].mean()),
                    "false_negative_rate": float(
                        ((part["actual"] == 1) & (part["predicted"] == 0)).sum()
                        / max((part["actual"] == 1).sum(), 1)
                    ),
                }
            )
    pd.DataFrame(slices).to_csv(output_dir / f"error_slices_{model_name}.csv", index=False)


def save_shap_summary(
    model_name: str,
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    output_dir,
) -> None:
    try:
        import shap
    except ImportError:
        print("[WARN] SHAP is not installed. Skipping explainability plots.")
        return
    estimator = pipeline.named_steps["model"]
    if model_name in {"stacking", "soft_voting"}:
        return
    X_transformed = pipeline.named_steps["preprocessor"].transform(X_sample)
    try:
        if hasattr(estimator, "feature_importances_"):
            explainer = shap.TreeExplainer(estimator)
            values = explainer.shap_values(X_transformed)
            if isinstance(values, list):
                values = values[-1]
        elif hasattr(estimator, "coef_"):
            explainer = shap.LinearExplainer(estimator, X_transformed)
            values = explainer.shap_values(X_transformed)
        else:
            return
        plt.figure()
        shap.summary_plot(values, X_transformed, feature_names=X_sample.columns, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_summary_{model_name}.png", dpi=140, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"[WARN] SHAP failed for {model_name}: {exc}")


FAST_MODELS = {"logistic_regression", "hist_gradient_boosting", "lightgbm"}
ENSEMBLE_MODELS = {"soft_voting", "stacking"}


def select_model_names(
    all_names: list[str],
    requested_model: str | None,
    mode: str,
) -> list[str]:
    if requested_model:
        if requested_model not in all_names and requested_model not in ENSEMBLE_MODELS:
            raise ValueError(f"Unknown model '{requested_model}'. Available models: {all_names + sorted(ENSEMBLE_MODELS)}")
        return [requested_model]
    if mode == "fast":
        return [name for name in all_names if name in FAST_MODELS]
    return all_names


def train(
    config_path: str = "configs/config.yaml",
    requested_model: str | None = None,
    mode: str = "full",
    skip_explain: bool = False,
    skip_ensembles: bool = False,
) -> None:
    root = project_root()
    cfg = load_yaml(config_path)
    params = load_yaml("configs/model_params.yaml")
    artifacts_dir = ensure_dir(root / cfg["serving"]["artifacts_dir"])
    models_dir = ensure_dir(artifacts_dir / "models")
    reports_dir = ensure_dir(artifacts_dir / "reports")

    raw_df = load_diabetes_data(cfg)
    target = cfg["data"]["target"]
    prepared = prepare_features(raw_df, add_features=cfg["features"]["add_domain_features"])
    X_train, X_valid, X_test, y_train, y_valid, y_test = stratified_train_valid_test(
        prepared,
        target=target,
        test_size=cfg["data"]["split"]["test_size"],
        validation_size=cfg["data"]["split"]["validation_size"],
        random_state=cfg["project"]["random_state"],
    )
    X_train.to_csv(resolve_path("data/processed/train_features.csv"), index=False)
    X_valid.to_csv(resolve_path("data/processed/valid_features.csv"), index=False)
    X_test.to_csv(resolve_path("data/processed/test_features.csv"), index=False)

    enabled = cfg["training"]["models"]
    base_estimators = build_models(params, enabled)
    selected_names = select_model_names(list(base_estimators.keys()), requested_model, mode)
    train_base_names = [name for name in selected_names if name not in ENSEMBLE_MODELS]
    train_ensemble_names = [name for name in selected_names if name in ENSEMBLE_MODELS]
    if mode == "fast":
        skip_explain = True
        skip_ensembles = True
    base_estimators = {name: estimator for name, estimator in base_estimators.items() if name in train_base_names}
    trained: dict[str, Pipeline] = {}
    manifest: dict[str, Any] = {
        "dataset": {
            "source": cfg["data"]["source"],
            "rows": int(len(prepared)),
            "features": int(X_train.shape[1]),
            "target": target,
            "positive_rate": float(prepared[target].mean()),
        },
        "models": {},
        "default_model": cfg["serving"]["default_model"],
    }

    for name, estimator in base_estimators.items():
        print(f"[INFO] Training {name}")
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline
        y_valid_prob = pipeline.predict_proba(X_valid)[:, 1]
        thresholds = pick_threshold(
            y_valid.to_numpy(),
            y_valid_prob,
            recall_target=cfg["training"]["recall_target"],
        )
        selected_threshold = thresholds[cfg["training"]["threshold_policy"]]["threshold"]
        y_test_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "validation": evaluate_classifier(y_valid.to_numpy(), y_valid_prob, selected_threshold),
            "test": evaluate_classifier(y_test.to_numpy(), y_test_prob, selected_threshold),
            "thresholds": thresholds,
        }
        save_joblib(models_dir / f"{name}.joblib", pipeline)
        write_json(reports_dir / f"metrics_{name}.json", metrics)
        threshold_candidates(y_valid.to_numpy(), y_valid_prob).to_csv(
            reports_dir / f"threshold_grid_{name}.csv", index=False
        )
        pd.DataFrame(
            {
                "actual": y_test.to_numpy(),
                "probability": y_test_prob,
                "prediction": (y_test_prob >= selected_threshold).astype(int),
            }
        ).to_csv(reports_dir / f"predictions_test_{name}.csv", index=False)
        plot_roc_pr(y_test.to_numpy(), y_test_prob, reports_dir, f"{name}_test")
        plot_reliability(y_test.to_numpy(), y_test_prob, reports_dir, f"{name}_test")
        save_feature_importance(name, pipeline, feature_names(X_train), reports_dir)
        save_error_slices(name, X_test, y_test, y_test_prob, selected_threshold, reports_dir)
        if not skip_explain:
            save_shap_summary(
                name,
                pipeline,
                X_valid.sample(
                    min(cfg["training"]["shap_sample_size"], len(X_valid)),
                    random_state=cfg["project"]["random_state"],
                ),
                reports_dir,
            )
        manifest["models"][name] = {
            "artifact": f"artifacts/models/{name}.joblib",
            "metrics": f"artifacts/reports/metrics_{name}.json",
            "threshold": float(selected_threshold),
            "test_roc_auc": metrics["test"]["roc_auc"],
            "test_pr_auc": metrics["test"]["pr_auc"],
            "test_f1": metrics["test"]["f1"],
            "test_recall": metrics["test"]["recall"],
        }

    ensemble_members = [
        (name, pipeline)
        for name, pipeline in trained.items()
        if name in {"logistic_regression", "random_forest", "extra_trees", "lightgbm", "xgboost"}
    ]

    should_train_soft_voting = (
        not skip_ensembles
        and enabled["soft_voting"]["enabled"]
        and len(ensemble_members) >= 3
        and (not train_ensemble_names or "soft_voting" in train_ensemble_names)
    )
    should_train_stacking = (
        not skip_ensembles
        and enabled["stacking"]["enabled"]
        and len(ensemble_members) >= 3
        and (not train_ensemble_names or "stacking" in train_ensemble_names)
    )

    if should_train_soft_voting:
        print("[INFO] Training soft_voting")
        voting = VotingClassifier(estimators=ensemble_members, voting="soft", n_jobs=cfg["training"]["n_jobs"])
        voting.fit(X_train, y_train)
        trained["soft_voting"] = voting

    if should_train_stacking:
        print("[INFO] Training stacking")
        stacking = StackingClassifier(
            estimators=ensemble_members,
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
            cv=cfg["training"]["cv_folds"],
            n_jobs=cfg["training"]["n_jobs"],
            stack_method="predict_proba",
        )
        stacking.fit(X_train, y_train)
        trained["stacking"] = stacking

    for name in ["soft_voting", "stacking"]:
        if name not in trained:
            continue
        pipeline = trained[name]
        y_valid_prob = pipeline.predict_proba(X_valid)[:, 1]
        thresholds = pick_threshold(
            y_valid.to_numpy(),
            y_valid_prob,
            recall_target=cfg["training"]["recall_target"],
        )
        selected_threshold = thresholds[cfg["training"]["threshold_policy"]]["threshold"]
        y_test_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "validation": evaluate_classifier(y_valid.to_numpy(), y_valid_prob, selected_threshold),
            "test": evaluate_classifier(y_test.to_numpy(), y_test_prob, selected_threshold),
            "thresholds": thresholds,
        }
        save_joblib(models_dir / f"{name}.joblib", pipeline)
        write_json(reports_dir / f"metrics_{name}.json", metrics)
        threshold_candidates(y_valid.to_numpy(), y_valid_prob).to_csv(
            reports_dir / f"threshold_grid_{name}.csv", index=False
        )
        plot_roc_pr(y_test.to_numpy(), y_test_prob, reports_dir, f"{name}_test")
        plot_reliability(y_test.to_numpy(), y_test_prob, reports_dir, f"{name}_test")
        save_error_slices(name, X_test, y_test, y_test_prob, selected_threshold, reports_dir)
        manifest["models"][name] = {
            "artifact": f"artifacts/models/{name}.joblib",
            "metrics": f"artifacts/reports/metrics_{name}.json",
            "threshold": float(selected_threshold),
            "test_roc_auc": metrics["test"]["roc_auc"],
            "test_pr_auc": metrics["test"]["pr_auc"],
            "test_f1": metrics["test"]["f1"],
            "test_recall": metrics["test"]["recall"],
        }

    finalize_artifacts(artifacts_dir, cfg, manifest)
    raw_df.head(1000).to_csv(artifacts_dir / "dataset_preview.csv", index=False)
    print(json.dumps(manifest, indent=2))


def read_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finalize_artifacts(
    artifacts_dir: Path | None = None,
    cfg: dict[str, Any] | None = None,
    current_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = project_root()
    cfg = cfg or load_yaml("configs/config.yaml")
    artifacts_dir = artifacts_dir or root / cfg["serving"]["artifacts_dir"]
    reports_dir = artifacts_dir / "reports"
    models_dir = artifacts_dir / "models"
    existing_manifest_path = artifacts_dir / "manifest.json"
    manifest = current_manifest or {
        "dataset": {},
        "models": {},
        "default_model": cfg["serving"]["default_model"],
    }
    if existing_manifest_path.exists() and current_manifest is None:
        manifest = read_metrics(existing_manifest_path)

    rows = []
    for metrics_path in sorted(reports_dir.glob("metrics_*.json")):
        name = metrics_path.stem.replace("metrics_", "", 1)
        metrics = read_metrics(metrics_path)
        if not (models_dir / f"{name}.joblib").exists():
            continue
        manifest.setdefault("models", {})[name] = {
            "artifact": f"artifacts/models/{name}.joblib",
            "metrics": f"artifacts/reports/metrics_{name}.json",
            "threshold": metrics["test"]["threshold"],
            "test_roc_auc": metrics["test"]["roc_auc"],
            "test_pr_auc": metrics["test"]["pr_auc"],
            "test_f1": metrics["test"]["f1"],
            "test_recall": metrics["test"]["recall"],
        }
        rows.append(
            {
                "model": name,
                **{
                    f"test_{key}": value
                    for key, value in metrics["test"].items()
                    if not isinstance(value, dict)
                },
            }
        )
    if rows:
        comparison = pd.DataFrame(rows).sort_values("test_pr_auc", ascending=False)
        comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    write_json(artifacts_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--mode", choices=["fast", "full"], default="full")
    parser.add_argument("--skip-explain", action="store_true")
    parser.add_argument("--skip-ensembles", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.finalize_only:
        finalize_artifacts()
    else:
        train(
            args.config,
            requested_model=args.model,
            mode=args.mode,
            skip_explain=args.skip_explain,
            skip_ensembles=args.skip_ensembles,
        )


if __name__ == "__main__":
    main()
