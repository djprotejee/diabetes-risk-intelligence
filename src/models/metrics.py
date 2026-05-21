from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def threshold_candidates(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "f1": f1_score(y_true, pred, zero_division=0),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, pred),
            }
        )
    return pd.DataFrame(rows)


def pick_threshold(y_true: np.ndarray, y_prob: np.ndarray, recall_target: float) -> dict[str, Any]:
    grid = threshold_candidates(y_true, y_prob)
    best_f1 = grid.sort_values(["f1", "balanced_accuracy"], ascending=False).iloc[0].to_dict()
    recall_ok = grid[grid["recall"] >= recall_target]
    if recall_ok.empty:
        recall_choice = grid.sort_values(["recall", "f1"], ascending=False).iloc[0].to_dict()
    else:
        recall_choice = recall_ok.sort_values(["precision", "f1"], ascending=False).iloc[0].to_dict()
    return {"balanced_f1": best_f1, "recall_oriented": recall_choice}


def evaluate_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path, prefix: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC={roc_auc_score(y_true, y_prob):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve - {prefix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"roc_curve_{prefix}.png", dpi=140)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR-AUC={average_precision_score(y_true, y_prob):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curve - {prefix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"pr_curve_{prefix}.png", dpi=140)
    plt.close()


def plot_reliability(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path, prefix: str) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title(f"Reliability curve - {prefix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"reliability_{prefix}.png", dpi=140)
    plt.close()

