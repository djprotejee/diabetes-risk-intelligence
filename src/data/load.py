from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import resolve_path
from src.utils.io import ensure_dir


EXPECTED_COLUMNS = [
    "Diabetes_binary",
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]


def load_local_csv(path: str | Path) -> pd.DataFrame | None:
    target = resolve_path(path)
    if target.exists():
        return pd.read_csv(target)
    return None


def fetch_from_uci(uci_id: int, output_path: str | Path) -> pd.DataFrame:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "Dataset file was not found locally and ucimlrepo is not installed. "
            "Place the CDC CSV into data/raw or install ucimlrepo."
        ) from exc

    dataset = fetch_ucirepo(id=uci_id)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    target_name = targets.columns[0]
    df = pd.concat([targets[target_name], features], axis=1)
    target = resolve_path(output_path)
    ensure_dir(target.parent)
    df.to_csv(target, index=False)
    return df


def load_diabetes_data(config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    for key in ("local_csv", "archive_csv", "fallback_csv"):
        df = load_local_csv(data_cfg[key])
        if df is not None:
            return normalize_columns(df)
    return normalize_columns(fetch_from_uci(data_cfg["source"]["uci_id"], data_cfg["fallback_csv"]))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: column.strip() for column in df.columns}
    df = df.rename(columns=renamed)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df[EXPECTED_COLUMNS].copy()
