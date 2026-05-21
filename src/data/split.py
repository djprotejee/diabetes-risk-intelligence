from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_train_valid_test(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    relative_valid = validation_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=relative_valid,
        stratify=y_train_valid,
        random_state=random_state,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test

