from __future__ import annotations

import pandas as pd


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bmi_obese"] = (out["BMI"] >= 30).astype(int)
    out["bmi_overweight"] = ((out["BMI"] >= 25) & (out["BMI"] < 30)).astype(int)
    out["cardio_metabolic_load"] = (
        out["HighBP"] + out["HighChol"] + out["HeartDiseaseorAttack"] + out["Stroke"]
    )
    out["healthy_behavior_score"] = (
        out["PhysActivity"] + out["Fruits"] + out["Veggies"] + out["HvyAlcoholConsump"].rsub(1)
    )
    out["care_access_risk"] = (
        out["NoDocbcCost"] + out["AnyHealthcare"].rsub(1) + out["CholCheck"].rsub(1)
    )
    out["mobility_health_burden"] = out["DiffWalk"] + (out["PhysHlth"] > 14).astype(int)
    out["mental_physical_burden"] = (out["MentHlth"] + out["PhysHlth"]).clip(0, 60)
    out["age_bmi_interaction"] = out["Age"] * out["BMI"]
    out["general_health_burden"] = out["GenHlth"] + out["DiffWalk"] + out["PhysHlth"] / 30
    return out


def prepare_features(df: pd.DataFrame, add_features: bool = True) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.fillna(out.median(numeric_only=True))
    if add_features:
        out = add_domain_features(out)
    return out

