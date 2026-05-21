from __future__ import annotations

import pandas as pd

from src.features.engineering import prepare_features


def test_prepare_features_adds_domain_features() -> None:
    frame = pd.DataFrame(
        [
            {
                "HighBP": 1,
                "HighChol": 1,
                "CholCheck": 1,
                "BMI": 32,
                "Smoker": 0,
                "Stroke": 0,
                "HeartDiseaseorAttack": 1,
                "PhysActivity": 1,
                "Fruits": 1,
                "Veggies": 1,
                "HvyAlcoholConsump": 0,
                "AnyHealthcare": 1,
                "NoDocbcCost": 0,
                "GenHlth": 3,
                "MentHlth": 2,
                "PhysHlth": 5,
                "DiffWalk": 0,
                "Sex": 1,
                "Age": 9,
                "Education": 5,
                "Income": 6,
            }
        ]
    )
    prepared = prepare_features(frame)
    assert prepared.loc[0, "bmi_obese"] == 1
    assert prepared.loc[0, "cardio_metabolic_load"] == 3
    assert "age_bmi_interaction" in prepared.columns

