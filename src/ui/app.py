from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
ROOT = Path(__file__).resolve().parents[2]


def api_get(path: str):
    response = requests.get(f"{API_URL}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30)
def load_health() -> dict:
    return api_get("/ready")


@st.cache_data(ttl=60)
def load_comparison() -> pd.DataFrame:
    return pd.DataFrame(api_get("/model/comparison"))


@st.cache_data(ttl=60)
def load_dataset(limit: int) -> pd.DataFrame:
    return pd.DataFrame(api_get(f"/dataset/preview?limit={limit}"))


def patient_form(models: list[str], default_model: str) -> tuple[str, str, dict, bool]:
    with st.sidebar:
        st.header("Patient Profile")
        model = st.selectbox(
            "Model",
            models,
            index=models.index(default_model) if default_model in models else 0,
        )
        threshold_policy = st.radio(
            "Operating Mode",
            ["balanced_f1", "recall_oriented"],
            format_func=lambda value: "Balanced F1" if value == "balanced_f1" else "High Recall",
            horizontal=True,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            high_bp = st.toggle("High BP", value=True)
            high_chol = st.toggle("High Chol", value=True)
            smoker = st.toggle("Smoker")
            stroke = st.toggle("Stroke")
            heart = st.toggle("Heart Disease")
            diff_walk = st.toggle("Diff Walk")
        with col_b:
            chol_check = st.toggle("Chol Check", value=True)
            activity = st.toggle("Activity", value=True)
            fruits = st.toggle("Fruits", value=True)
            veggies = st.toggle("Veggies", value=True)
            alcohol = st.toggle("Heavy Alcohol")
            healthcare = st.toggle("Healthcare", value=True)
        bmi = st.slider("BMI", 10.0, 60.0, 31.0, 0.5)
        general_health = st.slider("General Health", 1, 5, 3)
        mental = st.slider("Mental Health Days", 0, 30, 2)
        physical = st.slider("Physical Health Days", 0, 30, 3)
        age = st.slider("Age Category", 1, 13, 9)
        education = st.slider("Education", 1, 6, 5)
        income = st.slider("Income", 1, 8, 6)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        no_doc = st.toggle("No Doctor Because Cost")
        submit = st.button("Analyze Risk", type="primary", use_container_width=True)
    features = {
        "HighBP": int(high_bp),
        "HighChol": int(high_chol),
        "CholCheck": int(chol_check),
        "BMI": bmi,
        "Smoker": int(smoker),
        "Stroke": int(stroke),
        "HeartDiseaseorAttack": int(heart),
        "PhysActivity": int(activity),
        "Fruits": int(fruits),
        "Veggies": int(veggies),
        "HvyAlcoholConsump": int(alcohol),
        "AnyHealthcare": int(healthcare),
        "NoDocbcCost": int(no_doc),
        "GenHlth": general_health,
        "MentHlth": mental,
        "PhysHlth": physical,
        "DiffWalk": int(diff_walk),
        "Sex": 1 if sex == "Male" else 0,
        "Age": age,
        "Education": education,
        "Income": income,
    }
    return model, threshold_policy, features, submit


def prediction_view() -> None:
    try:
        health = load_health()
    except Exception as exc:
        st.error(f"API is unavailable: {exc}")
        return
    available_models = health.get("available_models") or []
    default_model = health.get("default_model")
    if not available_models or not default_model:
        st.error("API is reachable, but model registry metadata is missing. Restart API after training artifacts are available.")
        st.json(health)
        return
    model, threshold_policy, features, submit = patient_form(available_models, default_model)
    st.title("Diabetes Risk Intelligence")
    st.caption("Ensemble ML PoC with calibrated risk scores, model comparison, and prediction logging.")
    if submit:
        result = api_post(
            "/predict",
            {"model_name": model, "threshold_policy": threshold_policy, "features": features},
        )
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Risk Probability", f"{result['probability']:.1%}")
        col2.metric("Decision", "Risk" if result["prediction"] else "No Risk")
        col3.metric("Risk Segment", result["risk_label"].title())
        col4.metric("Threshold", f"{result['threshold']:.2f}")
        st.caption(f"Operating mode: {result['threshold_policy']}")
        st.progress(min(max(result["probability"], 0), 1))
        st.dataframe(pd.DataFrame([features]), use_container_width=True)
    else:
        st.info("Set patient profile parameters in the sidebar and run analysis.")


def model_lab_view() -> None:
    st.title("Model Lab")
    try:
        comparison = load_comparison()
    except Exception as exc:
        st.error(f"Model comparison is unavailable: {exc}")
        return
    st.dataframe(comparison, use_container_width=True)
    metric_cols = [col for col in comparison.columns if col.startswith("test_") and col != "test_threshold"]
    metric = st.selectbox("Metric", metric_cols, index=metric_cols.index("test_pr_auc") if "test_pr_auc" in metric_cols else 0)
    fig = px.bar(comparison, x="model", y=metric, color=metric, title=f"Model comparison by {metric}")
    st.plotly_chart(fig, use_container_width=True)
    model = st.selectbox("Inspect model", comparison["model"].tolist())
    metrics = api_get(f"/model/metrics/{model}")
    st.json(metrics)
    artifact_dir = ROOT / "artifacts" / "reports"
    images = [
        artifact_dir / f"roc_curve_{model}_test.png",
        artifact_dir / f"pr_curve_{model}_test.png",
        artifact_dir / f"reliability_{model}_test.png",
        artifact_dir / f"feature_importance_{model}.png",
        artifact_dir / f"shap_summary_{model}.png",
    ]
    for image in images:
        if image.exists():
            st.image(str(image), caption=image.name, use_column_width=True)


def dataset_view() -> None:
    st.title("Dataset Explorer")
    limit = st.slider("Rows", 100, 1000, 300, 100)
    try:
        df = load_dataset(limit)
    except Exception as exc:
        st.error(f"Dataset preview is unavailable: {exc}")
        return
    st.dataframe(df, use_container_width=True)
    if "Diabetes_binary" in df.columns:
        fig = px.histogram(df, x="Diabetes_binary", title="Target distribution")
        st.plotly_chart(fig, use_container_width=True)
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        st.dataframe(numeric.describe().T, use_container_width=True)
        selected = st.multiselect("Numeric feature distributions", numeric.columns.tolist(), default=numeric.columns[:4].tolist())
        for column in selected:
            st.plotly_chart(px.histogram(df, x=column, title=column), use_container_width=True)


def history_view() -> None:
    st.title("Prediction History")
    try:
        data = api_get("/history?limit=100")
    except Exception as exc:
        st.error(f"History is unavailable: {exc}")
        return
    if not data:
        st.info("No logged predictions yet.")
        return
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
    fig = px.histogram(df, x="risk_label", color="prediction", title="Logged risk segments")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Diabetes Risk Intelligence", layout="wide")
    tabs = st.tabs(["Predict", "Model Lab", "Dataset", "History"])
    with tabs[0]:
        prediction_view()
    with tabs[1]:
        model_lab_view()
    with tabs[2]:
        dataset_view()
    with tabs[3]:
        history_view()


if __name__ == "__main__":
    main()
